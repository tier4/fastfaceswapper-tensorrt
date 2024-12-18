#ifndef __TENSORRT_UTILS__INFERENCE_ENGINE_HPP__
#define __TENSORRT_UTILS__INFERENCE_ENGINE_HPP__

#include <NvInfer.h>

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <tensorrt_utils/tensorrt_utils.hpp>
#include <unordered_map>

namespace tensorrt_utils {

class InferenceEngine {
 public:
  // Constructor for InferenceEngine
  // Initializes the engine by deserializing the engine file and setting up the execution context
  InferenceEngine(const std::filesystem::path& enginePath, nvinfer1::ILogger& logger) {
    // Read the engine file using readEngineFile function
    auto engineDataOr = readEngineFile(enginePath);
    if (!engineDataOr.ok()) {
      throw std::runtime_error(std::string(engineDataOr.status().message()));
    }
    std::vector<std::uint8_t> engineData = std::move(engineDataOr).value();

    // Create the runtime and deserialize the engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
      throw std::runtime_error("Failed to create TensorRT runtime");
    }

    logger_ = &logger;
    auto engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
      throw std::runtime_error("Failed to deserialize CUDA engine");
    }
    logger_->log(nvinfer1::ILogger::Severity::kINFO,
                 absl::StrFormat("Deserialized engine from file: %s", enginePath).c_str());
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(engine);
    auto context = engine_->createExecutionContext();
    if (!context) {
      throw std::runtime_error("Failed to create execution context");
    }
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(context);
    context_->setOptimizationProfileAsync(0, *stream_);

    // Get min and max batch sizes for the engine
    const auto bsOr = getProfileBatchSizes(engine_.get(), 0);
    if (!bsOr.ok()) {
      throw std::runtime_error(std::string(bsOr.status().message()));
    }
    const auto [minBs, optBs, maxBs] = bsOr.value();
    minBs_ = minBs;
    maxBs_ = maxBs;
    optBs_ = optBs;
    logger_->log(
        nvinfer1::ILogger::Severity::kINFO,
        absl::StrFormat("Batch sizes: min: %d, opt: %d, max: %d", minBs_, optBs_, maxBs_).c_str());

    // Allocate memory for binding buffers
    auto status = allocateMemory();
    if (!status.ok()) {
      throw std::runtime_error(std::string(status.message()));
    }
  }

  absl::StatusOr<std::unordered_map<std::string, std::vector<std::uint8_t>>> infer(
      const std::unordered_map<std::string, std::vector<std::uint8_t>>& inputs) {
    // Set input tensors
    std::int32_t bs = -1;
    for (std::int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
      const auto tensorName = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
        const auto tensorNameStr = std::string(tensorName);
        if (inputs.find(tensorNameStr) == inputs.end()) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Missing input tensor: %s", tensorNameStr));
        }
        auto data = inputs.at(tensorNameStr);
        auto dims = engine_->getTensorShape(tensorName);
        auto dtype = engine_->getTensorDataType(tensorName);
        const auto elemSizeOr = dataTypeToSize(dtype);
        if (!elemSizeOr.ok()) {
          return absl::InternalError(elemSizeOr.status().message());
        }
        const auto frameSize = std::accumulate(dims.d + 1, dims.d + dims.nbDims, elemSizeOr.value(),
                                               std::multiplies<>());
        if (data.size() % frameSize != 0) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Invalid input size for tensor: %s, expected multiple of %d, got: %d",
                              tensorName, frameSize, data.size()));
        }
        if (bs == -1) {
          bs = data.size() / frameSize;
        }
        if (static_cast<std::size_t>(bs) != data.size() / frameSize) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Invalid batch size for tensor: %s, expected: %d, got: %d",
                              tensorName, bs, data.size() / frameSize));
        }
        if (data.size() > frameSize * maxBs_) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Invalid input size for tensor: %s, expected <= %d, got: %d",
                              tensorName, frameSize * maxBs_, data.size()));
        }
        std::copy(data.begin(), data.end(), hbuffs_.at(i).get());
        CHECK_CUDA_ERROR(cudaMemcpyAsync(dbuffs_.at(i).get(), hbuffs_.at(i).get(), data.size(),
                                         cudaMemcpyHostToDevice, *stream_));
      }
    }
    // Execute inference
    context_->enqueueV3(*stream_);

    // Device to host copy for output tensors
    std::unordered_map<std::string, std::vector<std::uint8_t>> outputs;
    for (std::int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
      const auto tensorName = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT) {
        const auto tensorNameStr = std::string(tensorName);
        auto dims = engine_->getTensorShape(tensorName);
        auto dtype = engine_->getTensorDataType(tensorName);
        const auto elemSizeOr = dataTypeToSize(dtype);
        if (!elemSizeOr.ok()) {
          return absl::InternalError(elemSizeOr.status().message());
        }
        const auto frameSize = std::accumulate(dims.d + 1, dims.d + dims.nbDims, elemSizeOr.value(),
                                               std::multiplies<>());
        CHECK_CUDA_ERROR(cudaMemcpyAsync(hbuffs_.at(i).get(), dbuffs_.at(i).get(), frameSize * bs,
                                         cudaMemcpyDeviceToHost, *stream_));
        outputs[tensorNameStr].resize(frameSize * bs);
      }
    }
    cudaStreamSynchronize(*stream_);

    for (std::int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
      const auto tensorName = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT) {
        const auto tensorNameStr = std::string(tensorName);
        std::copy(hbuffs_.at(i).get(), hbuffs_.at(i).get() + outputs[tensorNameStr].size(),
                  outputs[tensorNameStr].begin());
      }
    }
    return outputs;
  }

 private:
  // TensorRT engine
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  // Execution context for the engine
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  // Logger for TensorRT
  nvinfer1::ILogger* logger_;
  // Device buffers for input/output tensors
  std::vector<cuda_utils::CudaUniquePtr<std::uint8_t[]>> dbuffs_;
  // Host buffers for input/output tensors
  std::vector<cuda_utils::CudaUniquePtrHost<std::uint8_t[]>> hbuffs_;
  // CUDA stream for asynchronous operations
  cuda_utils::StreamUniquePtr stream_{cuda_utils::makeCudaStream()};
  // Minimum batch size for the engine
  std::int32_t minBs_;
  // Maximum batch size for the engine
  std::int32_t maxBs_;
  // Optimum batch size for the engine
  std::int32_t optBs_;

  // Sets the batch size for the engine
  // Returns an error if the batch size is invalid
  absl::Status setBs(const std::int32_t bs) {
    if (bs < minBs_ || bs > maxBs_) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid batch size: %d, min: %d, max: %d", bs, minBs_, maxBs_));
    }
    for (std::int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
      const auto tensorName = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
        auto dims = engine_->getTensorShape(tensorName);
        dims.d[0] = bs;
        if (!context_->setInputShape(tensorName, dims)) {
          return absl::InternalError(
              absl::StrFormat("Failed to set input shape for tensor: %s", tensorName));
        }
      }
    }
    return absl::OkStatus();
  }

  // Allocates memory for input and output tensors
  // Returns an error if memory allocation fails
  absl::Status allocateMemory() {
    // Allocate memory for input and output tensors
    std::int32_t nbBindings = engine_->getNbIOTensors();
    for (int i = 0; i < nbBindings; ++i) {
      const auto tensorName = engine_->getIOTensorName(i);
      const auto dims = engine_->getTensorShape(tensorName);
      const auto dtype = engine_->getTensorDataType(tensorName);
      const bool isInput = engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT;
      size_t size = maxBs_;
      for (std::int32_t j = 1; j < dims.nbDims; ++j) {
        size *= dims.d[j];
      }
      auto elemSizeOr = dataTypeToSize(dtype);
      if (!elemSizeOr.ok()) {
        return absl::InternalError(elemSizeOr.status().message());
      }
      std::size_t bindingSize = size * elemSizeOr.value();  // Assuming float data type
      logger_->log(nvinfer1::ILogger::Severity::kINFO,
                   absl::StrFormat("Allocating memory for %s tensor: name=%s (%s), size=%d",
                                   isInput ? "input" : "output", tensorName,
                                   engine_->getTensorFormatDesc(tensorName), bindingSize)
                       .c_str());

      dbuffs_.emplace_back(cuda_utils::make_unique<std::uint8_t[]>(bindingSize));
      hbuffs_.emplace_back(
          cuda_utils::make_unique_host<std::uint8_t[]>(bindingSize, cudaHostAllocPortable));
      if (!context_->setTensorAddress(tensorName, dbuffs_.at(i).get())) {
        return absl::InternalError(
            absl::StrFormat("Failed to set tensor address for tensor: %s", tensorName));
      }
    }
    return absl::OkStatus();
  }
};

}  // namespace tensorrt_utils

#endif  // __TENSORRT_UTILS__INFERENCE_ENGINE_HPP__
