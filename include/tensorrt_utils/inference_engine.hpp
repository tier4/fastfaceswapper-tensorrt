#ifndef __TENSORRT_UTILS__INFERENCE_ENGINE_HPP__
#define __TENSORRT_UTILS__INFERENCE_ENGINE_HPP__

#include <NvInfer.h>

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <filesystem>
#include <iostream>
#include <tensorrt_utils/tensorrt_utils.hpp>

namespace tensorrt_utils {

class InferenceEngine {
 public:
  // Constructor for InferenceEngine
  // Initializes the engine by deserializing the engine file and setting up the execution context
  InferenceEngine(const std::filesystem::path& enginePath, nvinfer1::ILogger& logger,
                  const std::int32_t maxBs) {
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
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(engine);
    auto context = engine_->createExecutionContext();
    if (!context) {
      throw std::runtime_error("Failed to create execution context");
    }
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(context);

    if (maxBs < 1) {
      throw std::runtime_error("Maximum batch size must be greater than 0");
    }
    maxBs_ = maxBs;

    // Allocate memory for binding buffers
    auto status = allocateMemory();
    if (!status.ok()) {
      throw std::runtime_error(std::string(status.message()));
    }
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
  // Maximum batch size
  std::int32_t maxBs_;
  // CUDA stream for asynchronous operations
  cuda_utils::StreamUniquePtr stream_{cuda_utils::makeCudaStream()};

  // Sets the batch size for the engine
  // Returns an error if the batch size is invalid
  absl::Status setBs(const std::int32_t bs) {
    if (bs < 1 || bs > maxBs_) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Batch size must be between 1 and %d", maxBs_));
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
                   absl::StrFormat("Allocating memory for tensor: %s (%s), size: %d", tensorName,
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
