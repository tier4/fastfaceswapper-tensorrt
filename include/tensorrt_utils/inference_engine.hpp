#ifndef __TENSORRT_UTILS__INFERENCE_ENGINE_HPP__
#define __TENSORRT_UTILS__INFERENCE_ENGINE_HPP__

#include <NvInfer.h>

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <filesystem>
#include <iostream>
#include <tensorrt_utils/tensorrt_utils.hpp>

namespace tensorrt_utils {

class InferenceEngine {
 public:
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
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  nvinfer1::ILogger* logger_;
  std::vector<cuda_utils::CudaUniquePtr<std::uint8_t[]>> dbuffs_;
  std::vector<cuda_utils::CudaUniquePtrHost<std::uint8_t[]>> hbuffs_;
  std::int32_t maxBs_;

  absl::Status allocateMemory() {
    // Allocate memory for input and output tensors
    std::int32_t nbBindings = engine_->getNbIOTensors();
    for (int i = 0; i < nbBindings; ++i) {
      auto tensorName = engine_->getIOTensorName(i);
      nvinfer1::Dims dims = engine_->getTensorShape(tensorName);
      auto dtype = engine_->getTensorDataType(tensorName);
      size_t size = maxBs_;
      for (std::int32_t j = 1; j < dims.nbDims; ++j) {
        size *= dims.d[j];
      }
      auto elemSizeOr = dataTypeToSize(dtype);
      if (!elemSizeOr.ok()) {
        return absl::InternalError(elemSizeOr.status().message());
      }
      std::size_t bindingSize = size * elemSizeOr.value();  // Assuming float data type
      auto strOr = dataTypeToString(dtype);
      if (!strOr.ok()) {
        return absl::InternalError(strOr.status().message());
      }
      logger_->log(nvinfer1::ILogger::Severity::kINFO,
                   absl::StrFormat("Allocating memory for tensor: %s (dtype: %s), size: %d",
                                   tensorName, strOr.value(), bindingSize)
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

#endif
