#ifndef __TENSORRT_UTILS__INFERENCE_ENGINE_HPP__
#define __TENSORRT_UTILS__INFERENCE_ENGINE_HPP__

#include <NvInfer.h>

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <iostream>
#include <tensorrt_utils/tensorrt_utils.hpp>
#include <unordered_map>

namespace tensorrt_utils {

class InferenceEngine {
 public:
  InferenceEngine(const std::string& enginePath, Logger& logger, const std::int32_t maxBs) {
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

    logger_ = std::shared_ptr<Logger>(&logger);
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

    // Clean up runtime
    delete runtime;

    // Allocate memory for binding buffers
    auto status = allocateMemory();
    if (!status.ok()) {
      throw std::runtime_error(std::string(status.message()));
    }
  }

  ~InferenceEngine() { engine_.reset(); }

 private:
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::shared_ptr<tensorrt_utils::Logger> logger_;
  std::unordered_map<std::string, cuda_utils::CudaUniquePtr<void>> buffs_;
  std::int32_t maxBs_;

  absl::Status allocateMemory() {
    // Allocate memory for input and output tensors
    std::int32_t nbBindings = engine_->getNbIOTensors();
    for (int i = 0; i < nbBindings; ++i) {
      auto tensorName = std::string(engine_->getIOTensorName(i));
      nvinfer1::Dims dims = engine_->getTensorShape(tensorName.c_str());
      size_t size = maxBs_;
      for (std::int32_t j = 1; j < dims.nbDims; ++j) {
        size *= dims.d[j];
      }
      std::size_t bindingSize = size * sizeof(float);  // Assuming float data type

      buffs_[tensorName] = cuda_utils::make_unique<void>(bindingSize);
      if (!context_->setTensorAddress(tensorName, buffs_[tensorName].get())) {
        return absl::InternalError(
            absl::StrFormat("Failed to set tensor address for tensor: %s", tensorName));
      }
    }
    return absl::OkStatus();
  }
};
}  // namespace tensorrt_utils

#endif
