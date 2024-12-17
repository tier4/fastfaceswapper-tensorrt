#ifndef __INCLUDE_TRT_UTILS_HPP__
#define __INCLUDE_TRT_UTILS_HPP__

#include <NvInfer.h>
#include <absl/log/log.h>
#include <absl/status/statusor.h>

#include <iostream>
#include <unordered_map>

namespace trt_utils {

// Custom Logger class
class Logger : public nvinfer1::ILogger {
 public:
  // Constructor to specify the log level
  explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
      : mSeverity(severity) {}

  // The log function is overridden to handle logging messages based on severity
  // Note: Higher severity levels are assigned lower integer values
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= mSeverity) {
      switch (severity) {
        case Severity::kINTERNAL_ERROR:
          LOG(ERROR) << msg;
          break;
        case Severity::kERROR:
          LOG(ERROR) << msg;
          break;
        case Severity::kWARNING:
          LOG(WARNING) << msg;
          break;
        case Severity::kINFO:
          LOG(INFO) << msg;
          break;
        case Severity::kVERBOSE:
          LOG(INFO) << msg;
          break;
        default:
          LOG(INFO) << msg;
          break;
      }
    }
  }

 private:
  Severity mSeverity;  // Configured minimum severity level
};

absl::StatusOr<std::vector<std::uint8_t>> readEngineFile(const std::string& enginePath) {
  std::ifstream file(enginePath, std::ios::binary);
  if (!file) {
    return absl::NotFoundError("Failed to open engine file: " + enginePath);
  }

  file.seekg(0, file.end);
  size_t fileSize = file.tellg();
  file.seekg(0, file.beg);

  std::vector<std::uint8_t> engineData(fileSize);
  file.read(reinterpret_cast<char*>(engineData.data()), fileSize);
  file.close();

  return engineData;
}

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
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(engine);
    auto context = engine_->createExecutionContext();
    if (!context) {
      throw std::runtime_error("Failed to create execution context");
    }
    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(context);

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

  ~InferenceEngine() {
    for (auto& pair : buffs_) {
      pair.second.reset();
    }
    context_.reset();
    engine_.reset();
  }

 private:
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::shared_ptr<trt_utils::Logger> logger_;
  std::unordered_map<std::string, std::shared_ptr<void>> buffs_;
  std::int32_t maxBs_;

  absl::Status allocateMemory() {
    // Allocate memory for input and output tensors
    std::int32_t nbBindings = engine_->getNbIOTensors();
    for (int i = 0; i < nbBindings; ++i) {
      auto tensorName = engine_->getIOTensorName(i);
      nvinfer1::Dims dims = engine_->getTensorShape(tensorName);
      size_t size = maxBs_;
      for (std::int32_t j = 1; j < dims.nbDims; ++j) {
        size *= dims.d[j];
      }
      std::size_t bindingSize = size * sizeof(float);  // Assuming float data type

      void* buffer;
      if (cudaMalloc(&buffer, bindingSize) != cudaSuccess) {
        return absl::InternalError(
            absl::StrFormat("Failed to allocate CUDA memory for tensor: %s", tensorName));
      }
      if (!context_->setTensorAddress(tensorName, buffer)) {
        return absl::InternalError(
            absl::StrFormat("Failed to set tensor address for tensor: %s", tensorName));
      }
      buffs_[std::string(tensorName)] = std::shared_ptr<void>(buffer, cudaFree);
    }
    return absl::OkStatus();
  }
};

// Function to convert nvinfer1::Dims to a string
std::string dimsToString(const nvinfer1::Dims& dims) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < dims.nbDims; ++i) {
    oss << dims.d[i];
    if (i < dims.nbDims - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

}  // namespace trt_utils

#endif
