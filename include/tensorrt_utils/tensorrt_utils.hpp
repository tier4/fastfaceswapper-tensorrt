#ifndef __TENSORRT_UTILS__TENSORRT_UTILS_HPP__
#define __TENSORRT_UTILS__TENSORRT_UTILS_HPP__

#include <NvInfer.h>
#include <absl/log/log.h>
#include <absl/status/statusor.h>

#include <iostream>

namespace tensorrt_utils {

// Custom Logger class
class Logger : public nvinfer1::ILogger {
 public:
  // Constructor to specify the log level
  explicit Logger(
      const nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
      : mSeverity(severity) {}

  // The log function is overridden to handle logging messages based on severity
  // Note: Higher severity levels are assigned lower integer values
  void log(const Severity severity, const char* msg) noexcept override {
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

}  // namespace tensorrt_utils

#endif
