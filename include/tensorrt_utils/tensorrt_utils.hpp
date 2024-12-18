#ifndef __TENSORRT_UTILS__TENSORRT_UTILS_HPP__
#define __TENSORRT_UTILS__TENSORRT_UTILS_HPP__

#include <NvInfer.h>
#include <absl/log/log.h>
#include <absl/status/statusor.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

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

// Function to convert nvinfer1::DataType to a string
absl::StatusOr<std::string> dataTypeToString(const nvinfer1::DataType dataType) {
  switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
      return "FLOAT";
    case nvinfer1::DataType::kHALF:
      return "HALF";
    case nvinfer1::DataType::kBF16:
      return "BF16";
    case nvinfer1::DataType::kFP8:
      return "FP8";
    case nvinfer1::DataType::kINT64:
      return "INT64";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kUINT8:
      return "UINT8";
    case nvinfer1::DataType::kINT4:
      return "INT4";
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
  }
  return absl::InvalidArgumentError("Unknown data type");
}

// Function to get the size in bytes of nvinfer1::DataType
absl::StatusOr<std::size_t> dataTypeToSize(const nvinfer1::DataType dataType) {
  switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kUINT8:
      return 1;
    case nvinfer1::DataType::kINT4:
      return 1;
    case nvinfer1::DataType::kBOOL:
      return 1;
  }
  return absl::InvalidArgumentError("Unknown data type");
}

absl::StatusOr<std::vector<std::uint8_t>> readEngineFile(const std::filesystem::path& enginePath) {
  std::ifstream file(enginePath, std::ios::binary);
  if (!file) {
    return absl::NotFoundError("Failed to open engine file: " + enginePath.string());
  }

  file.seekg(0, file.end);
  size_t fileSize = file.tellg();
  file.seekg(0, file.beg);

  std::vector<std::uint8_t> engineData(fileSize);
  file.read(reinterpret_cast<char*>(engineData.data()), fileSize);
  file.close();

  return engineData;
}

absl::StatusOr<std::tuple<std::int32_t, std::int32_t, std::int32_t>> getProfileBatchSizes(
    nvinfer1::ICudaEngine* engine, std::int32_t profileIndex) {
  std::int32_t minBs = -1, optBs = -1, maxBs = -1;
  for (std::int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
    const auto tensorName = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
      const auto minDims =
          engine->getProfileShape(tensorName, profileIndex, nvinfer1::OptProfileSelector::kMIN);
      const auto optDims =
          engine->getProfileShape(tensorName, profileIndex, nvinfer1::OptProfileSelector::kOPT);
      const auto maxDims =
          engine->getProfileShape(tensorName, profileIndex, nvinfer1::OptProfileSelector::kMAX);
      minBs = minDims.d[0];
      optBs = optDims.d[0];
      maxBs = maxDims.d[0];
      break;
    }
  }
  if (minBs == -1 || optBs == -1 || maxBs == -1) {
    return absl::InternalError("Failed to get profile batch sizes");
  }
  return std::make_tuple(minBs, optBs, maxBs);
}

absl::StatusOr<std::size_t> getIOTensorFrameSize(nvinfer1::ICudaEngine* engine,
                                                 std::int32_t index) {
  const auto tensorName = engine->getIOTensorName(index);
  auto dims = engine->getTensorShape(tensorName);
  auto dtype = engine->getTensorDataType(tensorName);
  const auto elemSizeOr = dataTypeToSize(dtype);
  if (!elemSizeOr.ok()) {
    return absl::InternalError(elemSizeOr.status().message());
  }
  return std::accumulate(dims.d + 1, dims.d + dims.nbDims, elemSizeOr.value(), std::multiplies<>());
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
