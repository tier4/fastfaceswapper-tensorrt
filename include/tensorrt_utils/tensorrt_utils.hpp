// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _TENSORRT_UTILS__TENSORRT_UTILS_HPP_
#define _TENSORRT_UTILS__TENSORRT_UTILS_HPP_

#include <NvInfer.h>
#include <absl/log/log.h>
#include <absl/status/statusor.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

namespace tensorrt_utils {

// Function to check if nvinfer1::Dims matches the expected shape
bool expectDims(const nvinfer1::Dims& dims, const std::vector<std::int64_t>& expected) {
  if (dims.nbDims != static_cast<std::int32_t>(expected.size())) {
    return false;
  }
  for (int i = 0; i < dims.nbDims; ++i) {
    if (expected[i] != -1 && dims.d[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

// Custom Logger class
class Logger : public nvinfer1::ILogger {
 public:
  // Constructor to specify the log level
  explicit Logger(
      const nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
      : severity_(severity) {}

  // The log function is overridden to handle logging messages based on severity
  // Note: Higher severity levels are assigned lower integer values
  void log(const Severity severity, const char* msg) noexcept override {
    if (severity <= severity_) {
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
  Severity severity_;  // Configured minimum severity level
};

// Function to get the size in bytes of nvinfer1::DataType
// NOTE: This function returns a float value to support kINT4 and kBOOL data types (0.5, 0.125 bytes
// respectively)
absl::StatusOr<double> dataTypeToBytes(const nvinfer1::DataType dataType) {
  switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kUINT8:
      return 1;
    case nvinfer1::DataType::kBOOL:
      return 0.125;
#if NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >= 6
    case nvinfer1::DataType::kFP8:
      return 1;
#endif
#if NV_TENSORRT_MAJOR >= 10
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kINT4:
      return 0.5;
#endif
  }
  return absl::InvalidArgumentError("Unknown data type");
}

// Function to read engine file and return its content as a vector of bytes
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

// Function to get the profile batch sizes (min, opt, max) for a given profile index
absl::StatusOr<std::tuple<std::size_t, std::size_t, std::size_t>> getProfileBatchSizes(
    nvinfer1::ICudaEngine* engine, std::size_t profileIndex) {
  std::int64_t minBs = -1, optBs = -1, maxBs = -1;
  for (std::int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
    const auto tensorName = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
      const auto minDims =
          engine->getProfileShape(tensorName, profileIndex, nvinfer1::OptProfileSelector::kMIN);
      const auto optDims =
          engine->getProfileShape(tensorName, profileIndex, nvinfer1::OptProfileSelector::kOPT);
      const auto maxDims =
          engine->getProfileShape(tensorName, profileIndex, nvinfer1::OptProfileSelector::kMAX);
      if (minBs == -1) {
        minBs = minDims.d[0];
      }
      if (optBs == -1) {
        optBs = optDims.d[0];
      }
      if (maxBs == -1) {
        maxBs = maxDims.d[0];
      }
      if (minBs != minDims.d[0] || optBs != optDims.d[0] || maxBs != maxDims.d[0]) {
        return absl::InternalError("Profile batch sizes are not consistent");
      }
    }
  }
  if (minBs == -1 || optBs == -1 || maxBs == -1) {
    return absl::InternalError("Failed to get profile batch sizes");
  }
  return std::make_tuple(static_cast<std::size_t>(minBs), static_cast<std::size_t>(optBs),
                         static_cast<std::size_t>(maxBs));
}

// Function to get the size in bytes of a tensor frame for a given tensor index
absl::StatusOr<std::size_t> getIOTensorFrameBytes(nvinfer1::ICudaEngine* engine,
                                                  std::size_t index) {
  if (index >= static_cast<std::size_t>(engine->getNbIOTensors())) {
    return absl::InvalidArgumentError("Invalid tensor index");
  }
  const char* tensorName = engine->getIOTensorName(index);
  auto dims = engine->getTensorShape(tensorName);
  auto dtype = engine->getTensorDataType(tensorName);
  const auto elemSizeOr = dataTypeToBytes(dtype);
  if (!elemSizeOr.ok()) {
    return absl::InternalError(elemSizeOr.status().message());
  }
  std::size_t size = 1;
  for (std::int32_t i = 1; i < dims.nbDims; ++i) {
    size *= dims.d[i];
  }
  size = static_cast<std::size_t>(size * elemSizeOr.value());
  return size;
}

// Function to convert IO tensor name to IO tensor index
absl::StatusOr<std::size_t> getIOTensorIndex(nvinfer1::ICudaEngine* engine,
                                             const char* tensorName) {
  for (std::int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
    if (std::strcmp(engine->getIOTensorName(i), tensorName) == 0) {
      return i;
    }
  }
  return absl::NotFoundError("Tensor " + std::string(tensorName) + " seems to be not an io tensor");
}

// Function to get the names of all input tensors
std::vector<const char*> getInputTensorNames(nvinfer1::ICudaEngine* engine) {
  std::vector<const char*> inputTensorNames;
  for (std::int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
    const auto tensorName = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
      inputTensorNames.push_back(tensorName);
    }
  }
  return inputTensorNames;
}

// Function to get the names of all output tensors
std::vector<const char*> getOutputTensorNames(nvinfer1::ICudaEngine* engine) {
  std::vector<const char*> outputTensorNames;
  for (std::int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
    const auto tensorName = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT) {
      outputTensorNames.push_back(tensorName);
    }
  }
  return outputTensorNames;
}

// Function to convert nvinfer1::Dims to a string
std::string dimsToStr(const nvinfer1::Dims& dims) {
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
