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

#ifndef _TENSORRT_UTILS__INFERENCE_ENGINE_HPP_
#define _TENSORRT_UTILS__INFERENCE_ENGINE_HPP_

#include <NvInfer.h>

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <tensorrt_utils/tensorrt_utils.hpp>
#include <unordered_map>

namespace tensorrt_utils {

absl::StatusOr<std::vector<std::uint8_t>> toChannelLast(
    const std::vector<std::uint8_t>& channelFirstData, const std::size_t n, const std::size_t c,
    const std::size_t h, const std::size_t w) {
  const auto totalSize = n * c * h * w;
  if (channelFirstData.size() != totalSize) {
    return absl::InvalidArgumentError(absl::StrFormat("Invalid data size: expected %d, but got %d",
                                                      totalSize, channelFirstData.size()));
  }
  std::vector<std::uint8_t> channelLastData(n * h * w * c);
  for (std::size_t ni = 0; ni < n; ++ni) {
    for (std::size_t hi = 0; hi < h; ++hi) {
      for (std::size_t wi = 0; wi < w; ++wi) {
        for (std::size_t ci = 0; ci < c; ++ci) {
          channelLastData[ni * h * w * c + hi * w * c + wi * c + ci] =
              channelFirstData[ni * c * h * w + ci * h * w + hi * w + wi];
        }
      }
    }
  }
  return channelLastData;
}

absl::StatusOr<std::vector<std::uint8_t>> toChannelFirst(
    const std::vector<std::uint8_t>& channelLastData, const std::size_t n, const std::size_t c,
    const std::size_t h, const std::size_t w) {
  const auto totalSize = n * c * h * w;
  if (channelLastData.size() != totalSize) {
    return absl::InvalidArgumentError(absl::StrFormat("Invalid data size: expected %d, but got %d",
                                                      totalSize, channelLastData.size()));
  }
  std::vector<std::uint8_t> channelFirstData(n * c * h * w);
  for (std::size_t ni = 0; ni < n; ++ni) {
    for (std::size_t hi = 0; hi < h; ++hi) {
      for (std::size_t wi = 0; wi < w; ++wi) {
        for (std::size_t ci = 0; ci < c; ++ci) {
          channelFirstData[ni * c * h * w + ci * h * w + hi * w + wi] =
              channelLastData[ni * h * w * c + hi * w * c + wi * c + ci];
        }
      }
    }
  }
  return channelFirstData;
}

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

  // Getter for minimum batch size
  inline std::size_t getMinBatchSize() const { return minBs_; }

  // Getter for maximum batch size
  inline std::size_t getMaxBatchSize() const { return maxBs_; }

  // Getter for optimized batch size
  inline std::size_t getOptBathSize() const { return optBs_; }

  // Getter for input tensor shapes
  std::unordered_map<std::string, nvinfer1::Dims> getInputShapes(void) const {
    std::unordered_map<std::string, nvinfer1::Dims> inputShapes;
    for (const auto& tensorName : getInputTensorNames(engine_.get())) {
      inputShapes[tensorName] = engine_->getTensorShape(tensorName);
    }
    return inputShapes;
  }

  // Getter for output tensor shapes
  std::unordered_map<std::string, nvinfer1::Dims> getOutputShapes(void) const {
    std::unordered_map<std::string, nvinfer1::Dims> outputShapes;
    for (const auto& tensorName : getOutputTensorNames(engine_.get())) {
      outputShapes[tensorName] = engine_->getTensorShape(tensorName);
    }
    return outputShapes;
  }

  // Getter for input tensor names
  std::vector<std::string> getInputNames() const {
    auto names = getInputTensorNames(engine_.get());
    return std::vector<std::string>(names.begin(), names.end());
  }

  // Getter for output tensor names
  std::vector<std::string> getOutputNames() const {
    auto names = getOutputTensorNames(engine_.get());
    return std::vector<std::string>(names.begin(), names.end());
  }

  // Getter for the TensorRT engine
  inline const nvinfer1::ICudaEngine& getEngine() const { return *engine_; }

  // Perform inference on the input data and return the output data
  absl::StatusOr<std::unordered_map<std::string, std::vector<std::uint8_t>>> infer(
      const std::unordered_map<std::string, std::vector<std::uint8_t>>& inputs, std::size_t bs) {
    // Validate batch size
    if (bs < minBs_ || bs > maxBs_) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid batch size: %d, min: %d, max: %d", bs, minBs_, maxBs_));
    }

    // Set input tensors
    for (const auto& tensorName : getInputTensorNames(engine_.get())) {
      const auto tensorNameStr = std::string(tensorName);

      // Check if input data for the tensor is provided
      if (inputs.find(tensorNameStr) == inputs.end()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Missing data for input tensor %s", tensorNameStr));
      }

      auto data = inputs.at(tensorNameStr);

      // Get tensor index and frame size
      const auto indexOr = getIOTensorIndex(engine_.get(), tensorName);
      if (!indexOr.ok()) {
        return absl::InternalError(indexOr.status().message());
      }
      const auto frameSizeOr = getIOTensorFrameBytes(engine_.get(), indexOr.value());
      if (!frameSizeOr.ok()) {
        return absl::InternalError(frameSizeOr.status().message());
      }
      const auto frameSize = frameSizeOr.value();
      const auto totalSize = frameSize * bs;

      // Validate input data size
      if (data.size() < totalSize) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Insufficient input data size for tensor: %s. Expected at least %d "
                            "bytes, but got only %d bytes.",
                            tensorNameStr, totalSize, data.size()));
      }

      // Copy input data from host to device
      std::copy(data.begin(), data.begin() + totalSize, hbuffs_[tensorNameStr].get());
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dbuffs_[tensorNameStr].get(), hbuffs_[tensorNameStr].get(),
                                       totalSize, cudaMemcpyHostToDevice, *stream_));
    }

    // Set batch size
    if (!setBs(bs).ok()) {
      return absl::InternalError(absl::StrFormat("Failed to set batch size to %d", bs));
    }

    // Execute inference
    context_->enqueueV3(*stream_);

    // Device to host copy for output tensors
    std::unordered_map<std::string, std::vector<std::uint8_t>> outputs;
    for (const auto& tensorName : getOutputTensorNames(engine_.get())) {
      const auto tensorNameStr = std::string(tensorName);

      // Get tensor index and frame size
      const auto indexOr = getIOTensorIndex(engine_.get(), tensorName);
      if (!indexOr.ok()) {
        return absl::InternalError(indexOr.status().message());
      }
      const auto frameSizeOr = getIOTensorFrameBytes(engine_.get(), indexOr.value());
      if (!frameSizeOr.ok()) {
        return absl::InternalError(frameSizeOr.status().message());
      }
      const auto frameSize = frameSizeOr.value();
      const auto totalSize = frameSize * bs;

      // Copy output data from device to host
      CHECK_CUDA_ERROR(cudaMemcpyAsync(hbuffs_[tensorNameStr].get(), dbuffs_[tensorNameStr].get(),
                                       totalSize, cudaMemcpyDeviceToHost, *stream_));
      outputs[tensorNameStr].resize(totalSize);
    }

    // Synchronize CUDA stream
    cudaStreamSynchronize(*stream_);

    // Copy output data from host to output map
    for (const auto& tensorName : getOutputTensorNames(engine_.get())) {
      const auto tensorNameStr = std::string(tensorName);
      std::copy(hbuffs_[tensorNameStr].get(),
                hbuffs_[tensorNameStr].get() + outputs[tensorNameStr].size(),
                outputs[tensorNameStr].begin());
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
  std::unordered_map<std::string, cuda_utils::CudaUniquePtr<std::uint8_t[]>> dbuffs_;
  // Host buffers for input/output tensors
  std::unordered_map<std::string, cuda_utils::CudaUniquePtrHost<std::uint8_t[]>> hbuffs_;
  // CUDA stream for asynchronous operations
  cuda_utils::StreamUniquePtr stream_{cuda_utils::makeCudaStream()};
  // Minimum batch size for the engine
  std::size_t minBs_;
  // Maximum batch size for the engine
  std::size_t maxBs_;
  // Optimum batch size for the engine
  std::size_t optBs_;

  // Sets the batch size for the engine
  // Returns an error if the batch size is invalid
  absl::Status setBs(const std::size_t bs) {
    if (bs < minBs_ || bs > maxBs_) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid batch size: %d, min: %d, max: %d", bs, minBs_, maxBs_));
    }
    for (const auto tensorName : getInputTensorNames(engine_.get())) {
      auto dims = engine_->getTensorShape(tensorName);
      dims.d[0] = bs;
      if (!context_->setInputShape(tensorName, dims)) {
        return absl::InternalError(
            absl::StrFormat("Failed to set input shape for tensor: %s", tensorName));
      }
    }
    return absl::OkStatus();
  }

  // Allocates memory for input and output tensors
  // Returns an error if memory allocation fails
  absl::Status allocateMemory() {
    // Allocate memory for input and output tensors
    const auto nbBindings = engine_->getNbIOTensors();
    for (int i = 0; i < nbBindings; ++i) {
      const auto tensorName = engine_->getIOTensorName(i);
      const auto tensorNameStr = std::string(tensorName);
      const auto isInput = engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT;
      const auto frameSizeOr = getIOTensorFrameBytes(engine_.get(), i);
      if (!frameSizeOr.ok()) {
        return absl::InternalError(frameSizeOr.status().message());
      }
      const auto frameSize = frameSizeOr.value();
      std::size_t totalSize = frameSize * maxBs_;
      logger_->log(nvinfer1::ILogger::Severity::kINFO,
                   absl::StrFormat("Allocating memory for %s tensor: name=%s (%s), size=%d",
                                   isInput ? "input" : "output", tensorName,
                                   engine_->getTensorFormatDesc(tensorName), totalSize)
                       .c_str());

      dbuffs_[tensorNameStr] = cuda_utils::make_unique<std::uint8_t[]>(totalSize);
      hbuffs_[tensorNameStr] =
          cuda_utils::make_unique_host<std::uint8_t[]>(totalSize, cudaHostAllocPortable);
      if (!context_->setTensorAddress(tensorName, dbuffs_[tensorNameStr].get())) {
        return absl::InternalError(
            absl::StrFormat("Failed to set tensor address for tensor: %s", tensorName));
      }
    }
    return absl::OkStatus();
  }
};

}  // namespace tensorrt_utils

#endif  // __TENSORRT_UTILS__INFERENCE_ENGINE_HPP__
