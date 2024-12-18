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

#ifndef _FFSWP_FFSWP_HPP_
#define _FFSWP_FFSWP_HPP_

#include <NvInfer.h>

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorrt_utils/inference_engine.hpp>

// Define tensor names for the FastFaceSwapper
#define FFSWP_TENSORNAME_CONDITION "condition"
#define FFSWP_TENSORNAME_MASK "mask"
#define FFSWP_TENSORNAME_INPAINTED "inpainted"

namespace ffswp {

class FastFaceSwapper {
 public:
  // Get 2D image size
  std::tuple<std::int32_t, std::int32_t> getImgSize() const { return {imgH_, imgW_}; }

  // Get number of image channels
  std::int32_t getImgChannels() const { return imgC_; }

  // Constructor for FastFaceSwapper class
  // Initializes the inference engine with the provided engine path and logger
  FastFaceSwapper(const std::filesystem::path& enginePath, nvinfer1::ILogger& logger) {
    // Create inference engine
    inferEngine_ = std::make_unique<tensorrt_utils::InferenceEngine>(enginePath, logger);

    // Validate inference engine input tensor names
    const auto inputNames = inferEngine_->getInputNames();
    if (std::find(inputNames.begin(), inputNames.end(), FFSWP_TENSORNAME_CONDITION) ==
            inputNames.end() ||
        std::find(inputNames.begin(), inputNames.end(), FFSWP_TENSORNAME_MASK) ==
            inputNames.end()) {
      throw std::runtime_error(
          absl::StrFormat("Expected names of input tensors: %s, %s. But missing one or both.",
                          FFSWP_TENSORNAME_CONDITION, FFSWP_TENSORNAME_MASK));
    }

    // Validate inference engine output tensor names
    const auto outputNames = inferEngine_->getOutputNames();
    if (std::find(outputNames.begin(), outputNames.end(), FFSWP_TENSORNAME_INPAINTED) ==
        outputNames.end()) {
      throw std::runtime_error(absl::StrFormat(
          "Expected names of outptu tensors: %s. But got wrong name.", FFSWP_TENSORNAME_INPAINTED));
    }

    // Get tensor shapes from the inference engine
    const auto trtEngine = inferEngine_->getEngine();
    const auto conditionDims = trtEngine->getTensorShape(FFSWP_TENSORNAME_CONDITION);
    const auto maskDims = trtEngine->getTensorShape(FFSWP_TENSORNAME_MASK);
    const auto inpaintedDims = trtEngine->getTensorShape(FFSWP_TENSORNAME_INPAINTED);
    if (conditionDims.nbDims != 4) {
      throw std::runtime_error(absl::StrFormat("Invalid number of dimensions for %s tensor: %d.",
                                               FFSWP_TENSORNAME_CONDITION, conditionDims.nbDims));
    }
    const auto imgH = conditionDims.d[2];
    const auto imgW = conditionDims.d[3];

    // Check channel sizes of io tensors
    if (!tensorrt_utils::expectDims(conditionDims, {-1, 3, imgH, imgW})) {
      throw std::runtime_error(absl::StrFormat("Invalid shape for %s tensor: %s.",
                                               FFSWP_TENSORNAME_CONDITION,
                                               tensorrt_utils::dimsToString(conditionDims)));
    }
    if (!tensorrt_utils::expectDims(maskDims, {-1, 1, imgH, imgW})) {
      throw std::runtime_error(absl::StrFormat("Invalid shape for %s tensor: %s.",
                                               FFSWP_TENSORNAME_MASK,
                                               tensorrt_utils::dimsToString(maskDims)));
    }
    if (!tensorrt_utils::expectDims(inpaintedDims, {-1, 3, imgH, imgW})) {
      throw std::runtime_error(absl::StrFormat("Invalid shape for %s tensor: %s.",
                                               FFSWP_TENSORNAME_INPAINTED,
                                               tensorrt_utils::dimsToString(inpaintedDims)));
    }
    // Set image dimensions
    imgH_ = imgH;
    imgW_ = imgW;
    imgC_ = conditionDims.d[1];  // Assuming channels are at dimension 1
  }

  // Swap faces in the source image
  cv::Mat swap(const cv::Mat& srcImg) { return cv::Mat(); }

  // Destructor for FastFaceSwapper class
  ~FastFaceSwapper() { std::cout << "FastFaceSwapper destructor" << std::endl; }

 private:
  std::unique_ptr<tensorrt_utils::InferenceEngine> inferEngine_;  // Inference engine instance
  std::int64_t imgH_;                                             // Image height
  std::int64_t imgW_;                                             // Image width
  std::int64_t imgC_;                                             // Number of image channels
};

}  // namespace ffswp

#endif
