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

    // Ensure tensors are 4D
    if (conditionDims.nbDims != 4 || maskDims.nbDims != 4 || inpaintedDims.nbDims != 4) {
      throw std::runtime_error("All io tensors must be 4D, but got wrong dimensions.");
    }

    // Ensure mask channel size is 1
    if (maskDims.d[1] != 1) {
      throw std::runtime_error(
          absl::StrFormat("%s tensor must have only 1 channel.", FFSWP_TENSORNAME_MASK));
    }

    // Ensure condition channel size is 3
    if (conditionDims.d[1] != 3) {
      throw std::runtime_error(
          absl::StrFormat("%s tensor must have 3 channels.", FFSWP_TENSORNAME_CONDITION));
    }

    // Validate tensor dimensions
    const auto imgH = conditionDims.d[2];
    const auto imgW = conditionDims.d[3];
    if (maskDims.d[2] != imgH || maskDims.d[3] != imgW || inpaintedDims.d[2] != imgH ||
        inpaintedDims.d[3] != imgW) {
      throw std::runtime_error(absl::StrFormat(
          "All io tensors must have the same 2d spatial dimensions (%dx%d).", imgH, imgW));
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
  std::int32_t imgH_;                                             // Image height
  std::int32_t imgW_;                                             // Image width
  std::int32_t imgC_;                                             // Number of image channels
};

}  // namespace ffswp

#endif
