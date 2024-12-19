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

#include <cv_utils/cv_utils.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorrt_utils/inference_engine.hpp>

namespace ffswp {

class FastFaceSwapper {
 public:
  // Get 2D image size
  inline std::tuple<std::int32_t, std::int32_t> getImgSize() const { return {imgH_, imgW_}; }

  // Get number of image channels
  inline std::int32_t getImgChannels() const { return imgC_; }

  // Constructor for FastFaceSwapper class
  // Initializes the inference engine with the provided engine path and logger
  FastFaceSwapper(const std::filesystem::path& enginePath, nvinfer1::ILogger& logger,
                  const std::vector<std::string> names = {"condition", "mask", "inpainted"}) {
    // Create inference engine
    inferEngine_ = std::make_unique<tensorrt_utils::InferenceEngine>(enginePath, logger);

    // Validate given names
    if (names.size() != 3) {
      throw std::runtime_error(absl::StrFormat(
          "Expected 3 names for input/output tensors. But got wrong number = %d.", names.size()));
    }

    // Set io tensor names
    ioNameCondition_ = names[0];
    ioNameMask_ = names[1];
    ioNameOutput_ = names[2];

    // Validate given io names
    auto inputNames = inferEngine_->getInputNames();
    auto ioNames = inferEngine_->getOutputNames();
    ioNames.insert(ioNames.end(), inputNames.begin(), inputNames.end());
    for (const auto& name : names) {
      if (std::find(ioNames.begin(), ioNames.end(), name) == ioNames.end()) {
        throw std::runtime_error(absl::StrFormat(
            "Could not find any io tensors of name = %s in the tensorRT engine.", name));
      }
    }

    // Get tensor shapes from the inference engine
    const auto& engine = inferEngine_->getEngine();
    const auto dimsCondition = engine.getTensorShape(ioNameCondition_.c_str());
    const auto dimsMask = engine.getTensorShape(ioNameMask_.c_str());
    const auto dimsOutput = engine.getTensorShape(ioNameOutput_.c_str());
    if (dimsCondition.nbDims != 4) {
      throw std::runtime_error(
          absl::StrFormat("Invalid number of dimensions for %s tensor: %d (expected 4).",
                          ioNameCondition_, dimsCondition.nbDims));
    }
    const auto imgH = dimsCondition.d[2];
    const auto imgW = dimsCondition.d[3];

    // Check channel sizes of io tensors
    if (!tensorrt_utils::expectDims(dimsCondition, {-1, 3, imgH, imgW})) {
      throw std::runtime_error(absl::StrFormat(
          "Invalid shape for %s tensor of the tensorRT engine: %s (expected {-1, 3, %d, %d}).",
          ioNameCondition_, tensorrt_utils::dimsToStr(dimsCondition), imgH, imgW));
    }
    if (!tensorrt_utils::expectDims(dimsMask, {-1, 1, imgH, imgW})) {
      throw std::runtime_error(absl::StrFormat(
          "Invalid shape for %s tensor of the tensorRT engine: %s (expected {-1, 1, %d, %d}).",
          ioNameMask_, tensorrt_utils::dimsToStr(dimsMask), imgH, imgW));
    }
    if (!tensorrt_utils::expectDims(dimsOutput, {-1, 3, imgH, imgW})) {
      throw std::runtime_error(absl::StrFormat(
          "Invalid shape for %s tensor of the tensorRT engine: %s (expected {-1, 3, %d, %d}).",
          ioNameOutput_, tensorrt_utils::dimsToStr(dimsOutput), imgH, imgW));
    }
    // Set image dimensions
    imgH_ = imgH;
    imgW_ = imgW;
    imgC_ = dimsCondition.d[1];  // Assuming channels are at dimension 1
  }

  // Swap faces in the source image
  cv::Mat swap(const cv::Mat& srcImg, bool inplace = false) { return cv::Mat(); }

  // Destructor for FastFaceSwapper class
  ~FastFaceSwapper() { std::cout << "FastFaceSwapper destructor" << std::endl; }

 private:
  std::unique_ptr<tensorrt_utils::InferenceEngine> inferEngine_;  // Inference engine instance
  std::int64_t imgH_;                                             // Image height
  std::int64_t imgW_;                                             // Image width
  std::int64_t imgC_;                                             // Number of image channels
  std::string ioNameCondition_;
  std::string ioNameMask_;
  std::string ioNameOutput_;
};

}  // namespace ffswp

#endif
