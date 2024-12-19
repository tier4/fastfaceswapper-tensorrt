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

// Function to create a mask with the given image size and ROI
absl::StatusOr<cv::Mat> createMask(const cv::Size& imgSize, const cv::Rect& roi,
                                   const std::int32_t channels = 1,
                                   const cv::Scalar& roiValue = cv::Scalar::all(0),
                                   const cv::Scalar& bgValue = cv::Scalar::all(1.0)) {
  // Create a mask with the given image size and number of channels, initialized to the background
  // value
  cv::Mat mask(imgSize, CV_MAKE_TYPE(CV_32F, channels), bgValue);

  // Calculate the intersection of the ROI with the image boundaries
  const auto and_roi = roi & cv::Rect({}, mask.size());

  // Check if the intersection is empty, meaning the ROI is outside the image boundaries
  if (and_roi.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "The provided ROI does not intersect with the image boundaries.");
  }

  // Set the ROI area in the mask to the ROI value
  mask(and_roi).setTo(roiValue);

  return mask;
}

// Function to create input images for the face swapping model
absl::StatusOr<std::tuple<cv::Mat, cv::Mat>> createInput(
    const cv::Mat& srcImg, const cv::Rect& roi, const cv::Size& inputSize,
    const float cropScale = 1.4, const std::int32_t conditionChannels = 3,
    const std::tuple<float, float>& conditionValueRange = {-1.0, 1.0},
    const cv::Scalar& conditionROIValue = cv::Scalar::all(0.0), const std::int32_t maskChannels = 1,
    const cv::Scalar& maskROIValue = cv::Scalar::all(0.0),
    const cv::Scalar& maskBgValue = cv::Scalar::all(1.0)) {
  // Check if the source image is valid
  if (srcImg.type() != CV_MAKE_TYPE(CV_8U, conditionChannels)) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Invalid source image type. Expected CV_8UC%d.", conditionChannels));
  }

  // Calculate ROI to crop from the source image
  const auto cropROI =
      cv_utils::scaleRect(cv_utils::calcEnclosingSquare(roi), cropScale, cropScale, true);

  // Crop the source image at the given ROI
  auto croppedOr = cv_utils::crop(srcImg, cropROI);
  if (!croppedOr.ok()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Failed to create input: %s", croppedOr.status().message()));
  }
  cv::Mat cropped = croppedOr.value();

  // Convert color order from bgr to rgb
  cv::cvtColor(cropped, cropped, cv::COLOR_BGR2RGB);

  // Resize the cropped image to the input size
  cv::resize(cropped, cropped, inputSize, 0, 0, cv::InterpolationFlags::INTER_LINEAR);

  // Normalize values of image to the range conditionValueRange[0]~conditionValueRange[1]
  const auto [conditionMinVal, conditionMaxVal] = conditionValueRange;
  cropped.convertTo(cropped, CV_MAKE_TYPE(CV_32F, cropped.channels()), 1.0 / 255.0);
  cropped = cropped * (conditionMaxVal - conditionMinVal) + conditionMinVal;

  // Fill the ROI area inside the cropped image with the conditionROIValue
  auto scaleX = static_cast<double>(inputSize.width) / cropped.cols;
  auto scaleY = static_cast<double>(inputSize.height) / cropped.rows;
  auto roiInCropped = cv_utils::scaleRect(roi - cropROI.tl(), scaleX, scaleY);
  cropped(roiInCropped).setTo(conditionROIValue);

  // Create the mask image
  auto maskOr = createMask(inputSize, roiInCropped, maskChannels, maskROIValue, maskBgValue);
  if (!maskOr.ok()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrFormat("Failed to create input: %s", maskOr.status().message()));
  }

  // Return the resized image and the mask
  return std::make_tuple(cropped, maskOr.value());
}

class FastFaceSwapper {
 public:
  // Get 2D image size
  inline cv::Size getImgSize() const { return cv::Size(imgW_, imgH_); }

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
  cv::Mat swap(const cv::Mat& srcImg, const std::vector<cv::Rect>& rois, bool inplace = false) {
    const auto maxBs = inferEngine_->getMaxBatchSize();
    for (std::size_t offset = 0; offset < rois.size(); offset += maxBs) {
      const auto bs = std::min(maxBs, static_cast<std::int32_t>(rois.size() - offset));
      std::vector<cv::Mat> conditions;
      std::vector<cv::Mat> masks;
      for (std::size_t i = 0; i < bs; ++i) {
        const auto roi = rois[offset + i];
        // Create input condition and mask
        // NOTE: Resize and normalize are done inside createInput
        auto inputOr = createInput(srcImg, roi, getImgSize());
        if (!inputOr.ok()) {  // NOTE: Skip invalid input
          continue;
        }
        auto [condition, mask] = inputOr.value();
        conditions.push_back(condition);
        masks.push_back(mask);
      }
      if (conditions.size() == 0 || masks.size() == 0) {  // Skip if no valid input
        continue;
      }

      // Create flatten blob
      // NOTE: Convert memory format from NHWC to NCHW
      // Assumes conditions & masks have been already resized and normalized
      auto batchConditions =
          cv::dnn::blobFromImages(conditions, 1.0, cv::Size(), cv::Scalar(), true);
      auto batchMasks = cv::dnn::blobFromImages(masks, 1.0, cv::Size(), cv::Scalar(), true);

      // Set input tensors
      // std::unordered_map<std::string, std::vector<std::uint8_t>> inputs;
      // for (std::size_t i = 0; i < bs; ++i) {
      //   const auto& condition = conditions[i];
      //   const auto& mask = masks[i];
      //   const auto conditionSize = condition.total() * condition.elemSize();
      //   const auto maskSize = mask.total() * mask.elemSize();
      //   inputs[ioNameCondition_] = std::vector<std::uint8_t>(conditionSize);
      //   inputs[ioNameMask_] = std::vector<std::uint8_t>(maskSize);
      //   std::memcpy(inputs[ioNameCondition_].data(), condition.data, conditionSize);
      //   std::memcpy(inputs[ioNameMask_].data(), mask.data, maskSize);
      // }

      // // Perform inference
      // const auto outputsOr = inferEngine_->infer(inputs, bs);
      // if (!outputsOr.ok()) {
      //   throw std::runtime_error(
      //       absl::StrFormat("Failed to perform inference: %s", outputsOr.status().message()));
      // }

      // // Get output tensors
      // const auto& outputs = outputsOr.value();
      // for (std::size_t i = 0; i < bs; ++i) {
      //   const auto& output = outputs.at(ioNameOutput_);
      //   const auto outputSize = output.size();
      //   cv::Mat out(getImgSize(), CV_MAKE_TYPE(CV_32F, getImgChannels()), output.data());
      //   cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
      //   cv::resize(out, out, rois[offset + i].size(), 0, 0,
      //   cv::InterpolationFlags::INTER_LINEAR); out.convertTo(out, CV_MAKE_TYPE(CV_8U,
      //   getImgChannels()), 255.0); if (in
      // }
      return cv::Mat();
    }
  }

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
