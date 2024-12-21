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

#define FFSWP_IMG_C 3
#define FFSWP_MASK_C 1
#define FFSWP_DEFAULT_MASK_ROI_VAL (cv::Scalar::all(0.0))
#define FFSWP_DEFAULT_MASK_BG_VAL (cv::Scalar::all(1.0))
#define FFSWP_DEFAULT_CONDITION_MINVAL -1.0
#define FFSWP_DEFAULT_CONDITION_MAXVAL 1.0
#define FFSWP_DEFAULT_CONDITION_ROI_VAL (cv::Scalar::all(0.0))
#define FFSWP_DEFAULT_CROP_SCALE 1.7

namespace ffswp {

// Function to create a mask with the given image size and ROI
absl::StatusOr<cv::Mat> createMask(const cv::Size& imgSize, const cv::Rect& roi,
                                   const cv::Scalar& roiValue = FFSWP_DEFAULT_MASK_ROI_VAL,
                                   const cv::Scalar& bgValue = FFSWP_DEFAULT_MASK_BG_VAL) {
  // Create a mask with the given image size and number of channels, initialized to the background
  // value
  cv::Mat mask(imgSize, CV_MAKE_TYPE(CV_32F, FFSWP_MASK_C), bgValue);

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
absl::StatusOr<std::tuple<cv::Mat, cv::Mat, cv::Rect>> createInput(
    const cv::Mat& srcImg, const cv::Rect& roi, const cv::Size& inputSize,
    const double cropScale = FFSWP_DEFAULT_CROP_SCALE,
    const std::tuple<float, float>& conditionValueRange = {FFSWP_DEFAULT_CONDITION_MINVAL,
                                                           FFSWP_DEFAULT_CONDITION_MAXVAL},
    const cv::Scalar& conditionROIValue = FFSWP_DEFAULT_CONDITION_ROI_VAL,
    const cv::Scalar& maskROIValue = FFSWP_DEFAULT_MASK_ROI_VAL,
    const cv::Scalar& maskBgValue = FFSWP_DEFAULT_MASK_BG_VAL) {
  // Check if the source image is valid
  if (srcImg.type() != CV_MAKE_TYPE(CV_8U, FFSWP_IMG_C)) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Invalid source image type. Expected CV_8UC%d.", FFSWP_IMG_C));
  }

  // Calculate the ROI to crop from the source image
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

  // Convert color order from BGR to RGB
  cv::cvtColor(cropped, cropped, cv::COLOR_BGR2RGB);

  // Resize the cropped image to the input size
  const auto scaleX = static_cast<double>(inputSize.width) / cropped.cols;
  const auto scaleY = static_cast<double>(inputSize.height) / cropped.rows;
  cv::resize(cropped, cropped, inputSize, 0, 0, cv::InterpolationFlags::INTER_LINEAR);

  // Normalize values of the image to the range conditionValueRange[0]~conditionValueRange[1]
  const auto [conditionMinVal, conditionMaxVal] = conditionValueRange;
  cropped.convertTo(cropped, CV_MAKE_TYPE(CV_32F, cropped.channels()), 1.0 / 255.0);
  cropped = cropped * (conditionMaxVal - conditionMinVal) + conditionMinVal;

  // Fill the ROI area inside the cropped image with the conditionROIValue
  const auto roiInCroppedScaled = cv_utils::scaleRect(roi - cropROI.tl(), scaleX, scaleY);
  auto filledOr = cv_utils::fill(cropped, roiInCroppedScaled, conditionROIValue, false);
  if (!filledOr.ok()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrFormat("Failed to create input: %s", filledOr.status().message()));
  }

  // Create the mask image
  auto maskOr = createMask(inputSize, roiInCroppedScaled, maskROIValue, maskBgValue);
  if (!maskOr.ok()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrFormat("Failed to create input: %s", maskOr.status().message()));
  }

  // Return the resized image and the mask
  return std::make_tuple(filledOr.value(), maskOr.value(), cropROI);
}

class FastFaceSwapper {
 public:
  // Get 2D image size
  inline cv::Size getImgSize() const { return cv::Size(imgW_, imgH_); }

  // Get image height
  inline std::size_t getImgH() const { return imgH_; }

  // Get image width
  inline std::size_t getImgW() const { return imgW_; }

  // Get number of image channels
  inline std::size_t getImgC() const { return imgC_; }

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
    if (!tensorrt_utils::expectDims(dimsCondition, {-1, FFSWP_IMG_C, imgH, imgW})) {
      throw std::runtime_error(absl::StrFormat(
          "Invalid shape for %s tensor of the tensorRT engine: %s (expected {-1, %d, %d, %d}).",
          ioNameCondition_, tensorrt_utils::dimsToStr(dimsCondition), FFSWP_IMG_C, imgH, imgW));
    }
    if (!tensorrt_utils::expectDims(dimsMask, {-1, FFSWP_MASK_C, imgH, imgW})) {
      throw std::runtime_error(absl::StrFormat(
          "Invalid shape for %s tensor of the tensorRT engine: %s (expected {-1, %d, %d, %d}).",
          ioNameMask_, tensorrt_utils::dimsToStr(dimsMask), FFSWP_MASK_C, imgH, imgW));
    }
    if (!tensorrt_utils::expectDims(dimsOutput, {-1, FFSWP_IMG_C, imgH, imgW})) {
      throw std::runtime_error(absl::StrFormat(
          "Invalid shape for %s tensor of the tensorRT engine: %s (expected {-1, %d, %d, %d}).",
          ioNameOutput_, tensorrt_utils::dimsToStr(dimsOutput), FFSWP_IMG_C, imgH, imgW));
    }
    // Set image dimensions
    imgH_ = imgH;
    imgW_ = imgW;
    imgC_ = FFSWP_IMG_C;
  }

  // Swap faces in the source image
  absl::StatusOr<cv::Mat> swap(
      cv::Mat& srcImg, const std::vector<cv::Rect>& rois, const bool inplace = false,
      const double cropScale = FFSWP_DEFAULT_CROP_SCALE,
      std::tuple<float, float> conditionValueRange = {FFSWP_DEFAULT_CONDITION_MINVAL,
                                                      FFSWP_DEFAULT_CONDITION_MAXVAL},
      const cv::Scalar& conditionROIValue = FFSWP_DEFAULT_CONDITION_ROI_VAL) {
    auto srcImg_ = inplace ? srcImg : srcImg.clone();
    const auto maxBs = inferEngine_->getMaxBatchSize();
    const auto [conditionMinVal, conditionMaxVal] = conditionValueRange;
    for (std::size_t offset = 0; offset < rois.size(); offset += maxBs) {
      const auto bs = std::min(maxBs, rois.size() - offset);
      std::vector<cv::Mat> conditions;
      std::vector<cv::Mat> masks;
      std::vector<cv::Rect> cropROIs;
      for (std::size_t i = 0; i < bs; ++i) {
        const auto roi = rois[offset + i];
        // Create input condition and mask
        // NOTE: Resize and normalize are done inside createInput
        auto inputOr = createInput(srcImg_, roi, getImgSize(), cropScale, conditionValueRange,
                                   conditionROIValue);
        if (!inputOr.ok()) {  // NOTE: Skip invalid inputs
          return absl::Status(
              absl::StatusCode::kInternal,
              absl::StrFormat("Failed to create input: %s", inputOr.status().message()));
        }
        const auto [condition, mask, cropROI] = inputOr.value();
        conditions.push_back(condition);
        masks.push_back(mask);
        cropROIs.push_back(cropROI);
      }

      // Create flatten blob
      // NOTE: Convert memory format from NHWC to NCHW
      // Assumes conditions & masks have been already resized and normalized
      auto batchConditions = cv_utils::makeContinuous(
          cv_utils::flatten(cv::dnn::blobFromImages(conditions, 1.0, cv::Size(), cv::Scalar())));
      auto batchMasks = cv_utils::makeContinuous(
          cv_utils::flatten(cv::dnn::blobFromImages(masks, 1.0, cv::Size(), cv::Scalar())));

      // Convert to byte data
      std::vector<std::uint8_t> conditionData(
          batchConditions.ptr(), batchConditions.ptr() + cv_utils::bytes(batchConditions));
      std::vector<std::uint8_t> maskData(batchMasks.ptr(),
                                         batchMasks.ptr() + cv_utils::bytes(batchMasks));

      // Perform inference
      const auto outputOr =
          inferEngine_->infer({{ioNameCondition_, conditionData}, {ioNameMask_, maskData}}, bs);
      if (!outputOr.ok()) {
        return absl::Status(
            absl::StatusCode::kInternal,
            absl::StrFormat("Failed to perform inference: %s", outputOr.status().message()));
      }

      // NCHW to NHWC
      const auto& outputNCHW = outputOr.value().at(ioNameOutput_);
      auto outputNHWCOr = tensorrt_utils::toChannelLast(outputNCHW, bs, imgC_, imgH_, imgW_, 4);
      if (!outputNHWCOr.ok()) {
        return absl::Status(absl::StatusCode::kInternal,
                            absl::StrFormat("Failed to convert output tensor to NHWC: %s",
                                            outputNHWCOr.status().message()));
      }
      auto& outputNHWC = outputNHWCOr.value();
      assert(outputNHWC.size() == outputNCHW.size());

      // Decode output to cv::Mat
      for (std::size_t i = 0; i < bs; ++i) {
        // Get the current ROI
        const auto roi = rois[offset + i];

        // Decode the output tensor to a cv::Mat
        auto decoded =
            cv::Mat(getImgSize(), CV_MAKE_TYPE(CV_32F, getImgC()),
                    outputNHWC.data() + i * inferEngine_->getIOFrameSizes().at(ioNameOutput_));

        // Clip the output values to the condition value range
        cv_utils::clip(decoded, conditionMinVal, conditionMaxVal, true);

        // Convert the image to 8-bit
        decoded.convertTo(decoded, CV_MAKE_TYPE(CV_32F, decoded.channels()),
                          255.0 / (conditionMaxVal - conditionMinVal),
                          -conditionMinVal * 255.0 / (conditionMaxVal - conditionMinVal));
        decoded.convertTo(decoded, CV_MAKE_TYPE(CV_8U, decoded.channels()));

        // Convert color order from RGB to BGR
        cv::cvtColor(decoded, decoded, cv::COLOR_RGB2BGR);

        // Embed the decoded image back into the source image at the ROI
        const auto cropROI = cropROIs[i];
        const auto scaleX = static_cast<double>(imgW_) / cropROI.width;
        const auto scaleY = static_cast<double>(imgH_) / cropROI.height;
        const auto roiScaled = cv_utils::scaleRect(roi - cropROI.tl(), scaleX, scaleY);
        auto embeddedOr = cv_utils::embed(srcImg_, roi, decoded(roiScaled), true);
        if (!embeddedOr.ok()) {
          return absl::Status(
              absl::StatusCode::kInternal,
              absl::StrFormat("Failed to embed output: %s", embeddedOr.status().message()));
        }
      }
    }
    return srcImg_;
  }

 private:
  std::unique_ptr<tensorrt_utils::InferenceEngine> inferEngine_;  // Inference engine instance
  std::size_t imgH_;                                              // Image height
  std::size_t imgW_;                                              // Image width
  std::size_t imgC_;                                              // Number of image channels
  std::string ioNameCondition_;  // Name of the condition input tensor
  std::string ioNameMask_;       // Name of the mask input tensor
  std::string ioNameOutput_;     // Name of the output tensor
};



}  // namespace ffswp

#endif
