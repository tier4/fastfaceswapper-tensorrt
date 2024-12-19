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

#ifndef _CV_UTILS__CV_UTILS_HPP_
#define _CV_UTILS__CV_UTILS_HPP_

#include <absl/status/statusor.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <absl/strings/strip.h>

#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

namespace cv_utils {

// Ensure the cv::Mat is continuously mapped on memory
inline cv::Mat makeContinuous(const cv::Mat& img) {
  if (img.isContinuous()) {
    return img;
  } else {
    return img.clone();
  }
}

// Flatten the image to a 1D array
inline cv::Mat flatten(const cv::Mat& img) { return img.reshape(1, img.total()); }

// Check if the file specified by the given path is a valid image
inline bool isValidImg(const std::filesystem::path& filePath) {
  return (std::filesystem::exists(filePath) && std::filesystem::is_regular_file(filePath) &&
          !cv::imread(filePath, cv::IMREAD_UNCHANGED).empty());
}

// Get the total bytes of an image
inline std::size_t bytes(const cv::Mat& img) { return img.total() * img.elemSize(); }

// Crop image at the given ROI
absl::StatusOr<cv::Mat> crop(const cv::Mat& src, const cv::Rect& roi, const bool deepCopy = true,
                             double borderValue = 0) {
  const auto andRoi = roi & cv::Rect({}, src.size());
  if (andRoi.empty()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("The specified ROI does not intersect with the source image."));
  }
  if (!deepCopy && andRoi == roi) {
    return src(andRoi);
  }
  cv::Mat out(roi.size(), src.type(), cv::Scalar::all(borderValue));
  src(andRoi).copyTo(out(andRoi - roi.tl()));
  return out;
}

// Embed the image patch `patch` at the rectangular area (`roi`) of the destination image `dst`.
absl::StatusOr<cv::Mat> embed(
    cv::Mat& dst, const cv::Rect& roi, const cv::Mat& patch, bool inplace = false,
    cv::InterpolationFlags interpolation = cv::InterpolationFlags::INTER_LINEAR) {
  cv::Mat patchResized;
  cv::resize(patch, patchResized, roi.size(), 0, 0, interpolation);
  const auto andRoi = cv::Rect({}, dst.size()) & roi;
  if (andRoi.empty()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("The specified ROI does not intersect with the destination image."));
  }
  if (inplace) {
    patchResized(andRoi - roi.tl()).copyTo(dst(andRoi));
    return dst;
  } else {
    cv::Mat copied = dst.clone();
    patchResized(andRoi - roi.tl()).copyTo(copied(andRoi));
    return copied;
  }
}

// Clip the values of the image to the range [min, max]
inline cv::Mat clip(cv::Mat& img, double minVal, double maxVal, bool inplace = false) {
  if (inplace) {
    cv::max(img, minVal, img);
    cv::min(img, maxVal, img);
    return img;
  } else {
    return cv::min(cv::max(img, minVal), maxVal);
  }
}

// Calculate the center point of rect
template <typename T>
inline cv::Point_<T> calcCenter(const cv::Rect_<T>& rect) {
  return cv::Point_<T>(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
}

// Template function to scale rect
template <typename T>
cv::Rect_<T> scaleRect(const cv::Rect_<T>& rect, double scaleX, double scaleY,
                       bool centerPreserved = false) {
  if (centerPreserved) {
    cv::Point_<T> center = calcCenter(rect);
    cv::Size_<T> newSize(rect.width * scaleX, rect.height * scaleY);
    return cv::Rect_<T>(center - cv::Point_<T>(newSize.width / 2, newSize.height / 2), newSize);
  } else {
    return cv::Rect_<T>(rect.tl().x * scaleX, rect.tl().y * scaleY, rect.width * scaleX,
                        rect.height * scaleY);
  }
}

// Calculate the smallest square that encloses the given rect and shares the same center point
template <typename T>
cv::Rect_<T> calcEnclosingSquare(const cv::Rect_<T>& rect) {
  T side = std::max(rect.width, rect.height);
  cv::Point_<T> center = calcCenter(rect);
  return cv::Rect_<T>(center.x - side / 2, center.y - side / 2, side, side);
}

// Load rectangle ROIs from a text file
absl::StatusOr<std::vector<cv::Rect>> loadROIsFromFile(const std::filesystem::path& filePath) {
  if (!std::filesystem::exists(filePath)) {
    return absl::Status(absl::StatusCode::kNotFound,
                        absl::StrFormat("%s does not exist", filePath));
  } else if (!std::filesystem::is_regular_file(filePath)) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrFormat("%s is not a regular file", filePath));
  }

  // Open file
  std::ifstream file(filePath);
  if (!file.is_open()) {
    return absl::Status(absl::StatusCode::kUnavailable,
                        absl::StrFormat("Could not open %s", filePath));
  }
  std::string line;
  std::size_t lno = 0;
  std::vector<cv::Rect> ret;
  while (std::getline(file, line)) {
    line = absl::StripAsciiWhitespace(line);
    const std::vector<std::string> split = absl::StrSplit(line, ' ');
    if (split.size() != 4) {
      return absl::Status(absl::StatusCode::kInternal,
                          absl::StrFormat("Error at line %d in %s: Expected 4 floating-point "
                                          "values per line, but found %d values",
                                          lno, filePath, split.size()));
    }
    std::vector<double> vals;
    for (const auto& s : split) {
      double v;
      if (!absl::SimpleAtod(s, &v)) {
        return absl::Status(
            absl::StatusCode::kInternal,
            absl::StrFormat(
                "Error at line %d in %s: Could not convert \"%s\" to a floating-point value", lno,
                filePath, s));
      }
      vals.push_back(v);
    }
    ret.emplace_back(
        cv::Rect(cv::Point(vals.at(0), vals.at(1)), cv::Point(vals.at(2), vals.at(3))));
    lno++;
  }
  return ret;
}

}  // namespace cv_utils
#endif
