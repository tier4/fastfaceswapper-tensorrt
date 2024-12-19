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

// Check if the file specified by the given path is a valid image
inline bool isValidImg(const std::filesystem::path& filePath) {
  return (std::filesystem::exists(filePath) && std::filesystem::is_regular_file(filePath) &&
          !cv::imread(filePath, cv::IMREAD_UNCHANGED).empty());
}

// Get the 3D shape (h, w, c) of an image
inline std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> shape(const cv::Mat& img) {
  return {img.rows, img.cols, img.channels()};
}

// Alias of shape
inline std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> shape3d(const cv::Mat& img) {
  return shape(img);
}

// Get the 2D shape (h, w) of an image
inline std::tuple<std::uint32_t, std::uint32_t> shape2d(const cv::Mat& img) {
  return {img.rows, img.cols};
}

// Get the total bytes of an image
inline std::size_t bytes(const cv::Mat& img) { return img.rows * img.cols * img.elemSize(); }

// Get the total elements of an image (elements = pixels * channels)
inline std::size_t elems(const cv::Mat& img) { return img.rows * img.cols * img.channels(); }

// Get the size of an image element
inline std::size_t elemBytes(const cv::Mat& img) { return img.elemSize() / img.channels(); }

// Get the total pixels of an image
inline std::size_t pixels(const cv::Mat& img) { return img.rows * img.cols; }

// Get the size of an image pixel (size of an element * channels)
inline std::size_t pixelBytes(const cv::Mat& img) { return img.elemSize(); }

// Validate the shape (h, w, c) of an image
inline bool validateShape(const cv::Mat& img, const uint32_t height, const uint32_t width,
                          const uint32_t channels) {
  return !((height != 0 && height != static_cast<std::uint32_t>(img.rows)) ||
           (width != 0 && width != static_cast<std::uint32_t>(img.cols)) ||
           (channels != 0 && channels != static_cast<std::uint32_t>(img.channels())));
}

// Check if `img_a` & `img_b` have the same data type
inline bool isSameType(const cv::Mat& imgA, const cv::Mat& imgB) {
  return imgA.depth() == imgB.depth();
}

// Check if `img_a` & `img_b` have the same number of pixels
inline bool isSamePixelCount(const cv::Mat& imgA, const cv::Mat& imgB) {
  return pixels(imgA) == pixels(imgB);
}

// Check if `img_a` & `img_b` have the same number of elements
inline bool isSameElemCount(const cv::Mat& imgA, const cv::Mat& imgB) {
  return elems(imgA) == elems(imgB);
}

// Check if `img_a` & `img_b` have the same 3D (h, w, c) shape
inline bool isSameShape(const cv::Mat& imgA, const cv::Mat& imgB) {
  return shape(imgA) == shape(imgB);
}

// Alias of isSameShape
inline bool isSameShape3d(const cv::Mat& imgA, const cv::Mat& imgB) {
  return shape3d(imgA) == shape3d(imgB);
}

// Check if `img_a` & `img_b` have the same 2D (h, w) shape
inline bool isSameShape2d(const cv::Mat& imgA, const cv::Mat& imgB) {
  return shape2d(imgA) == shape2d(imgB);
}

// Crop image at the given ROI
absl::StatusOr<cv::Mat> crop(const cv::Mat& src, const cv::Rect& roi, const bool deepCopy = true,
                             double borderValue = 0) {
  const auto andRoi = roi & cv::Rect({}, src.size());
  if (andRoi.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrFormat("Zero IOU between src & ROI"));
  }
  if (!deepCopy && andRoi == roi) {
    return src(andRoi);
  }
  cv::Mat out(roi.size(), src.type(), cv::Scalar::all(borderValue));
  src(andRoi).copyTo(out(andRoi - roi.tl()));
  return out;
}

// Embed image patch `patch` at the rectangle area (`roi`) of the destination image `dst`.
absl::StatusOr<cv::Mat> embed(
    cv::Mat& dst, const cv::Rect& roi, const cv::Mat& patch, bool inplace = false,
    cv::InterpolationFlags interpolation = cv::InterpolationFlags::INTER_LINEAR) {
  cv::Mat patchResized;
  cv::resize(patch, patchResized, roi.size(), 0, 0, interpolation);
  const auto andRoi = cv::Rect({}, dst.size()) & roi;
  if (andRoi.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrFormat("Zero IOU between dst & ROI"));
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

// Template function to scale rect
template <typename T>
inline cv::Rect_<T> scaleRect(const cv::Rect_<T>& rect, double scale) {
  return cv::Rect_<T>(rect.tl() * scale, rect.br() * scale);
}

// Calculate the center point of rect
template <typename T>
inline cv::Point_<T> calcCenter(const cv::Rect_<T>& rect) {
  return cv::Point_<T>(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
}

// Load rectangle ROIs from a text file
absl::StatusOr<std::vector<cv::Rect>> loadRoisFromFile(const std::filesystem::path& filePath) {
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
      return absl::Status(
          absl::StatusCode::kInternal,
          absl::StrFormat("Expected 4 floating-point values per line, but L.%d of %s has %d", lno,
                          filePath, split.size()));
    }
    std::vector<double> vals;
    for (const auto& s : split) {
      double v;
      if (!absl::SimpleAtod(s, &v)) {
        return absl::Status(
            absl::StatusCode::kInternal,
            absl::StrFormat(
                "Could not convert from text to floating-point value (L.%d of %s: \"%s\")", lno,
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
