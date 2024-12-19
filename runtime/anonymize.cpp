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

#include <gflags/gflags.h>

#include <cv_utils/cv_utils.hpp>
#include <ffswp/ffswp.hpp>
#include <filesystem>
#include <iostream>
#include <tensorrt_utils/inference_engine.hpp>

#define IMG_DIR_NAME "images"
#define ANNOT_DIR_NAME "annotations"

DEFINE_string(engine_path, "", "Path to the input tensorRT engine (required).");
DEFINE_string(data_dir, "",
              "Path to the root directory containing images and annotations (required).");
DEFINE_string(out_dir, "",
              "Path to the output directory (required). If not exists, it will be created.");

// Structure to hold parsed arguments
typedef struct ParsedArgs {
  std::filesystem::path engine_path;
  std::filesystem::path data_dir;
  std::filesystem::path out_dir;
} ParsedArgs;

// Function to validate command line flags
absl::StatusOr<ParsedArgs> validateFlags() {
  ParsedArgs args;
  if (FLAGS_engine_path.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "Missing required flag --engine_path");
  }
  if (FLAGS_data_dir.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "Missing required flag --data_dir");
  }
  if (FLAGS_out_dir.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "Missing required flag --out_dir");
  }
  for (const auto& path : {FLAGS_engine_path, FLAGS_data_dir}) {
    if (!std::filesystem::exists(path)) {
      return absl::Status(absl::StatusCode::kNotFound,
                          absl::StrFormat("Could not find file/directory at %s", path));
    }
  }
  args.engine_path = FLAGS_engine_path;
  args.data_dir = FLAGS_data_dir;
  args.out_dir = FLAGS_out_dir;
  return args;
}

int main(int argc, char** argv) {
  // Parse command line flags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Validate flags
  const auto args = validateFlags();
  if (!args.ok()) {
    LOG(FATAL) << args.status();
  }
  const auto parsed = args.value();

  // Initialize logger
  tensorrt_utils::Logger logger(nvinfer1::ILogger::Severity::kINFO);

  // Initialize FastFaceSwapper
  ffswp::FastFaceSwapper faceSwapper(parsed.engine_path, logger);

  const auto imgDir = parsed.data_dir / IMG_DIR_NAME;
  const auto annotDir = parsed.data_dir / ANNOT_DIR_NAME;

  // Check if the image and annotation directories exist
  if (!std::filesystem::exists(imgDir)) {
    LOG(FATAL) << "Could not find directory for images at " << imgDir;
  }
  if (!std::filesystem::exists(annotDir)) {
    LOG(FATAL) << "Could not find directory for annotations at " << annotDir;
  }

  // Create output directory if it does not exist
  if (!std::filesystem::exists(parsed.out_dir)) {
    std::filesystem::create_directories(parsed.out_dir);
  }

  // Iterate over all images in the image directory
  for (const auto& entry : std::filesystem::recursive_directory_iterator(parsed.data_dir)) {
    const auto& imgPath = entry.path();
    if (!cv_utils::isValidImg(imgPath)) {  // Skip non-image files or unreadable images
      continue;
    }

    // Find corresponding annotation file
    const auto annotPath = annotDir / (imgPath.stem().string() + ".txt");
    if (!std::filesystem::exists(annotPath)) {
      LOG(ERROR) << "Could not find annotation file for image at " << imgPath;
      continue;
    }

    // Load image
    cv::Mat srcImg = cv::imread(imgPath, cv::IMREAD_COLOR);

    // Load annotations
    auto roisOr = cv_utils::loadROIsFromFile(annotPath);
    if (!roisOr.ok()) {
      LOG(ERROR) << "Failed to load ROIs from " << annotPath << ": " << roisOr.status().message();
      continue;
    }
    const auto& rois = roisOr.value();

    // Swap faces in the source image
    auto swappedOr = faceSwapper.swap(srcImg, rois, true);
    if (!swappedOr.ok()) {
      LOG(ERROR) << "Failed to swap faces in " << imgPath << ": " << swappedOr.status().message();
      continue;
    }
    const auto& swapped = swappedOr.value();

    // Save output image
    const auto outPath = parsed.out_dir / imgPath.filename();
    if (!cv::imwrite(outPath, swapped)) {
      LOG(ERROR) << "Failed to save output image to " << outPath;
    } else {
      LOG(INFO) << "Saved output image to " << outPath;
    }
  }

  return 0;
}
