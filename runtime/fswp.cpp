#include <gflags/gflags.h>

#include <filesystem>
#include <iostream>
#include <tensorrt_utils/inference_engine.hpp>

DEFINE_string(engine_path, "", "Path to the input tensorRT engine (required).");
DEFINE_int32(maxBs, 32, "Maximum batch size (optional).");

// Structure to hold parsed arguments
typedef struct ParsedArgs {
  std::filesystem::path engine_path;
  std::int32_t maxBs;
} ParsedArgs;

// Function to validate command line flags
absl::StatusOr<ParsedArgs> validateFlags() {
  ParsedArgs args;
  if (FLAGS_engine_path.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "Missing required flag --engine_path");
  }
  if (!std::filesystem::exists(FLAGS_engine_path)) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrFormat("Could not find tensorRT engine at %s", FLAGS_engine_path));
  }
  if (FLAGS_maxBs < 1) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Maximum batch size must be greater than 0");
  }
  args.engine_path = FLAGS_engine_path;
  args.maxBs = FLAGS_maxBs;
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
  tensorrt_utils::Logger logger;

  // Create InferenceEngine object
  tensorrt_utils::InferenceEngine engine(parsed.engine_path, logger, parsed.maxBs);

  return 0;
}
