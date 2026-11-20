// Stereo matching backends behind StereoDepth (internal).
#pragma once

#include <opencv2/core.hpp>

namespace einstar::detail {

// OpenCV StereoSGBM in MODE_SGBM_3WAY with minDisparity 0, disp12MaxDiff 1
// and preFilterCap 0, searching [0, num_disparities).
struct SgbmConfig {
  int num_disparities = 16;  // multiple of 16
  int block_size = 5;
  int p1 = 200, p2 = 2400;
  int uniqueness_ratio = 20;
  int speckle_window = 200;
  int speckle_range = 1;
};

// CV_16S disparity in 1/16 pixel (cv::StereoSGBM::compute); invalid
// pixels are -16. Inputs are CV_8UC1 of equal size.
cv::Mat sgbm_cpu(const cv::Mat& left, const cv::Mat& right, const SgbmConfig& c);

// True when a Metal GPU is present and libeinstar was built with Metal.
bool sgbm_metal_available();
// Whether sgbm_metal can match with this configuration: at most 1024
// disparities, an odd block of at most 15, and costs that fit in 16 bits.
bool sgbm_metal_supports(const SgbmConfig& c);
// Bit-identical to sgbm_cpu, matched on the GPU. Throws std::runtime_error
// if Metal is unavailable or the shaders fail to build.
cv::Mat sgbm_metal(const cv::Mat& left, const cv::Mat& right, const SgbmConfig& c);
// The GPU part alone: OpenCV's 3-way result before its 3x3 median and
// speckle filters (for tests).
cv::Mat sgbm_metal_raw(const cv::Mat& left, const cv::Mat& right, const SgbmConfig& c);

// OpenCV's post-processing in StereoSGBM::compute: 3x3 median, then
// speckle filtering.
void sgbm_postprocess(cv::Mat& disp16, const SgbmConfig& c);

}  // namespace einstar::detail
