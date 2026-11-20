#include "sgbm.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>

namespace einstar::detail {

cv::Mat sgbm_cpu(const cv::Mat& left, const cv::Mat& right, const SgbmConfig& c) {
  auto sgbm = cv::StereoSGBM::create(0, c.num_disparities, c.block_size, c.p1, c.p2, 1, 0, c.uniqueness_ratio,
                                     c.speckle_window, c.speckle_range, cv::StereoSGBM::MODE_SGBM_3WAY);
  cv::Mat d16;
  sgbm->compute(left, right, d16);
  return d16;
}

void sgbm_postprocess(cv::Mat& disp16, const SgbmConfig& c) {
  cv::medianBlur(disp16, disp16, 3);
  if (c.speckle_window > 0) cv::filterSpeckles(disp16, -16, c.speckle_window, 16 * c.speckle_range);
}

bool sgbm_metal_supports(const SgbmConfig& c) {
  if (c.num_disparities <= 0 || c.num_disparities % 16 != 0 || c.num_disparities > 1024) return false;
  if (c.block_size < 1 || c.block_size % 2 == 0 || c.block_size > 15) return false;
  // The GPU does not saturate costs as OpenCV does: the sum of three path
  // costs, each at most 93 * block^2 + P2, must fit in int16.
  const int p1 = c.p1 > 0 ? c.p1 : 2, p2 = std::max(c.p2 > 0 ? c.p2 : 5, p1 + 1);
  return 3 * (93 * c.block_size * c.block_size + p2) <= 32767;
}

#ifndef EINSTAR_WITH_METAL
bool sgbm_metal_available() { return false; }
cv::Mat sgbm_metal_raw(const cv::Mat&, const cv::Mat&, const SgbmConfig&) {
  throw std::runtime_error("libeinstar was built without Metal");
}
cv::Mat sgbm_metal(const cv::Mat&, const cv::Mat&, const SgbmConfig&) {
  throw std::runtime_error("libeinstar was built without Metal");
}
#endif

}  // namespace einstar::detail
