// Stereo matching on the GPU with libSGM (built with EINSTAR_WITH_CUDA).
#pragma once

#include <opencv2/core.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <tuple>

#include "einstar/depth.hpp"

namespace sgm {
class StereoSGM;
}

namespace einstar::detail {

// Whether a CUDA device is usable (the runtime reports at least one).
bool cuda_device_available();

class GpuMatcher {
 public:
  GpuMatcher();
  ~GpuMatcher();

  // Same contract as the CPU matcher: CV_32F disparity u - ur of `l`
  // against `r` (CV_8U, any widths, same height) in [min_disp, min_disp +
  // num_disp); unmatched pixels are -inf.
  cv::Mat match(const cv::Mat& l, const cv::Mat& r, int min_disp, int num_disp, const DepthParams& p);

 private:
  sgm::StereoSGM& sgm_for(int width, int height, int size, const DepthParams& p);

  std::mutex m_;
  // libSGM allocates its device buffers per image and window size; keep them.
  std::map<std::tuple<int, int, int>, std::unique_ptr<sgm::StereoSGM>> sgm_;
};

}  // namespace einstar::detail
