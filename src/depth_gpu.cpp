#include "depth_gpu.hpp"

#include <cuda_runtime.h>
#include <libsgm.h>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace einstar::detail {

namespace {

// libSGM searches at most 256 disparities per pass. Wider windows are
// covered by passes of 256 that overlap by a quarter; each pass is trusted
// only away from its edges (an eighth on each inner side), where a match is
// not just the edge of a search that missed the true disparity beyond it.
// The trusted parts tile the window, so each disparity has one trusted
// pass, and an overlapping neighbour must not contradict it. On recorded
// frames a half overlap gave the same depth with ~30% more passes.
constexpr int kPass = 256, kStep = 3 * kPass / 4, kMargin = kPass / 8;

int pass_size(int num_disp) {
  for (int s : {64, 128}) if (num_disp <= s) return s;
  return kPass;
}

}  // namespace

bool cuda_device_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

GpuMatcher::GpuMatcher() = default;
GpuMatcher::~GpuMatcher() = default;

sgm::StereoSGM& GpuMatcher::sgm_for(int width, int height, int size, const DepthParams& p) {
  auto& s = sgm_[{width, height, size}];
  if (!s) {
    // SGBM keeps a match when best * (100 + u) / 100 < second best; libSGM
    // when best <= uniqueness * second best.
    const sgm::StereoSGM::Parameters param(p.cuda_p1, p.cuda_p2, 100.f / (100.f + p.uniqueness_ratio), true,
                                           sgm::PathType::SCAN_8PATH, 0, 1);
    s = std::make_unique<sgm::StereoSGM>(width, height, size, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST, param);
  }
  return *s;
}

cv::Mat GpuMatcher::match(const cv::Mat& l, const cv::Mat& r, int min_disp, int num_disp, const DepthParams& p) {
  std::lock_guard<std::mutex> lock(m_);
  const int size = pass_size(num_disp);
  const int passes = num_disp <= size ? 1 : (num_disp - kPass + kStep - 1) / kStep + 1;
  // A libSGM instance holds a cost volume of width * height * size bytes
  // per path (~1 GB at the defaults) and fixes its window, so one instance
  // searches [0, size) for every pass and the right image is shifted by
  // the pass's start instead. Padding both images by `size` columns on the
  // left keeps every reachable right column inside the image; the width
  // (a multiple of 32 for libSGM's kernels) then depends only on `l`.
  const int pad = size, width = (pad + l.cols + 31) / 32 * 32;
  cv::Mat lpad(l.rows, width, CV_8U, cv::Scalar(0));
  l.copyTo(lpad.colRange(pad, pad + l.cols));
  sgm::StereoSGM& s = sgm_for(width, l.rows, size, p);
  const short invalid = static_cast<short>(s.get_invalid_disparity());
  std::vector<cv::Mat> d(passes);
  for (int i = 0; i < passes; ++i) {
    // The last pass ends at the window's end.
    const int lo = min_disp + std::min(i * kStep, std::max(0, num_disp - size));
    // Right column c goes to pad + lo + c, so a match d' found by libSGM is
    // the disparity d' + lo.
    cv::Mat rpad(r.rows, width, CV_8U, cv::Scalar(0));
    const int c0 = std::max(0, -(pad + lo)), c1 = std::min(r.cols, width - (pad + lo));
    if (c1 > c0) r.colRange(c0, c1).copyTo(rpad.colRange(pad + lo + c0, pad + lo + c1));
    cv::Mat d16(l.rows, width, CV_16S);
    s.execute(lpad.data, rpad.data, d16.data);
    // Report a failed kernel or copy as an error (Auto then falls back to the CPU).
    if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess)
      throw std::runtime_error(std::string("libSGM: ") + cudaGetErrorString(err));
    d16 = d16.colRange(pad, pad + l.cols).clone();
    const cv::Mat bad = d16 == invalid;
    d16 += cv::Scalar(16 * lo);
    d16.setTo(std::numeric_limits<short>::min(), bad);
    d[i] = d16;
  }

  // Merge, in units of 1/16 pixel.
  constexpr short none = std::numeric_limits<short>::min();
  cv::Mat merged(l.rows, l.cols, CV_16S, cv::Scalar(none));
  const int end = min_disp + num_disp;
  for (int v = 0; v < l.rows; ++v) {
    short* out = merged.ptr<short>(v);
    for (int u = 0; u < l.cols; ++u) {
      int best = none;
      bool ok = true;
      for (int i = 0; i < passes && ok; ++i) {
        const short x = d[i].ptr<short>(v)[u];
        if (x == none) continue;
        const int lo = min_disp + std::min(i * kStep, std::max(0, num_disp - size)), hi = lo + size;
        const int core_lo = lo == min_disp ? lo : lo + kMargin, core_hi = hi >= end ? hi : hi - kMargin;
        const double xd = x / 16.0;
        if (xd >= core_lo && xd < core_hi) {
          if (best != none && std::abs(best - x) > 16) ok = false;
          best = x;
        }
      }
      if (!ok || best == none) continue;
      // Every pass whose window holds the match must agree or have none.
      for (int i = 0; i < passes && ok; ++i) {
        const short x = d[i].ptr<short>(v)[u];
        const int lo = min_disp + std::min(i * kStep, std::max(0, num_disp - size));
        if (x != none && best >= lo * 16 && best < (lo + size) * 16 && std::abs(best - x) > 16) ok = false;
      }
      if (ok) out[u] = static_cast<short>(best);
    }
  }
  // SGBM's speckle filter, which libSGM lacks.
  if (p.speckle_window > 0) cv::filterSpeckles(merged, none, p.speckle_window, 16 * p.speckle_range);

  cv::Mat out;
  merged.convertTo(out, CV_32F, 1.0 / 16.0);
  out.setTo(-std::numeric_limits<float>::infinity(), merged == none);
  return out;
}

}  // namespace einstar::detail
