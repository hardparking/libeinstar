#include "einstar/depth.hpp"

#include "sgbm.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef EINSTAR_WITH_CUDA
#include "depth_gpu.hpp"
#endif

namespace einstar {

cv::Mat camera_matrix(const CameraIntrinsics& in) {
  return cv::Mat(cv::Matx33d(in.fx, 0, in.cx, 0, in.fy, in.cy, 0, 0, 1));
}

cv::Mat distortion_coeffs(const CameraIntrinsics& in) {
  return cv::Mat(cv::Matx<double, 1, 5>(in.distortion[0], in.distortion[1], in.distortion[2],
                                        in.distortion[3], in.distortion[4]));
}

cv::Mat to_calibration_orientation(const cv::Mat& image, unsigned stream_camera) {
  if (stream_orientation(stream_camera) == StreamOrientation::AsCalibrated) return image;
  cv::Mat out;
  cv::rotate(image, out, cv::ROTATE_180);
  return out;
}

bool backend_available(MatchBackend backend) {
  switch (backend) {
    case MatchBackend::Metal: return detail::sgbm_metal_available();
    case MatchBackend::Cuda:
#ifdef EINSTAR_WITH_CUDA
      return detail::cuda_device_available();
#else
      return false;
#endif
    default: return true;
  }
}

std::vector<MatchBackend> available_backends() {
  std::vector<MatchBackend> out;
  for (MatchBackend b : {MatchBackend::Cpu, MatchBackend::Metal, MatchBackend::Cuda})
    if (backend_available(b)) out.push_back(b);
  return out;
}

const char* to_string(MatchBackend backend) {
  switch (backend) {
    case MatchBackend::Auto: return "auto";
    case MatchBackend::Cpu: return "cpu";
    case MatchBackend::Metal: return "metal";
    case MatchBackend::Cuda: return "cuda";
  }
  return "?";
}

std::optional<MatchBackend> parse_match_backend(std::string_view name) {
  for (MatchBackend b : {MatchBackend::Auto, MatchBackend::Cpu, MatchBackend::Metal, MatchBackend::Cuda})
    if (name == to_string(b)) return b;
  return std::nullopt;
}

cv::Mat color_in_calibration_orientation(const cv::Mat& raw_bayer) {
  cv::Mat bgr;
  cv::cvtColor(raw_bayer, bgr, cv::COLOR_BayerRG2BGR);
  return to_calibration_orientation(bgr, 2);
}

namespace {

cv::Mat R_of(const Pose& p) { return cv::Mat(3, 3, CV_64F, const_cast<double*>(p.R.data())).clone(); }
cv::Mat T_of(const Pose& p) { return cv::Mat(3, 1, CV_64F, const_cast<double*>(p.T.data())).clone(); }

int floor16(double v) { return static_cast<int>(std::floor(v / 16.0)) * 16; }
int ceil16(double v) { return static_cast<int>(std::ceil(v / 16.0)) * 16; }

// Bounds of a camera's image border after rectification by R with focal
// length f and principal point (0, 0).
cv::Rect2d rectified_bounds(const CameraIntrinsics& in, const cv::Mat& R, double f, cv::Size size) {
  std::vector<cv::Point2f> border, rect;
  const int n = 32;
  for (int i = 0; i <= n; ++i) {
    const float x = static_cast<float>(i) * (size.width - 1) / n, y = static_cast<float>(i) * (size.height - 1) / n;
    border.insert(border.end(), {{x, 0.f}, {x, size.height - 1.f}, {0.f, y}, {size.width - 1.f, y}});
  }
  const cv::Mat P = (cv::Mat_<double>(3, 3) << f, 0, 0, 0, f, 0, 0, 0, 1);
  cv::undistortPoints(border, rect, camera_matrix(in), distortion_coeffs(in), R, P);
  return cv::boundingRect(rect);
}

detail::SgbmConfig sgbm_config(const DepthParams& p, int num_disp) {
  const int b = p.block_size;
  detail::SgbmConfig c;
  c.num_disparities = num_disp;
  c.block_size = b;
  c.p1 = p.p1_factor * b * b;
  c.p2 = p.p2_factor * b * b;
  c.uniqueness_ratio = p.uniqueness_ratio;
  c.speckle_window = p.speckle_window;
  c.speckle_range = p.speckle_range;
  return c;
}

// {start, size} of the disparity window searched against the right image
// stretched by k, for left columns up to last_u. Column u matching right
// column ur sits at k*ur in the stretched image, so the search runs over
// u - k*ur = (1 - k)*u + k*d for d in [min_disp, min_disp + num_disp).
std::pair<int, int> stretch_window(double k, double last_u, int min_disp, int num_disp) {
  const int min_k = floor16(std::min(0.0, (1 - k) * last_u) + k * min_disp);
  const int max_k = static_cast<int>(std::ceil(std::max(0.0, (1 - k) * last_u) + k * (min_disp + num_disp)));
  return {min_k, std::max(16, ceil16(max_k - min_k))};
}

// SGBM disparity u - ur for column u of `l` matching column ur of `r`,
// searched in [min_disp, min_disp + num_disp); unmatched pixels are -inf.
// SGBM searches [minDisparity, minDisparity + numDisparities) and leaves
// that many leftmost columns invalid. The window here starts far from zero
// (possibly below it), so pad the images instead: `l` by num_disp + pad_l
// and `r` by num_disp + pad_r columns on the left, with pad_l - pad_r =
// -min_disp, and search [0, num_disp). The SGBM border then falls in the
// padding.
cv::Mat match(const cv::Mat& l, const cv::Mat& r, int min_disp, int num_disp, const DepthParams& p,
              detail::GpuMatcher* cuda) {
#ifdef EINSTAR_WITH_CUDA
  // libSGM (Cuda) takes any window and does its own padding. As with Metal,
  // Auto uses the CPU for good after a failure.
  static std::atomic<bool> cuda_failed{false};
  if (cuda && p.backend == MatchBackend::Cuda) return cuda->match(l, r, min_disp, num_disp, p);
  if (cuda && !cuda_failed) {
    try {
      return cuda->match(l, r, min_disp, num_disp, p);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "einstar: CUDA matching failed, using the CPU: %s\n", e.what());
      cuda_failed = true;
    }
  }
#else
  (void)cuda;
#endif
  const int pad_l = std::max(0, -min_disp), pad_r = std::max(0, min_disp);
  const int width = std::max(l.cols, r.cols) + num_disp + std::max(pad_l, pad_r);
  cv::Mat lpad, rpad;
  cv::copyMakeBorder(l, lpad, 0, 0, num_disp + pad_l, width - l.cols - num_disp - pad_l, cv::BORDER_CONSTANT,
                     cv::Scalar(0));
  cv::copyMakeBorder(r, rpad, 0, 0, num_disp + pad_r, width - r.cols - num_disp - pad_r, cv::BORDER_CONSTANT,
                     cv::Scalar(0));
  const detail::SgbmConfig c = sgbm_config(p, num_disp);
  cv::Mat d16, d;
  // Auto matches on the CPU where Metal does not support the configuration,
  // and for good after a Metal failure (StereoDepth rejects unsupported
  // configurations for an explicit Metal backend).
  static std::atomic<bool> metal_failed{false};
  const bool metal = p.backend == MatchBackend::Metal ||
                     (p.backend == MatchBackend::Auto && !metal_failed && detail::sgbm_metal_available() &&
                      detail::sgbm_metal_supports(c));
  if (metal && p.backend == MatchBackend::Auto) {
    try {
      d16 = detail::sgbm_metal(lpad, rpad, c);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "einstar: Metal matching failed, using the CPU: %s\n", e.what());
      metal_failed = true;
    }
  } else if (metal) {
    d16 = detail::sgbm_metal(lpad, rpad, c);
  }
  if (d16.empty()) d16 = detail::sgbm_cpu(lpad, rpad, c);
  d16.colRange(num_disp + pad_l, num_disp + pad_l + l.cols).convertTo(d, CV_32F, 1.0 / 16.0);
  d.setTo(-std::numeric_limits<float>::infinity(), d < 0);
  d += min_disp;
  return d;
}

}  // namespace

StereoDepth::StereoDepth(const CameraModel& model, cv::Size size, DepthParams params)
    : size_(size), params_(params) {
  if (!backend_available(params.backend))
    throw std::invalid_argument(std::string("stereo matching backend not available: ") + to_string(params.backend));
#ifdef EINSTAR_WITH_CUDA
  // Auto uses Cuda whenever a device is usable.
  if (params.backend == MatchBackend::Cuda || (params.backend == MatchBackend::Auto && backend_available(MatchBackend::Cuda)))
    gpu_ = std::make_shared<detail::GpuMatcher>();
#endif
  // Pick the geometric left IR camera: OpenCV's horizontal rectification
  // expects the second camera at negative x in the first camera's frame.
  const bool swap = model.relative(kIrLeftCamera, kIrRightCamera).T[0] > 0;
  left_ = swap ? kIrRightCamera : kIrLeftCamera;
  right_ = swap ? kIrLeftCamera : kIrRightCamera;
  const auto& l = model.cameras[left_].intrinsics;
  const auto& r = model.cameras[right_].intrinsics;
  const auto& c = model.cameras[kColorCamera].intrinsics;
  const Pose lr = model.relative(left_, right_);
  const Pose lc = model.relative(left_, kColorCamera);

  cv::Mat R1, R2, P1, P2, Q;
  // OpenCV's default scaling keeps roughly the native focal length (~1161 px
  // on hardware, the value EXStar's own rectification uses); alpha=1 would
  // shrink it to ~640 px to fit the verged pair's whole overlap in one shared
  // frame, halving the speckle resolution stereo matching sees.
  cv::stereoRectify(camera_matrix(l), distortion_coeffs(l), camera_matrix(r), distortion_coeffs(r), size_, R_of(lr), T_of(lr), R1, R2, P1, P2, Q,
                    cv::CALIB_ZERO_DISPARITY, -1);
  baseline_mm_ = -P2.at<double>(0, 3) / P2.at<double>(0, 0);
  // The verged views then look in different directions, so give each camera
  // its own principal point (as EXStar does) and size the rectified images to
  // hold each camera's whole view. Disparity then carries the constant
  // offset cx_l - cx_r.
  const double f = P1.at<double>(0, 0);
  const cv::Rect2d view_l = rectified_bounds(l, R1, f, size_), view_r = rectified_bounds(r, R2, f, size_);
  const double top = std::min(view_l.y, view_r.y);
  const double bottom = std::max(view_l.br().y, view_r.br().y);
  P1.at<double>(0, 2) = -view_l.x;
  P2.at<double>(0, 2) = -view_r.x;
  P1.at<double>(1, 2) = P2.at<double>(1, 2) = -top;
  P2.at<double>(0, 3) = -f * baseline_mm_;
  K_l_ = camera_matrix(l), D_l_ = distortion_coeffs(l), R1_ = R1.clone(), P1_ = P1.clone();
  K_r_ = camera_matrix(r), D_r_ = distortion_coeffs(r), R2_ = R2.clone(), P2_ = P2.clone();
  scaled_ = cv::Size(static_cast<int>(std::ceil(std::max(view_l.width, view_r.width) * params.scale)),
                     static_cast<int>(std::ceil((bottom - top) * params.scale)));
  // Rectify straight into the scaled image by scaling the projection rows.
  cv::Mat P1s = P1.clone(), P2s = P2.clone();
  P1s.rowRange(0, 2) *= params.scale;
  P2s.rowRange(0, 2) *= params.scale;
  cv::initUndistortRectifyMap(camera_matrix(l), distortion_coeffs(l), R1, P1s, scaled_, CV_32FC1, map_l1_, map_l2_);
  cv::initUndistortRectifyMap(camera_matrix(r), distortion_coeffs(r), R2, P2s, scaled_, CV_32FC1, map_r1_, map_r2_);
  // Pixels of the rectified views that sample outside the source images.
  const cv::Mat full(size_, CV_8U, cv::Scalar(255));
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {params.block_size, params.block_size});
  cv::remap(full, mask_l_, map_l1_, map_l2_, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
  cv::remap(full, mask_r_, map_r1_, map_r2_, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
  cv::erode(mask_l_, mask_l_, kernel);
  cv::erode(mask_r_, mask_r_, kernel);
  f_ = P1s.at<double>(0, 0);
  cx_ = P1s.at<double>(0, 2);
  cy_ = P1s.at<double>(1, 2);
  disp_offset_ = P1s.at<double>(0, 2) - P2s.at<double>(0, 2);

  // Rectified disparity u_l - u_r = f*B/Z + disp_offset_.
  min_disp_ = floor16(f_ * baseline_mm_ / params.max_depth_mm + disp_offset_);
  num_disp_ = std::max(16, ceil16(f_ * baseline_mm_ / params.min_depth_mm + disp_offset_ - min_disp_));
  if (params.backend == MatchBackend::Metal)
    for (const double k : params.stretches)
      if (k > 0 &&
          !detail::sgbm_metal_supports(sgbm_config(params, stretch_window(k, scaled_.width - 1, min_disp_, num_disp_).second)))
        throw std::invalid_argument(
            "Metal matching does not support these parameters (at most 1024 disparities: raise min_depth_mm or "
            "lower scale; block_size up to 15)");

  color_K_ = camera_matrix(c);
  cv::initUndistortRectifyMap(color_K_, distortion_coeffs(c), cv::Mat(), color_K_, size_, CV_32FC1, map_c1_, map_c2_);

  // X_color = R_lc * X_left + T_lc and X_left = R1^T * X_rect.
  cv::Mat R_rc = R_of(lc) * R1.t();
  R_rect_to_color_ = cv::Matx33d(R_rc.ptr<double>());
  T_rect_to_color_ = cv::Vec3d(lc.T[0], lc.T[1], lc.T[2]);
}

void StereoDepth::rectify(const cv::Mat& left, const cv::Mat& right, cv::Mat& lr, cv::Mat& rr) const {
  cv::remap(left, lr, map_l1_, map_l2_, cv::INTER_LINEAR);
  cv::remap(right, rr, map_r1_, map_r2_, cv::INTER_LINEAR);
}

cv::Mat StereoDepth::disparity(const cv::Mat& lr, const cv::Mat& rr) const {
  // Speckle brightness varies a lot across a frame (bright near faces, dim
  // far ones); equalising local contrast lets one cost scale fit both.
  cv::Mat le = lr, re = rr;
  if (params_.equalize) {
    auto clahe = cv::createCLAHE(3.0, {16, 16});
    clahe->apply(lr, le);
    clahe->apply(rr, re);
  }
  constexpr float invalid = -std::numeric_limits<float>::infinity();
  cv::Mat d(lr.size(), CV_32F, cv::Scalar(invalid));
  const double last_u = lr.cols - 1;
  for (const double k : params_.stretches) {
    if (k <= 0) continue;
    cv::Mat rk = re;
    if (k != 1.0)
      cv::resize(re, rk, {std::max(1, static_cast<int>(std::lround(re.cols * k))), re.rows}, 0, 0,
                 k < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
    // Column u matching right column ur sits at k*ur in the stretched image,
    // so the search runs over u - k*ur = (1 - k)*u + k*d for d in the window.
    const auto [min_k, num_k] = stretch_window(k, last_u, min_disp_, num_disp_);
    const cv::Mat dk = match(le, rk, min_k, num_k, params_, gpu_.get());
    // Keep a match only where earlier stretches found nothing, both pixels
    // sample inside their source images, and the depth is in range.
    for (int v = 0; v < d.rows; ++v) {
      float* row = d.ptr<float>(v);
      const float* rowk = dk.ptr<float>(v);
      const std::uint8_t* ml = mask_l_.ptr<std::uint8_t>(v);
      const std::uint8_t* mr = mask_r_.ptr<std::uint8_t>(v);
      for (int u = 0; u < d.cols; ++u) {
        if (row[u] != invalid || rowk[u] == invalid || !ml[u]) continue;
        const double ur = (u - rowk[u]) / k;
        const int uri = static_cast<int>(std::lround(ur));
        const double du = u - ur;
        if (uri < 0 || uri >= rr.cols || !mr[uri] || du < min_disp_ || du >= min_disp_ + num_disp_) continue;
        row[u] = static_cast<float>(du);
      }
    }
  }
  return d;
}

cv::Mat StereoDepth::depth_in_color(const cv::Mat& ir_left, const cv::Mat& ir_right) const {
  cv::Mat lr, rr;
  rectify(left_ == kIrLeftCamera ? ir_left : ir_right, left_ == kIrLeftCamera ? ir_right : ir_left, lr, rr);
  const cv::Mat disp = disparity(lr, rr);

  cv::Mat depth(size_, CV_32F, cv::Scalar(0));
  const double fx = color_K_.at<double>(0, 0), fy = color_K_.at<double>(1, 1);
  const double ccx = color_K_.at<double>(0, 2), ccy = color_K_.at<double>(1, 2);
  const double fb = f_ * baseline_mm_;
  const auto& R = R_rect_to_color_;
  const auto& T = T_rect_to_color_;
  // Each disparity pixel covers ~fx/f_ colour pixels; splat a small square so
  // the full-resolution depth map has no holes.
  const int footprint = std::max(1, static_cast<int>(std::ceil(fx / f_)));
  const int s0 = -(footprint - 1) / 2, s1 = footprint / 2;
  for (int v = 0; v < disp.rows; ++v) {
    const float* dr = disp.ptr<float>(v);
    for (int u = 0; u < disp.cols; ++u) {
      const float d = dr[u] - static_cast<float>(disp_offset_);
      if (!(d > 0)) continue;
      const double Z = fb / d;
      const cv::Vec3d p((u - cx_) * Z / f_, (v - cy_) * Z / f_, Z);
      const cv::Vec3d q = R * p + T;
      if (q[2] < params_.min_depth_mm || q[2] > params_.max_depth_mm) continue;
      const int cu = static_cast<int>(std::lround(fx * q[0] / q[2] + ccx));
      const int cv_ = static_cast<int>(std::lround(fy * q[1] / q[2] + ccy));
      for (int dv = s0; dv <= s1; ++dv)
        for (int du = s0; du <= s1; ++du) {
          const int x = cu + du, y = cv_ + dv;
          if (x < 0 || y < 0 || x >= size_.width || y >= size_.height) continue;
          float& dst = depth.at<float>(y, x);
          if (dst == 0 || q[2] < dst) dst = static_cast<float>(q[2]);
        }
    }
  }
  return depth;
}

cv::Mat StereoDepth::undistort_color(const cv::Mat& bgr) const {
  cv::Mat out;
  cv::remap(bgr, out, map_c1_, map_c2_, cv::INTER_LINEAR);
  return out;
}

namespace {

// Centres of bright elliptical discs. Retro-reflective markers are bright
// only in the IR camera next to the flood LED, so the test is local contrast
// (white top-hat over a 51 px background) rather than a fixed level. At
// ~250 mm a marker is a ~20-30 px disc, an ellipse when seen at an angle;
// opening with a 7 px disc removes laser speckle dots, and the size and
// ellipse-fill tests drop glints and small print.
std::vector<cv::Point2f> bright_blobs(const cv::Mat& img) {
  cv::Mat top, bin;
  cv::morphologyEx(img, top, cv::MORPH_TOPHAT, cv::getStructuringElement(cv::MORPH_ELLIPSE, {51, 51}));
  cv::threshold(top, bin, 70, 255, cv::THRESH_BINARY);
  cv::morphologyEx(bin, bin, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7}));
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  std::vector<cv::Point2f> out;
  for (const auto& c : contours) {
    const double area = cv::contourArea(c);
    if (area < 80 || area > 2500 || c.size() < 8) continue;
    const cv::RotatedRect e = cv::fitEllipse(c);
    const double lo = std::min(e.size.width, e.size.height), hi = std::max(e.size.width, e.size.height);
    if (lo < 9 || hi > 3 * lo || area < 0.8 * CV_PI * lo * hi / 4) continue;
    out.push_back(e.center);
  }
  return out;
}

}  // namespace

std::vector<cv::Point3f> StereoDepth::markers_in_color(const cv::Mat& ir_left, const cv::Mat& ir_right) const {
  const cv::Mat& l = left_ == kIrLeftCamera ? ir_left : ir_right;
  const cv::Mat& r = left_ == kIrLeftCamera ? ir_right : ir_left;
  std::vector<cv::Point2f> bl = bright_blobs(l), br = bright_blobs(r), rl, rr;
  std::vector<cv::Point3f> out;
  if (bl.empty() || br.empty()) return out;
  cv::undistortPoints(bl, rl, K_l_, D_l_, R1_, P1_);
  cv::undistortPoints(br, rr, K_r_, D_r_, R2_, P2_);
  const double f = P1_.at<double>(0, 0), cx = P1_.at<double>(0, 2), cy = P1_.at<double>(1, 2);
  const double offset = P1_.at<double>(0, 2) - P2_.at<double>(0, 2);
  constexpr double kRowTolerance = 1.5;  // px; markers are matched by rectified row
  for (const auto& p : rl) {
    int match = -1, candidates = 0;
    double best = kRowTolerance;
    for (std::size_t j = 0; j < rr.size(); ++j) {
      const double dy = std::abs(rr[j].y - p.y), d = p.x - rr[j].x - offset;
      if (dy > kRowTolerance || d <= 0) continue;
      const double Z = f * baseline_mm_ / d;
      if (Z < params_.min_depth_mm || Z > params_.max_depth_mm) continue;
      ++candidates;
      if (dy <= best) best = dy, match = static_cast<int>(j);
    }
    if (candidates != 1) continue;  // ambiguous rows are skipped rather than guessed
    const double Z = f * baseline_mm_ / (p.x - rr[match].x - offset);
    const cv::Vec3d q = R_rect_to_color_ * cv::Vec3d((p.x - cx) * Z / f, (p.y - cy) * Z / f, Z) + T_rect_to_color_;
    out.emplace_back(static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]));
  }
  // A marker can split into two blobs (seen ~2.5 mm apart on hardware); real
  // markers are further apart, so points within 4 mm are averaged into one.
  std::vector<cv::Point3f> merged;
  std::vector<bool> used(out.size(), false);
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (used[i]) continue;
    cv::Point3f sum = out[i];
    int n = 1;
    for (std::size_t j = i + 1; j < out.size(); ++j)
      if (!used[j] && cv::norm(out[j] - out[i]) < 4.0f) sum += out[j], ++n, used[j] = true;
    merged.push_back(sum * (1.0f / static_cast<float>(n)));
  }
  return merged;
}

}  // namespace einstar
