#include "einstar/depth.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

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

// SGBM disparity u - ur for column u of `l` matching column ur of `r`,
// searched in [min_disp, min_disp + num_disp); unmatched pixels are -inf.
// SGBM searches [minDisparity, minDisparity + numDisparities) and leaves
// that many leftmost columns invalid. The window here starts far from zero
// (possibly below it), so pad the images instead: `l` by num_disp + pad_l
// and `r` by num_disp + pad_r columns on the left, with pad_l - pad_r =
// -min_disp, and search [0, num_disp). The SGBM border then falls in the
// padding.
cv::Mat match(const cv::Mat& l, const cv::Mat& r, int min_disp, int num_disp, const DepthParams& p) {
  const int pad_l = std::max(0, -min_disp), pad_r = std::max(0, min_disp);
  const int width = std::max(l.cols, r.cols) + num_disp + std::max(pad_l, pad_r);
  cv::Mat lpad, rpad;
  cv::copyMakeBorder(l, lpad, 0, 0, num_disp + pad_l, width - l.cols - num_disp - pad_l, cv::BORDER_CONSTANT,
                     cv::Scalar(0));
  cv::copyMakeBorder(r, rpad, 0, 0, num_disp + pad_r, width - r.cols - num_disp - pad_r, cv::BORDER_CONSTANT,
                     cv::Scalar(0));
  const int b = p.block_size;
  auto sgbm = cv::StereoSGBM::create(0, num_disp, b, p.p1_factor * b * b, p.p2_factor * b * b, 1, 0,
                                     p.uniqueness_ratio, p.speckle_window, p.speckle_range,
                                     cv::StereoSGBM::MODE_SGBM_3WAY);
  cv::Mat d16, d;
  sgbm->compute(lpad, rpad, d16);
  d16.colRange(num_disp + pad_l, num_disp + pad_l + l.cols).convertTo(d, CV_32F, 1.0 / 16.0);
  d.setTo(-std::numeric_limits<float>::infinity(), d < 0);
  d += min_disp;
  return d;
}

}  // namespace

StereoDepth::StereoDepth(const CameraModel& model, cv::Size size, DepthParams params)
    : size_(size), params_(params) {
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
    const int min_k = floor16(std::min(0.0, (1 - k) * last_u) + k * min_disp_);
    const int max_k = static_cast<int>(std::ceil(std::max(0.0, (1 - k) * last_u) + k * (min_disp_ + num_disp_)));
    const cv::Mat dk = match(le, rk, min_k, std::max(16, ceil16(max_k - min_k)), params_);
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

}  // namespace einstar
