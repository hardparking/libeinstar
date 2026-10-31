// IR-stereo depth registered to the colour camera.
//
// The Einstar projects a speckle pattern that moves with the scanner, so the
// IR images are good for stereo matching but not for visual odometry. This
// turns an IR pair into a depth map in the colour camera's (undistorted)
// image plane, giving an RGB-D frame like an active-stereo depth camera.
#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <vector>

#include "einstar/calibration.hpp"

namespace einstar {

// OpenCV forms of a camera's intrinsics (CV_64F, 3x3 and 1x5).
cv::Mat camera_matrix(const CameraIntrinsics& in);
cv::Mat distortion_coeffs(const CameraIntrinsics& in);

// Raw stream image -> calibration orientation (see stream_orientation()).
// For the colour stream pass the debayered image: rotating raw Bayer data
// would change its pattern.
cv::Mat to_calibration_orientation(const cv::Mat& image, unsigned stream_camera);
// Raw colour stream (Bayer, as EXStar converts it with COLOR_BayerRG2BGR)
// -> BGR in calibration orientation.
cv::Mat color_in_calibration_orientation(const cv::Mat& raw_bayer);

struct DepthParams {
  // Matching runs on rectified images scaled by this factor, relative to
  // the native (~1161 px) rectified focal length; matching dominates the
  // cost. Measured on Apple M-series (Release), real calibration, synthetic
  // plane: 0.75 -> 248 ms / 0.08 mm median error, 0.6 -> 125 ms / 0.11 mm,
  // 0.5 -> 69 ms / 0.13 mm. On real handheld frames 0.5 matched the coverage
  // and repeatability of larger scales. EXStar also matches at half
  // resolution.
  double scale = 0.5;
  float min_depth_mm = 130.f;  // handheld scans on hardware were often 170-230 mm away
  float max_depth_mm = 800.f;
  // Matching. Tuned on mounted scans of a box (lit box faces, which must be
  // planar): block 5 with contrast equalisation (CLAHE) and a stronger
  // smoothness penalty doubled depth on the faces (34% -> 67% of the lit
  // box on held-out frames) while raising the share of depth within 3 mm of
  // a face plane (91% -> 96%). Larger blocks lost both on the fine speckle;
  // looser uniqueness or speckle filtering gained depth only by adding
  // wrong patches. Wrong depth breaks tracking; missing depth only costs
  // features.
  int block_size = 5;
  int p1_factor = 8;   // SGBM P1 = p1_factor * block^2
  int p2_factor = 96;  // SGBM P2 = p2_factor * block^2
  bool equalize = true;  // CLAHE (clip 3, 16x16 tiles) on the rectified pair before matching
  int uniqueness_ratio = 20;
  int speckle_window = 200;
  int speckle_range = 1;
  // Horizontal stretches of the rectified right image to match at, in
  // order; later ones only fill pixels the earlier ones left without depth.
  // At ~20 cm the verged pair sees a slanted surface at very different
  // widths (one box face spanned 15-31 cm and was ~2.5x wider in the right
  // view), which block matching cannot follow. On recorded frames of two
  // mounted box scans, {1, 0.6} raised depth coverage from 26% to 45% and
  // from 20% to 32%; where plain and stretched matching both found depth
  // they agreed to 0.39 mm (median). {1, 0.75, 0.55} added only 2-6 more
  // percentage points; stretches above 1 found almost nothing on those frames. Each
  // stretch costs one more matching pass (65 -> 131 ms per frame at scale
  // 0.5 on Apple M-series).
  std::vector<double> stretches = {1.0, 0.6};
};

class StereoDepth {
 public:
  StereoDepth(const CameraModel& model, cv::Size size = {1280, 1024}, DepthParams params = {});

  // Undistorted colour intrinsics used for the output depth/colour images.
  const cv::Mat& color_K() const { return color_K_; }
  cv::Size size() const { return size_; }
  double baseline_mm() const { return baseline_mm_; }
  // Disparity search window [min, min + num) in scaled pixels. Rectified
  // disparity is f*B/Z + disparity_offset() because each IR camera has its
  // own principal point.
  int min_disparity() const { return min_disp_; }
  int num_disparities() const { return num_disp_; }
  double disparity_offset() const { return disp_offset_; }

  // Calibration indices (kIrLeftCamera or kIrRightCamera) of the geometric
  // left and right IR cameras.
  unsigned left_index() const { return left_; }
  unsigned right_index() const { return right_; }

  // Rectified, scaled IR pair; inputs must already be ordered left, right.
  void rectify(const cv::Mat& left, const cv::Mat& right, cv::Mat& left_rect, cv::Mat& right_rect) const;
  // CV_32F, scaled pixels; invalid pixels are -infinity.
  cv::Mat disparity(const cv::Mat& left_rect, const cv::Mat& right_rect) const;
  // Depth in millimetres (CV_32F, 0 = unknown) in the undistorted colour frame,
  // from the images of calibration cameras kIrLeftCamera and kIrRightCamera.
  cv::Mat depth_in_color(const cv::Mat& ir_left, const cv::Mat& ir_right) const;
  cv::Mat undistort_color(const cv::Mat& bgr) const;

 private:
  cv::Size size_, scaled_;
  unsigned left_ = kIrLeftCamera, right_ = kIrRightCamera;
  DepthParams params_;
  double f_ = 0, cx_ = 0, cy_ = 0, baseline_mm_ = 0, disp_offset_ = 0;  // scaled rectified intrinsics
  int min_disp_ = 0, num_disp_ = 16;
  cv::Mat map_l1_, map_l2_, map_r1_, map_r2_, map_c1_, map_c2_;
  cv::Mat mask_l_, mask_r_;
  cv::Mat color_K_;
  cv::Matx33d R_rect_to_color_;
  cv::Vec3d T_rect_to_color_;
};

}  // namespace einstar
