// Synthetic check of the IR-stereo -> colour depth pipeline.
//
// Renders a speckle-textured plane into an ideal (distortion-free) stereo
// pair with a known baseline, runs StereoDepth, and checks the recovered
// depth in the colour camera.
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "einstar/depth.hpp"

using namespace einstar;

namespace {

constexpr int W = 1280, H = 1024;
constexpr double F = 1150, CX = 640, CY = 512;

CameraCalibration ideal_camera(double tx) {
  CameraCalibration c;
  c.intrinsics = {F, F, CX, CY, {}};
  c.world_to_camera.R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  c.world_to_camera.T = {tx, 0, 0};
  return c;
}

// Renders the plane Z = z_mm (world) into a camera at world x = -tx.
cv::Mat render(const cv::Mat& texture, double tx, double z_mm, double mm_per_texel) {
  cv::Mat mapx(H, W, CV_32F), mapy(H, W, CV_32F);
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      const double X = (u - CX) / F * z_mm - tx;
      const double Y = (v - CY) / F * z_mm;
      mapx.at<float>(v, u) = static_cast<float>(X / mm_per_texel + texture.cols / 2.0);
      mapy.at<float>(v, u) = static_cast<float>(Y / mm_per_texel + texture.rows / 2.0);
    }
  cv::Mat img;
  cv::remap(texture, img, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_REFLECT);
  return img;
}

// Renders the slanted plane Z = z_mm + slope * X (world) into a camera at
// world x = -tx, textured by world X, Y.
cv::Mat render_slanted(const cv::Mat& texture, double tx, double z_mm, double slope, double mm_per_texel) {
  cv::Mat mapx(H, W, CV_32F), mapy(H, W, CV_32F);
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      const double a = (u - CX) / F, b = (v - CY) / F;
      const double t = (z_mm - slope * tx) / (1 - slope * a);  // ray (-tx, 0, 0) + t * (a, b, 1)
      mapx.at<float>(v, u) = static_cast<float>((t * a - tx) / mm_per_texel + texture.cols / 2.0);
      mapy.at<float>(v, u) = static_cast<float>(t * b / mm_per_texel + texture.rows / 2.0);
    }
  cv::Mat img;
  cv::remap(texture, img, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_REFLECT);
  return img;
}

// Coverage and median |error| of depth (seen from the camera at the world
// origin) on the slanted plane over the columns [u0, u1).
std::pair<double, double> slanted_stats(const cv::Mat& depth, double z_mm, double slope, int u0, int u1) {
  std::vector<double> err;
  int total = 0;
  for (int v = H / 4; v < 3 * H / 4; v += 2)
    for (int u = u0; u < u1; u += 2) {
      ++total;
      const float d = depth.at<float>(v, u);
      if (d > 0) err.push_back(std::abs(d - z_mm / (1 - slope * (u - CX) / F)));
    }
  std::nth_element(err.begin(), err.begin() + err.size() / 2, err.end());
  return {err.size() / double(total), err.empty() ? 1e9 : err[err.size() / 2]};
}


// Renders the plane Z_left = z_mm (IR-left camera frame) into camera `cam` of a
// real calibration, including lens distortion.
cv::Mat render_real(const cv::Mat& texture, const CameraModel& m, unsigned cam, double z_mm, double texel) {
  const auto& in = m.cameras[cam].intrinsics;
  const cv::Mat D = distortion_coeffs(in);
  std::vector<cv::Point2f> px, norm;
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) px.emplace_back(static_cast<float>(u), static_cast<float>(v));
  cv::undistortPoints(px, norm, camera_matrix(in), D);
  const Pose to_left = m.relative(cam, kIrLeftCamera);
  cv::Mat mapx(H, W, CV_32F), mapy(H, W, CV_32F);
  for (int i = 0; i < W * H; ++i) {
    const Vec3 c = to_left.apply({0, 0, 0});
    const Vec3 tip = to_left.apply({norm[i].x, norm[i].y, 1.0});
    const double dz = tip[2] - c[2];
    const double t = (z_mm - c[2]) / dz;
    const double X = c[0] + t * (tip[0] - c[0]), Y = c[1] + t * (tip[1] - c[1]);
    mapx.at<float>(i / W, i % W) = static_cast<float>(X / texel + texture.cols / 2.0);
    mapy.at<float>(i / W, i % W) = static_cast<float>(Y / texel + texture.rows / 2.0);
  }
  cv::Mat img;
  cv::remap(texture, img, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_REFLECT);
  return img;
}

// Expected depth in the undistorted colour image for the plane Z_left = z_mm
// (Z in the IR-left camera frame).
double expected_color_depth(const CameraModel& m, const cv::Mat& Kc, int u, int v, double z_mm) {
  const double x = (u - Kc.at<double>(0, 2)) / Kc.at<double>(0, 0);
  const double y = (v - Kc.at<double>(1, 2)) / Kc.at<double>(1, 1);
  const Pose to_left = m.relative(kColorCamera, kIrLeftCamera);
  const Vec3 c = to_left.apply({0, 0, 0});
  const Vec3 tip = to_left.apply({x, y, 1.0});
  const double t = (z_mm - c[2]) / (tip[2] - c[2]);
  return t;  // ray parameter equals depth along the colour optical axis (ray z = 1)
}

// EINSTAR_BENCH=1: per-stage timings at several matching scales.
void bench(const CameraModel& m, const cv::Mat& left, const cv::Mat& right) {
  using clock = std::chrono::steady_clock;
  auto ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  for (double scale : {1.0, 0.75, 0.5}) {
    DepthParams p;
    p.scale = scale;
    StereoDepth sd(m, {W, H}, p);
    cv::Mat lr, rr;
    double t_rect = 0, t_disp = 0, t_full = 0;
    const int runs = 5;
    for (int i = 0; i < runs; ++i) {
      auto a = clock::now();
      sd.rectify(left, right, lr, rr);
      auto b = clock::now();
      cv::Mat d = sd.disparity(lr, rr);
      auto c = clock::now();
      cv::Mat depth = sd.depth_in_color(left, right);
      auto e = clock::now();
      t_rect += ms(a, b);
      t_disp += ms(b, c);
      t_full += ms(c, e);
    }
    std::printf("bench scale %.2f: rectify %.1f ms, disparity %.1f ms (num %d), full depth %.1f ms\n", scale,
                t_rect / runs, t_disp / runs, sd.num_disparities(), t_full / runs);
  }
}

int real_calibration_case(const char* path, const cv::Mat& texture) {
  auto r = load_calibration(path);
  if (!r.model) {
    std::printf("FAIL cannot load %s\n", path);
    return 1;
  }
  const CameraModel& m = *r.model;
  const double z = 350.0;
  const cv::Mat left = render_real(texture, m, kIrLeftCamera, z, 0.25);
  const cv::Mat right = render_real(texture, m, kIrRightCamera, z, 0.25);
  if (std::getenv("EINSTAR_BENCH")) bench(m, left, right);
  DepthParams params;
  if (const char* scale = std::getenv("EINSTAR_DEPTH_SCALE")) params.scale = std::atof(scale);
  StereoDepth sd(m, {W, H}, params);
  const auto t0 = std::chrono::steady_clock::now();
  const cv::Mat depth = sd.depth_in_color(left, right);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (const char* dir = std::getenv("EINSTAR_DEBUG_DIR")) {
    cv::Mat lr, rr;
    sd.rectify(sd.left_index() == kIrLeftCamera ? left : right, sd.left_index() == kIrLeftCamera ? right : left, lr,
               rr);
    cv::Mat d = sd.disparity(lr, rr), d8;
    d.convertTo(d8, CV_8U, 255.0 / (sd.min_disparity() + sd.num_disparities()));
    cv::imwrite(std::string(dir) + "/rect_left.png", lr);
    cv::imwrite(std::string(dir) + "/rect_right.png", rr);
    cv::imwrite(std::string(dir) + "/disp.png", d8);
    cv::imwrite(std::string(dir) + "/ir0.png", left);
    cv::imwrite(std::string(dir) + "/ir1.png", right);
    double mn, mx;
    cv::minMaxLoc(d, &mn, &mx);
    std::printf("debug: min_disp %d num %d disparity %.1f..%.1f\n", sd.min_disparity(), sd.num_disparities(), mn, mx);
  }
  // A colour pixel is coverable when its plane point projects inside both
  // rectified IR images with the SGBM margin; measure coverage over those.
  const auto& lin = m.cameras[sd.left_index()].intrinsics;
  const auto& rin = m.cameras[sd.right_index()].intrinsics;
  const Pose c2l = m.relative(kColorCamera, sd.left_index()), c2r = m.relative(kColorCamera, sd.right_index());
  const double margin = 32;
  std::vector<double> err;
  int valid = 0, coverable = 0;
  for (int v = H / 8; v < 7 * H / 8; v += 2)
    for (int u = 16; u < W - 16; u += 2) {
      const double zc = expected_color_depth(m, sd.color_K(), u, v, z);
      const Vec3 pc{(u - sd.color_K().at<double>(0, 2)) / sd.color_K().at<double>(0, 0) * zc,
                    (v - sd.color_K().at<double>(1, 2)) / sd.color_K().at<double>(1, 1) * zc, zc};
      const Vec3 pl = c2l.apply(pc), pr = c2r.apply(pc);
      const double ul = lin.fx * pl[0] / pl[2] + lin.cx, vl = lin.fy * pl[1] / pl[2] + lin.cy;
      const double ur = rin.fx * pr[0] / pr[2] + rin.cx, vr = rin.fy * pr[1] / pr[2] + rin.cy;
      if (ul < margin || ul > W - 32 || ur < 32 || ur > W - 32 || vl < 32 || vl > H - 32 || vr < 32 ||
          vr > H - 32)
        continue;
      ++coverable;
      const float d = depth.at<float>(v, u);
      if (d <= 0) continue;
      ++valid;
      err.push_back(std::abs(d - zc));
    }
  const double cover = coverable ? valid / double(coverable) : 0;
  std::nth_element(err.begin(), err.begin() + err.size() / 2, err.end());
  const double med = err.empty() ? 1e9 : err[err.size() / 2];
  std::printf("real calibration: baseline %.1f mm, coverage %.1f%% of %d overlap samples, median |error| %.3f mm, "
              "%.0f ms/frame\n",
              sd.baseline_mm(), cover * 100, coverable, med, ms);
  return (cover > 0.8 && med < 1.0) ? 0 : 1;
}

}  // namespace

int main() {
  cv::Mat texture(1600, 1600, CV_8U);
  cv::randu(texture, 0, 255);
  cv::GaussianBlur(texture, texture, cv::Size(5, 5), 1.2);
  cv::normalize(texture, texture, 0, 255, cv::NORM_MINMAX);

  const double baseline = 120.0, z = 300.0, texel = 0.25;
  CameraModel m;
  m.cameras[kColorCamera] = ideal_camera(0);  // colour coincident with IR left
  m.cameras[kIrLeftCamera] = ideal_camera(0);
  m.cameras[kIrRightCamera] = ideal_camera(-baseline);  // sits at world x = +baseline

  StereoDepth sd(m, {W, H});
  int failures = 0;
  if (std::abs(sd.baseline_mm() - baseline) > 0.5) {
    std::printf("FAIL baseline %.3f\n", sd.baseline_mm());
    ++failures;
  }

  const cv::Mat left = render(texture, 0, z, texel);
  const cv::Mat right = render(texture, -baseline, z, texel);
  const cv::Mat depth = sd.depth_in_color(left, right);
  if (std::getenv("EINSTAR_DEBUG")) {
    cv::Mat lr, rr;
    sd.rectify(left, right, lr, rr);
    cv::Mat d = sd.disparity(lr, rr);
    double mn, mx;
    cv::minMaxLoc(d, &mn, &mx);
    std::printf("min_disp %d num %d disp range %.1f..%.1f at centre %.1f, rect size %dx%d mean %.1f\n",
                sd.min_disparity(), sd.num_disparities(), mn, mx, d.at<float>(d.rows / 2, d.cols / 2), lr.cols,
                lr.rows, cv::mean(lr)[0]);
  }

  // Only the right part of the left/colour image is seen by both IR cameras:
  // at this depth the right camera misses the leftmost f*B/Z pixels.
  const int u0 = static_cast<int>(F * baseline / z) + 32, u1 = W - 16;
  std::vector<float> vals;
  for (int v = H / 4; v < 3 * H / 4; ++v)
    for (int u = u0; u < u1; ++u)
      if (float d = depth.at<float>(v, u); d > 0) vals.push_back(d);
  const double coverage = vals.size() / double((H / 2) * (u1 - u0));
  std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
  const double median = vals.empty() ? 0 : vals[vals.size() / 2];
  std::printf("coverage %.1f%%, median depth %.2f mm (expected %.1f)\n", coverage * 100, median, z);
  if (coverage < 0.8) { std::printf("FAIL coverage\n"); ++failures; }
  if (std::abs(median - z) > 1.0) { std::printf("FAIL depth\n"); ++failures; }

  // A plane slanted ~50 deg looks much wider in the right view than in the
  // left; the stretched matching pass recovers most of what plain matching
  // misses.
  {
    const double slope = 1.2;
    const cv::Mat sl = render_slanted(texture, 0, z, slope, texel);
    const cv::Mat sr = render_slanted(texture, -baseline, z, slope, texel);
    DepthParams plain;
    plain.stretches = {1.0};
    const auto [cov_plain, err_plain] =
        slanted_stats(StereoDepth(m, {W, H}, plain).depth_in_color(sl, sr), z, slope, W / 2, W - 16);
    const auto [cov, err] = slanted_stats(sd.depth_in_color(sl, sr), z, slope, W / 2, W - 16);
    std::printf("slanted plane: coverage %.1f%% (plain matching %.1f%%), median |error| %.2f mm\n", cov * 100,
                cov_plain * 100, err);
    if (cov < 0.85 || cov < cov_plain + 0.15) { std::printf("FAIL slanted coverage\n"); ++failures; }
    if (err > 1.0) { std::printf("FAIL slanted depth\n"); ++failures; }
  }
  if (const char* path = std::getenv("EINSTAR_CAMERA_MODEL")) failures += real_calibration_case(path, texture);
  std::printf(failures ? "FAIL\n" : "PASS\n");
  return failures ? 1 : 0;
}
