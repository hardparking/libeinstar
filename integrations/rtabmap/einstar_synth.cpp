// Renders a synthetic Einstar scanning sequence in einstar-probe replay format.
//
//   einstar-synth --calib file.data_cm --out DIR [--frames N] [--step MM]
//
// Scene: a textured height field z = 380 + 25 sin(x/45) cos(y/55) mm (world)
// with a poster-like albedo.
// IR frames show a dim scene plus a speckle pattern projected from the
// scanner; colour frames show the scene albedo as a Bayer RG mosaic. The
// ground-truth scanner poses (world <- colour camera) go to DIR/poses.txt as
// "id tx ty tz r00 r01 r02 r10 r11 r12 r20 r21 r22" in mm.
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "einstar/calibration.hpp"
#include "einstar/depth.hpp"

using namespace einstar;

namespace {

constexpr int W = 1280, H = 1024;

double height(double x, double y) { return 380.0 + 25.0 * std::sin(x / 45.0) * std::cos(y / 55.0); }

Pose rot_xyz(double ax, double ay, double az, Vec3 t) {
  const double cx = std::cos(ax), sx = std::sin(ax), cy = std::cos(ay), sy = std::sin(ay);
  const double cz = std::cos(az), sz = std::sin(az);
  // R = Rz * Ry * Rx
  Pose p;
  p.R = {cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
         sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
         -sy,     cy * sx,                cy * cx};
  p.T = t;
  return p;
}

// Intersects a ray (origin o, direction d, world frame) with the height field.
bool intersect(const Vec3& o, const Vec3& d, Vec3& hit) {
  double t = (380.0 - o[2]) / d[2];
  for (int i = 0; i < 30; ++i) {
    const double x = o[0] + t * d[0], y = o[1] + t * d[1];
    const double tn = (height(x, y) - o[2]) / d[2];
    if (std::abs(tn - t) < 1e-4) { t = tn; break; }
    t = tn;
  }
  if (t <= 0) return false;
  hit = {o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
  return true;
}

double sample(const cv::Mat& tex, double u, double v) {
  const int x = static_cast<int>(std::floor(u)), y = static_cast<int>(std::floor(v));
  const double fx = u - x, fy = v - y;
  auto at = [&](int xx, int yy) {
    xx = ((xx % tex.cols) + tex.cols) % tex.cols;
    yy = ((yy % tex.rows) + tex.rows) % tex.rows;
    return static_cast<double>(tex.at<std::uint8_t>(yy, xx));
  };
  return (1 - fy) * ((1 - fx) * at(x, y) + fx * at(x + 1, y)) + fy * ((1 - fx) * at(x, y + 1) + fx * at(x + 1, y + 1));
}

}  // namespace

int main(int argc, char** argv) {
  std::string calib, out;
  int frames = 40;
  double step = 3.0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--calib" && i + 1 < argc) calib = argv[++i];
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
    else if (a == "--step" && i + 1 < argc) step = std::atof(argv[++i]);
    else {
      std::fprintf(stderr, "usage: %s --calib file.data_cm --out DIR [--frames N] [--step MM]\n", argv[0]);
      return 2;
    }
  }
  auto r = load_calibration(calib);
  if (!r.model || out.empty()) {
    std::fprintf(stderr, "need a valid --calib and --out\n");
    return 2;
  }
  const CameraModel& m = *r.model;

  // Scene albedo (colour, ~0.5 mm/texel) and projector speckle (in projector pixels).
  // Poster-like albedo: overlapping shapes give repeatable corners for
  // visual odometry. 2048 texels at 0.25 mm/texel = 512 mm square, tiled.
  cv::Mat albedo(2048, 2048, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::RNG arng(3);
  for (int i = 0; i < 2500; ++i) {
    const cv::Scalar col(arng.uniform(0, 255), arng.uniform(0, 255), arng.uniform(0, 255));
    const cv::Point c(arng.uniform(0, 2048), arng.uniform(0, 2048));
    const int sz = arng.uniform(8, 60);
    if (i % 2) cv::circle(albedo, c, sz, col, -1, cv::LINE_AA);
    else cv::rectangle(albedo, c, c + cv::Point(sz, arng.uniform(8, 60)), col, -1, cv::LINE_AA);
  }
  cv::Mat albedo_gray;
  cv::cvtColor(albedo, albedo_gray, cv::COLOR_BGR2GRAY);
  std::vector<cv::Mat> albedo_ch;
  cv::split(albedo, albedo_ch);
  cv::Mat speckle(1024, 1280, CV_8U, cv::Scalar(0));
  cv::RNG rng(7);
  for (int i = 0; i < 90000; ++i)
    cv::circle(speckle, {rng.uniform(0, 1280), rng.uniform(0, 1024)}, 1, cv::Scalar(255), -1, cv::LINE_AA);
  cv::GaussianBlur(speckle, speckle, cv::Size(0, 0), 0.8);
  const double proj_f = 1100;  // projector intrinsics, at the IR-pair midpoint
  // Scanner body frame = colour camera. Projector between the IR cameras.
  const Vec3 cl = m.relative(kIrLeftCamera, kColorCamera).apply({0, 0, 0});
  const Vec3 cr = m.relative(kIrRightCamera, kColorCamera).apply({0, 0, 0});
  const Vec3 proj_c{(cl[0] + cr[0]) / 2, (cl[1] + cr[1]) / 2, (cl[2] + cr[2]) / 2};

  std::ofstream poses(out + "/poses.txt");
  std::vector<cv::Mat> D(3), K(3);
  for (int c = 0; c < 3; ++c) {
    const auto& in = m.cameras[c].intrinsics;
    K[c] = camera_matrix(in);
    D[c] = distortion_coeffs(in);
  }
  std::vector<cv::Point2f> px;
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) px.emplace_back(static_cast<float>(u), static_cast<float>(v));
  std::vector<std::vector<cv::Point2f>> rays(3);
  for (int c = 0; c < 3; ++c) cv::undistortPoints(px, rays[c], K[c], D[c]);

  for (int f = 0; f < frames; ++f) {
    // Scanner (colour camera) pose in the world: sweep along x with a gentle wobble.
    const double s = f * step;
    const Pose world_from_scanner = rot_xyz(0.03 * std::sin(f * 0.2), 0.02 * std::cos(f * 0.15), 0.01 * f,
                                            {s - 60.0, 5.0 * std::sin(f * 0.1), 0.0});
    poses << f;
    for (double t : world_from_scanner.T) poses << ' ' << t;
    for (double e : world_from_scanner.R) poses << ' ' << e;
    poses << '\n';

    for (unsigned stream = 0; stream < 3; ++stream) {
      const unsigned c = calibration_index_for_stream(stream);
      const Pose world_from_cam = world_from_scanner * m.relative(c, kColorCamera);
      const Pose scanner_from_world = world_from_scanner.inverse();
      // Rendered per calibration pixel, stored at the raw (sensor) position.
      const bool rotated = stream_orientation(stream) == StreamOrientation::Rotated180;
      cv::Mat img(H, W, CV_8U, cv::Scalar(0));
      for (int i = 0; i < W * H; ++i) {
        const int u = i % W, v = i / W;
        const int ru = rotated ? W - 1 - u : u, rv = rotated ? H - 1 - v : v;
        const Vec3 o = world_from_cam.apply({0, 0, 0});
        const Vec3 tip = world_from_cam.apply({rays[c][i].x, rays[c][i].y, 1.0});
        const Vec3 d{tip[0] - o[0], tip[1] - o[1], tip[2] - o[2]};
        Vec3 hit;
        if (!intersect(o, d, hit)) continue;
        const double au = hit[0] / 0.25 + 1024, av = hit[1] / 0.25 + 1024;
        double value;
        if (c == kColorCamera) {
          // Bayer RG on the sensor: R at (even, even), G at (odd, even)/(even,
          // odd), B at (odd, odd) of the raw position.
          const int ch = (rv % 2 == 0) ? (ru % 2 == 0 ? 2 : 1) : (ru % 2 == 0 ? 1 : 0);
          value = sample(albedo_ch[ch], au, av);
        } else {
          const Vec3 ps = scanner_from_world.apply(hit);
          const double pu = proj_f * (ps[0] - proj_c[0]) / (ps[2] - proj_c[2]) + 640;
          const double pv = proj_f * (ps[1] - proj_c[1]) / (ps[2] - proj_c[2]) + 512;
          value = 0.25 * sample(albedo_gray, au, av) + 0.75 * sample(speckle, pu, pv);
        }
        img.at<std::uint8_t>(rv, ru) = cv::saturate_cast<std::uint8_t>(value);
      }
      char name[64];
      std::snprintf(name, sizeof name, "/frame_%06d_cam%u.pgm", f, stream);
      cv::imwrite(out + name, img);
    }
    std::printf("rendered frame %d/%d\r", f + 1, frames);
    std::fflush(stdout);
  }
  std::printf("\n");
  return 0;
}
