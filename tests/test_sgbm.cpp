// The Metal SGBM must reproduce OpenCV's StereoSGBM (MODE_SGBM_3WAY) bit
// for bit. Compares the two on synthetic speckle pairs covering the edge
// cases (image borders, stripe boundaries, block sizes, uniqueness off), and
// on real frames when EINSTAR_CALIB and EINSTAR_FRAMES are set:
//
//   EINSTAR_CALIB=custom-info.bin EINSTAR_FRAMES=scan1 [EINSTAR_FRAME_IDS=0,100,200] einstar-sgbm-test
//
// EINSTAR_BENCH=1 also times both on the real frames.
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "einstar/calibration.hpp"
#include "einstar/depth.hpp"
#include "sgbm.hpp"

using namespace einstar;
using detail::SgbmConfig;

namespace {

// Speckle texture warped by a smooth disparity field, with sensor noise.
void synthetic_pair(cv::Size size, int max_disp, int seed, cv::Mat& left, cv::Mat& right) {
  cv::RNG rng(seed);
  cv::Mat tex(size.height, size.width + 2 * max_disp + 8, CV_8U);
  rng.fill(tex, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(tex, tex, {3, 3}, 0.8);
  cv::Mat mapx(size, CV_32F), mapy(size, CV_32F);
  for (int y = 0; y < size.height; ++y)
    for (int x = 0; x < size.width; ++x) {
      // Left column x matches right column x - d. A slanted plane with a
      // raised box: slopes, a step edge, occlusion.
      double d = 0.2 * max_disp + 0.4 * max_disp * x / size.width;
      if (std::abs(x - size.width / 2) < size.width / 6 && std::abs(y - size.height / 2) < size.height / 5)
        d += 0.2 * max_disp;
      mapx.at<float>(y, x) = static_cast<float>(x + d + max_disp);
      mapy.at<float>(y, x) = static_cast<float>(y);
    }
  cv::Mat ident_x(size, CV_32F), ident_y(size, CV_32F);
  for (int y = 0; y < size.height; ++y)
    for (int x = 0; x < size.width; ++x) {
      ident_x.at<float>(y, x) = static_cast<float>(x + max_disp);
      ident_y.at<float>(y, x) = static_cast<float>(y);
    }
  cv::remap(tex, left, ident_x, ident_y, cv::INTER_LINEAR);
  cv::remap(tex, right, mapx, mapy, cv::INTER_LINEAR);
  cv::Mat noise(size, CV_16S);
  rng.fill(noise, cv::RNG::NORMAL, 0, 3);
  for (cv::Mat* m : {&left, &right}) {
    cv::Mat s;
    m->convertTo(s, CV_16S);
    s += noise;
    s.convertTo(*m, CV_8U);
  }
}

int compare(const char* name, const cv::Mat& l, const cv::Mat& r, const SgbmConfig& c) {
  const cv::Mat cpu = detail::sgbm_cpu(l, r, c);
  const cv::Mat gpu = detail::sgbm_metal(l, r, c);
  const int diff = cv::countNonZero(cpu != gpu);
  const int valid = cv::countNonZero(cpu >= 0);
  std::printf("%-44s %5dx%-4d D=%-4d block %d: %d/%d pixels differ (%d valid)\n", name, l.cols, l.rows,
              c.num_disparities, c.block_size, diff, l.cols * l.rows, valid);
  if (diff) {
    for (int y = 0; y < cpu.rows; ++y)
      for (int x = 0; x < cpu.cols; ++x)
        if (cpu.at<short>(y, x) != gpu.at<short>(y, x)) {
          std::printf("  first difference at (%d, %d): cpu %d gpu %d\n", x, y, cpu.at<short>(y, x),
                      gpu.at<short>(y, x));
          return 1;
        }
  }
  return diff ? 1 : 0;
}

SgbmConfig config(int D, int block, int uniqueness = 20) {
  SgbmConfig c;
  c.num_disparities = D;
  c.block_size = block;
  c.p1 = 8 * block * block;
  c.p2 = 96 * block * block;
  c.uniqueness_ratio = uniqueness;
  return c;
}

int synthetic_cases() {
  int failures = 0;
  struct Case {
    const char* name;
    cv::Size size;
    int D, block, uniqueness;
  };
  const Case cases[] = {
      {"small", {200, 120}, 64, 5, 20},
      {"block 3", {240, 97}, 48, 3, 20},
      {"block 7, uniqueness off", {256, 131}, 80, 7, 0},
      {"16 disparities", {160, 64}, 16, 5, 20},
      {"tiny height (empty stripes)", {120, 5}, 32, 5, 20},
      {"D not a multiple of 32", {300, 150}, 112, 5, 20},
      {"realistic (scale 0.5, stretch 1)", {1501, 610}, 608, 5, 20},
  };
  for (const Case& t : cases) {
    cv::Mat l, r;
    synthetic_pair(t.size, t.D, 7, l, r);
    failures += compare(t.name, l, r, config(t.D, t.block, t.uniqueness));
  }
  return failures;
}

// The padded pair StereoDepth matches for its first stretch.
int real_cases(const char* calib_path, const char* frames) {
  auto calib = load_calibration(calib_path);
  if (!calib.model) {
    std::printf("cannot load %s\n", calib_path);
    return 1;
  }
  std::vector<int> ids = {0, 100, 200};
  if (const char* s = std::getenv("EINSTAR_FRAME_IDS")) {
    ids.clear();
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) ids.push_back(std::atoi(t.c_str()));
  }
  const bool bench = std::getenv("EINSTAR_BENCH") != nullptr;
  int failures = 0;
  for (const int id : ids) {
    cv::Mat img[2];
    for (int c = 0; c < 2; ++c) {
      char name[64];
      std::snprintf(name, sizeof name, "/frame_%06d_cam%d.pgm", id, c);
      img[c] = cv::imread(std::string(frames) + name, cv::IMREAD_GRAYSCALE);
      if (img[c].empty()) {
        std::printf("cannot read %s%s\n", frames, name);
        return failures + 1;
      }
    }
    cv::Mat left, right;
    for (unsigned stream = 0; stream < 2; ++stream)
      (calibration_index_for_stream(stream) == kIrLeftCamera ? left : right) =
          to_calibration_orientation(img[stream], stream);
    const DepthParams params;
    StereoDepth sd(*calib.model, img[0].size(), params);
    cv::Mat lr, rr;
    sd.rectify(sd.left_index() == kIrLeftCamera ? left : right, sd.left_index() == kIrLeftCamera ? right : left,
               lr, rr);
    auto clahe = cv::createCLAHE(3.0, {16, 16});
    clahe->apply(lr, lr);
    clahe->apply(rr, rr);
    const int D = sd.num_disparities(), min_d = sd.min_disparity();
    const int pad_l = std::max(0, -min_d), pad_r = std::max(0, min_d);
    const int width = lr.cols + D + std::max(pad_l, pad_r);
    cv::Mat lpad, rpad;
    cv::copyMakeBorder(lr, lpad, 0, 0, D + pad_l, width - lr.cols - D - pad_l, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(rr, rpad, 0, 0, D + pad_r, width - rr.cols - D - pad_r, cv::BORDER_CONSTANT, 0);
    const int b = params.block_size;
    SgbmConfig c = config(D, b, params.uniqueness_ratio);
    c.p1 = params.p1_factor * b * b;
    c.p2 = params.p2_factor * b * b;
    c.speckle_window = params.speckle_window;
    c.speckle_range = params.speckle_range;
    char name[64];
    std::snprintf(name, sizeof name, "real frame %d", id);
    failures += compare(name, lpad, rpad, c);
    if (bench) {
      auto time = [&](auto f) {
        f();  // warm-up (and shader build)
        const auto t0 = std::chrono::steady_clock::now();
        constexpr int n = 5;
        for (int i = 0; i < n; ++i) f();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
      };
      const double cpu = time([&] { detail::sgbm_cpu(lpad, rpad, c); });
      const double gpu_raw = time([&] { detail::sgbm_metal_raw(lpad, rpad, c); });
      const double gpu = time([&] { detail::sgbm_metal(lpad, rpad, c); });
      std::printf("  bench: cpu %.1f ms, metal %.1f ms (GPU part %.1f ms)\n", cpu, gpu, gpu_raw);
    }
    // End to end: StereoDepth with either backend gives the same depth.
    DepthParams pc = params, pm = params;
    pc.backend = MatchBackend::Cpu;
    pm.backend = MatchBackend::Metal;
    const StereoDepth dc(*calib.model, img[0].size(), pc), dm(*calib.model, img[0].size(), pm);
    const cv::Mat zc = dc.depth_in_color(left, right), zm = dm.depth_in_color(left, right);
    const int zdiff = cv::countNonZero(zc != zm);
    std::printf("  depth_in_color: %d pixels differ between backends (%d with depth)\n", zdiff, cv::countNonZero(zc > 0));
    failures += zdiff != 0;
    if (bench) {
      auto time = [&](const StereoDepth& sd) {
        sd.depth_in_color(left, right);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) sd.depth_in_color(left, right);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5;
      };
      std::printf("  depth_in_color bench: cpu %.1f ms, metal %.1f ms\n", time(dc), time(dm));
    }
  }
  return failures;
}

}  // namespace

int main() {
  if (!detail::sgbm_metal_available()) {
    std::printf("Metal unavailable: skipped\n");
    return 0;
  }
  int failures = synthetic_cases();
  const char* calib = std::getenv("EINSTAR_CALIB");
  const char* frames = std::getenv("EINSTAR_FRAMES");
  if (calib && frames) failures += real_cases(calib, frames);
  std::printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
  return failures ? 1 : 0;
}
