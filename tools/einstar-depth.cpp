// Computes registered depth for one captured frame set.
//
//   einstar-depth CALIB FRAME_DIR FRAME_ID [OUT_PREFIX]
//                 [--scale S] [--block B] [--uniqueness U] [--speckle-window W]
//                 [--min-depth MM] [--max-depth MM]
//
// Reads frame_<id>_cam{0,1,2}.pgm (einstar-probe --out), prints depth
// coverage and range, and writes OUT_PREFIX_depth.png (colour-mapped),
// OUT_PREFIX_color.png and OUT_PREFIX_rectified.png, all in calibration
// orientation.
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <string>
#include <vector>

#include "einstar/calibration.hpp"
#include "einstar/depth.hpp"

using namespace einstar;

int main(int argc, char** argv) {
  std::vector<std::string> pos;
  DepthParams params;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--scale" && i + 1 < argc) params.scale = std::atof(argv[++i]);
    else if (a == "--block" && i + 1 < argc) params.block_size = std::atoi(argv[++i]);
    else if (a == "--uniqueness" && i + 1 < argc) params.uniqueness_ratio = std::atoi(argv[++i]);
    else if (a == "--speckle-window" && i + 1 < argc) params.speckle_window = std::atoi(argv[++i]);
    else if (a == "--min-depth" && i + 1 < argc) params.min_depth_mm = static_cast<float>(std::atof(argv[++i]));
    else if (a == "--max-depth" && i + 1 < argc) params.max_depth_mm = static_cast<float>(std::atof(argv[++i]));
    else pos.push_back(a);
  }
  if (pos.size() < 3) {
    std::fprintf(stderr, "usage: %s CALIB FRAME_DIR FRAME_ID [OUT_PREFIX]\n", argv[0]);
    return 2;
  }
  auto calib = load_calibration(pos[0]);
  if (!calib.model) {
    std::fprintf(stderr, "calibration: %s\n", to_string(*calib.error));
    return 1;
  }
  cv::Mat img[3];
  for (int c = 0; c < 3; ++c) {
    char name[64];
    std::snprintf(name, sizeof name, "/frame_%06d_cam%d.pgm", std::atoi(pos[2].c_str()), c);
    img[c] = cv::imread(pos[1] + name, cv::IMREAD_GRAYSCALE);
    if (img[c].empty()) {
      std::fprintf(stderr, "cannot read %s%s\n", pos[1].c_str(), name);
      return 1;
    }
  }
  // Stream camera s carries calibration camera calibration_index_for_stream(s).
  cv::Mat left, right;
  for (unsigned stream = 0; stream < 2; ++stream)
    (calibration_index_for_stream(stream) == kIrLeftCamera ? left : right) =
        to_calibration_orientation(img[stream], stream);

  StereoDepth sd(*calib.model, img[0].size(), params);
  const cv::Mat depth = sd.depth_in_color(left, right);
  int valid = 0;
  double mn = 1e9, mx = 0, sum = 0;
  for (int v = 0; v < depth.rows; ++v)
    for (int u = 0; u < depth.cols; ++u)
      if (float d = depth.at<float>(v, u); d > 0) {
        ++valid;
        mn = std::min<double>(mn, d);
        mx = std::max<double>(mx, d);
        sum += d;
      }
  // Smoothness: share of depth pixels within 3 mm of their 5x5 median; real
  // surfaces score high, scattered mismatches low.
  cv::Mat med;
  cv::medianBlur(depth, med, 5);
  int smooth = 0;
  for (int v = 0; v < depth.rows; ++v)
    for (int u = 0; u < depth.cols; ++u) {
      const float d = depth.at<float>(v, u), m = med.at<float>(v, u);
      if (d > 0 && m > 0 && std::abs(d - m) < 3.f) ++smooth;
    }
  std::printf("smooth %.0f%% of valid, ", valid ? 100.0 * smooth / valid : 0.0);
  std::printf("depth coverage %.1f%%, range %.0f..%.0f mm, mean %.0f mm\n", 100.0 * valid / depth.total(), valid ? mn : 0,
              mx, valid ? sum / valid : 0);
  if (pos.size() >= 4) {
    cv::Mat d8, colored;
    depth.convertTo(d8, CV_8U, 255.0 / 800.0);
    cv::applyColorMap(d8, colored, cv::COLORMAP_TURBO);
    colored.setTo(cv::Scalar(0, 0, 0), depth == 0);
    cv::imwrite(pos[3] + "_depth.png", colored);
    cv::imwrite(pos[3] + "_color.png", sd.undistort_color(color_in_calibration_orientation(img[2])));
    // Rectified pair side by side with horizontal guide lines: matching
    // speckles must lie on the same row.
    cv::Mat lr, rr, pair;
    sd.rectify(left, right, lr, rr);
    cv::hconcat(lr, rr, pair);
    cv::cvtColor(pair, pair, cv::COLOR_GRAY2BGR);
    for (int y = 0; y < pair.rows; y += 40) cv::line(pair, {0, y}, {pair.cols, y}, cv::Scalar(0, 255, 0), 1);
    cv::imwrite(pos[3] + "_rectified.png", pair);
  }
  return 0;
}
