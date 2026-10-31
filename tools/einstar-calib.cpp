// Converts an EXStar calibration into OpenCV-style YAML.
//
//   einstar-calib Project1.data_cm > einstar.yaml
//   einstar-calib /Applications/EXStar.app/Contents/Resources/res/Einscan-E10/200x150
//   einstar-calib custom-info.bin      (from einstar-probe --dump-flash)
#include <cstdio>

#include "einstar/calibration.hpp"

using namespace einstar;

namespace {

void print_matrix(const char* name, int rows, int cols, const double* d) {
  std::printf("%s: !!opencv-matrix\n  rows: %d\n  cols: %d\n  dt: d\n  data: [", name, rows, cols);
  for (int i = 0; i < rows * cols; ++i) std::printf("%s%.12g", i ? ", " : "", d[i]);
  std::printf("]\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <file.data_cm | EXStar CCF directory | custom-info.bin>\n", argv[0]);
    return 2;
  }
  auto r = load_calibration(argv[1]);
  if (!r.model) {
    std::fprintf(stderr, "error: %s\n", to_string(*r.error));
    return 1;
  }
  const CameraModel& m = *r.model;
  static const char* kNames[] = {"color", "ir_left", "ir_right"};
  std::printf("%%YAML:1.0\n---\nimage_width: %d\nimage_height: %d\n", m.image_width, m.image_height);
  for (unsigned c = 0; c < 3; ++c) {
    const auto& in = m.cameras[c].intrinsics;
    const double K[9] = {in.fx, 0, in.cx, 0, in.fy, in.cy, 0, 0, 1};
    char key[64];
    std::snprintf(key, sizeof key, "%s_K", kNames[c]);
    print_matrix(key, 3, 3, K);
    std::snprintf(key, sizeof key, "%s_D", kNames[c]);
    print_matrix(key, 1, 5, in.distortion.data());
  }
  for (unsigned c : {kColorCamera, kIrRightCamera}) {
    // Transform from ir_left into camera c (x_c = R x_ir_left + T, mm).
    const Pose p = m.relative(kIrLeftCamera, c);
    char key[64];
    std::snprintf(key, sizeof key, "%s_R", kNames[c]);
    print_matrix(key, 3, 3, p.R.data());
    std::snprintf(key, sizeof key, "%s_T", kNames[c]);
    print_matrix(key, 3, 1, p.T.data());
  }
  return 0;
}
