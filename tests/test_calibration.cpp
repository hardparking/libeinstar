#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "check.hpp"
#include "einstar/calibration.hpp"

using namespace einstar;

namespace {

void put_double(std::vector<std::uint8_t>& b, std::size_t index, double v) {
  std::uint64_t bits;
  std::memcpy(&bits, &v, sizeof v);
  for (int i = 0; i < 8; ++i) b[index * 8 + i] = static_cast<std::uint8_t>(bits >> (8 * i));
}

Mat3 rot_y(double rad) {
  const double c = std::cos(rad), s = std::sin(rad);
  return {c, 0, s, 0, 1, 0, -s, 0, c};
}

void put_camera(std::vector<std::uint8_t>& b, std::size_t o, double f, const Pose& p) {
  const double k[10] = {f, f + 1, 640, 512, -0.15, 0.15, 1e-4, -1e-4, 0.05, 0};
  for (int i = 0; i < 10; ++i) put_double(b, o + i, k[i]);
  for (int i = 0; i < 9; ++i) put_double(b, o + 10 + i, p.R[i]);
  for (int i = 0; i < 3; ++i) put_double(b, o + 19 + i, p.T[i]);
}

std::vector<std::uint8_t> synthetic_model(Pose a, Pose b, Pose c) {
  std::vector<std::uint8_t> bytes(kCameraModelFileSize, 0);
  put_camera(bytes, 4, 1143, a);
  put_camera(bytes, 38, 1160, b);
  put_camera(bytes, 60, 1162, c);
  const Pose ba = a * b.inverse();
  for (int i = 0; i < 3; ++i) put_double(bytes, 35 + i, ba.T[i]);
  const std::int32_t sizes[4] = {640, 512, 320, 256};
  std::memcpy(bytes.data() + bytes.size() - 16, sizes, sizeof sizes);  // little-endian host
  return bytes;
}

void check_pose_near(const Pose& x, const Pose& y, double tol) {
  for (int i = 0; i < 9; ++i) CHECK_NEAR(x.R[i], y.R[i], tol);
  for (int i = 0; i < 3; ++i) CHECK_NEAR(x.T[i], y.T[i], tol);
}

}  // namespace

TEST(synthetic_model_parses_and_relative_poses_compose) {
  const Pose a{rot_y(0.1), {-6, 13, 186}};
  const Pose b{rot_y(-0.2), {59, 13, 200}};
  const Pose c{rot_y(0.05), {-25, 13, 193}};
  auto r = parse_camera_model(synthetic_model(a, b, c));
  CHECK(r.model.has_value());
  if (!r.model) return;
  const CameraModel& m = *r.model;
  CHECK_NEAR(m.cameras[0].intrinsics.fx, 1143.0, 1e-12);
  CHECK_NEAR(m.cameras[1].intrinsics.fy, 1161.0, 1e-12);
  CHECK_NEAR(m.cameras[2].intrinsics.distortion[0], -0.15, 1e-12);
  check_pose_near(m.relative(1, 0), a * b.inverse(), 1e-9);
  // The stored pair translation equals the camera1 -> camera0 translation.
  for (int i = 0; i < 3; ++i) CHECK_NEAR(m.pair_translation[i], m.relative(1, 0).T[i], 1e-9);
  check_pose_near(m.relative(0, 1) * m.relative(1, 0), Pose{{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}}, 1e-9);
  CHECK(m.tail_sizes[0] == 640 && m.tail_sizes[3] == 256);
}

TEST(wrong_size_and_non_rotation_are_rejected) {
  std::vector<std::uint8_t> small(100, 0);
  CHECK(parse_camera_model(small).error == CalibrationError::WrongSize);
  std::vector<std::uint8_t> zeros(kCameraModelFileSize, 0);
  CHECK(parse_camera_model(zeros).error == CalibrationError::NotARotation);
}

// Optional check against a real EXStar project file; set EINSTAR_CAMERA_MODEL
// to a *.data_cm path. Real calibration data is never committed.
TEST(real_camera_model_is_self_consistent_when_provided) {
  const char* path = std::getenv("EINSTAR_CAMERA_MODEL");
  if (!path) return;
  auto r = load_camera_model(path);
  CHECK(r.model.has_value());
  if (!r.model) return;
  const Pose rel = r.model->relative(1, 0);
  for (int i = 0; i < 3; ++i) CHECK_NEAR(r.model->pair_translation[i], rel.T[i], 1e-3);
}

// Optional: EXStar's CCF cache must decode to the same cameras as a .data_cm
// of the same scanner. Set EINSTAR_CCF_DIR and EINSTAR_CAMERA_MODEL.
TEST(exstar_ccf_cache_matches_camera_model_when_provided) {
  const char* dir = std::getenv("EINSTAR_CCF_DIR");
  const char* path = std::getenv("EINSTAR_CAMERA_MODEL");
  if (!dir || !path) return;
  auto ccf = load_exstar_ccf(dir);
  auto cm = load_camera_model(path);
  CHECK(ccf.model.has_value() && cm.model.has_value());
  if (!ccf.model || !cm.model) return;
  for (unsigned c = 0; c < 3; ++c) {
    const auto& a = ccf.model->cameras[c];
    const auto& b = cm.model->cameras[c];
    CHECK_NEAR(a.intrinsics.fx, b.intrinsics.fx, 1e-9);
    CHECK_NEAR(a.intrinsics.cy, b.intrinsics.cy, 1e-9);
    for (int i = 0; i < 6; ++i) CHECK_NEAR(a.intrinsics.distortion[i], b.intrinsics.distortion[i], 1e-9);
    for (int i = 0; i < 9; ++i) CHECK_NEAR(a.world_to_camera.R[i], b.world_to_camera.R[i], 1e-9);
    for (int i = 0; i < 3; ++i) CHECK_NEAR(a.world_to_camera.T[i], b.world_to_camera.T[i], 1e-9);
  }
}

// IR fields are offset by key words stored in the Left/Right files; both IR
// cameras share the combined words (left[i] +/- right[i]).
TEST(ccf_ir_key_words_from_the_files_are_removed) {
  const Mat3 rl = rot_y(0.17), rr = rot_y(-0.22);
  const double true_left[33] = {1154.5, 1154.25, 0, 0, 645.5, 505.0, 0, 0, 1e-4, 0, -0.16, 0.17, 5e-5, 6e-5, 0.012, 0,
                                0, 0, 0, 0, 64.0, -11.5, 199.5, rl[0], rl[1], rl[2], rl[3], rl[4], rl[5], rl[6],
                                rl[7], rl[8], 0};
  double true_right[33];
  std::copy(true_left, true_left + 33, true_right);
  true_right[20] = -21.5;
  for (int i = 0; i < 9; ++i) true_right[23 + i] = rr[i];
  const double tex[33] = {1152, 1152, 0, 0, 640, 512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0};
  // Key words: left 3*i - 40, right 17 - i; combined per word: "+" sums, "-" subtracts.
  const char* combine = "++-----++++----+++++----+-+++-+-++-+-+-++-+-+--+++-+-++-+++++---";
  auto combined = [&](int w) { return combine[w] == '+' ? (3 * w - 40) + (17 - w) : (3 * w - 40) - (17 - w); };
  const int use[22][3] = {{0, 0, -1},   {1, 1, 1},    {4, 2, 1},    {5, 3, -1},   {8, 4, -1},   {10, 5, -1},
                          {11, 6, 1},   {12, 7, -1},  {13, 8, -1},  {14, 9, -1},  {20, 50, 1},  {21, 51, 1},
                          {22, 52, 1},  {23, 33, 1},  {24, 34, -1}, {25, 35, 1},  {26, 36, -1}, {27, 37, -1},
                          {28, 38, -1}, {29, 39, 1},  {30, 40, 1},  {31, 41, 1}};
  auto ir_file = [&](const double* truth, bool left) {
    std::vector<std::uint8_t> b(kCcfIrSize, 0);
    double stored[33];
    std::copy(truth, truth + 33, stored);
    for (const auto& u : use) stored[u[0]] = truth[u[0]] - u[2] * combined(u[1]);
    for (int i = 0; i < 33; ++i) put_double(b, i, stored[i]);
    for (int w = 0; w < 64; ++w) {
      const std::int32_t v = left ? 3 * w - 40 : 17 - w;
      std::memcpy(b.data() + 264 + 4 * w, &v, 4);  // little-endian host
    }
    return b;
  };
  std::vector<std::uint8_t> t(kCcfColorSize, 0);
  for (int i = 0; i < 33; ++i) put_double(t, i, tex[i]);
  auto r = parse_exstar_ccf(t, ir_file(true_left, true), ir_file(true_right, false));
  CHECK(r.model.has_value());
  if (!r.model) return;
  using Truth = std::pair<unsigned, const double*>;
  for (const auto& [cam, truth] : {Truth{kIrLeftCamera, true_left}, Truth{kIrRightCamera, true_right}}) {
    const auto& c = r.model->cameras[cam];
    CHECK_NEAR(c.intrinsics.fx, truth[0], 1e-9);
    CHECK_NEAR(c.intrinsics.fy, truth[1], 1e-9);
    CHECK_NEAR(c.intrinsics.cx, truth[4], 1e-9);
    CHECK_NEAR(c.intrinsics.cy, truth[5], 1e-9);
    CHECK_NEAR(c.intrinsics.distortion[5], truth[8], 1e-12);
    for (int i = 0; i < 5; ++i) CHECK_NEAR(c.intrinsics.distortion[i], truth[10 + i], 1e-12);
    for (int i = 0; i < 3; ++i) CHECK_NEAR(c.world_to_camera.T[i], truth[20 + i], 1e-9);
    for (int i = 0; i < 9; ++i) CHECK_NEAR(c.world_to_camera.R[i], truth[23 + i], 1e-12);
  }
  CHECK_NEAR(r.model->cameras[kColorCamera].intrinsics.fx, 1152, 0.0);
}

TEST(ccf_with_wrong_sizes_is_rejected) {
  std::vector<std::uint8_t> small(10, 0);
  CHECK(parse_exstar_ccf(small, small, small).error == CalibrationError::MissingFile);
}

namespace {

std::vector<std::uint8_t> read_all(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Places CCF slots into a custom-info blob at the offsets EXStar reads.
std::vector<std::uint8_t> device_blob(const std::vector<std::uint8_t>& tex, const std::vector<std::uint8_t>& left,
                                      const std::vector<std::uint8_t>& right) {
  std::vector<std::uint8_t> blob(kDeviceInfoSize, 0);
  std::uint8_t* block = blob.data() + kDeviceCcfBlockOffset;
  auto put_i32 = [](std::uint8_t* p, std::int32_t v) { std::memcpy(p, &v, 4); };  // little-endian host
  put_i32(block, 4);
  auto slot = [&](std::size_t off, const std::vector<std::uint8_t>& data) {
    put_i32(block + off, static_cast<std::int32_t>(data.size()));
    std::copy(data.begin(), data.end(), block + off + 4);
  };
  slot(0x108, left);
  slot(0x610, right);
  slot(0xB18, tex);
  std::memcpy(block + 0xF1C, "FQFQ", 4);
  return blob;
}

}  // namespace

TEST(device_blob_without_magic_or_type_is_rejected) {
  std::vector<std::uint8_t> blob(kDeviceInfoSize, 0);
  CHECK(parse_device_calibration(blob).error == CalibrationError::BadDeviceBlock);
  std::memcpy(blob.data() + kDeviceCcfBlockOffset + 0xF1C, "FQFQ", 4);
  CHECK(parse_device_calibration(blob).error == CalibrationError::BadDeviceBlock);  // type 0
  CHECK(parse_device_calibration(ByteView(blob.data(), 100)).error == CalibrationError::BadDeviceBlock);
}

// Optional: a blob assembled from a real CCF cache decodes to the same model.
TEST(device_blob_decodes_like_the_ccf_cache_when_provided) {
  const char* dir = std::getenv("EINSTAR_CCF_DIR");
  if (!dir) return;
  const std::string d = dir;
  const auto blob = device_blob(read_all(d + "/TexCCF.txt"), read_all(d + "/LeftCCF.txt"), read_all(d + "/RightCCF.txt"));
  auto from_blob = parse_device_calibration(blob);
  auto from_files = load_exstar_ccf(d);
  CHECK(from_blob.model.has_value() && from_files.model.has_value());
  if (!from_blob.model || !from_files.model) return;
  for (unsigned c = 0; c < 3; ++c) {
    CHECK_NEAR(from_blob.model->cameras[c].intrinsics.fx, from_files.model->cameras[c].intrinsics.fx, 0.0);
    for (int i = 0; i < 3; ++i)
      CHECK_NEAR(from_blob.model->cameras[c].world_to_camera.T[i], from_files.model->cameras[c].world_to_camera.T[i], 0.0);
  }
}
