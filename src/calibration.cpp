#include "einstar/calibration.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace einstar {
namespace {

double le_double(ByteView b, std::size_t index) {
  std::uint64_t bits = 0;
  for (int i = 7; i >= 0; --i) bits = (bits << 8) | b[index * 8 + i];
  double d;
  std::memcpy(&d, &bits, sizeof d);
  return d;
}

std::int32_t le_i32(ByteView b, std::size_t off) {
  std::uint32_t u = std::uint32_t{b[off]} | (std::uint32_t{b[off + 1]} << 8) |
                    (std::uint32_t{b[off + 2]} << 16) | (std::uint32_t{b[off + 3]} << 24);
  return static_cast<std::int32_t>(u);
}

constexpr std::size_t kCameraOffsets[3] = {4, 38, 60};

// CCF field order: fx fy 0 0 cx cy 0 0 d5 0 k1 k2 p1 p2 k3 0 0 0 0 0 T(3) R(9).
constexpr std::size_t kCcfFields = 32;
// Offsets added to LeftCCF/RightCCF fields (identical for both files and
// recovered by comparison with an EXStar .data_cm of the same scanner).
constexpr double kCcfIrKey[kCcfFields] = {5, -7, 0, 0, 8, -3, 0, 0, 5, 0, 1, 0, 11, 9, 13, 0,
                                          0, 0, 0, 0, 1, -10, -1, -7, -6, -10, -1, 14, -3, -15, -11, -4};

CameraCalibration read_ccf(ByteView b, bool ir) {
  double f[kCcfFields];
  for (std::size_t i = 0; i < kCcfFields; ++i) f[i] = le_double(b, i) - (ir ? kCcfIrKey[i] : 0.0);
  CameraCalibration c;
  c.intrinsics.fx = f[0];
  c.intrinsics.fy = f[1];
  c.intrinsics.cx = f[4];
  c.intrinsics.cy = f[5];
  c.intrinsics.distortion = {f[10], f[11], f[12], f[13], f[14], f[8]};
  for (std::size_t i = 0; i < 3; ++i) c.world_to_camera.T[i] = f[20 + i];
  for (std::size_t i = 0; i < 9; ++i) c.world_to_camera.R[i] = f[23 + i];
  return c;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
constexpr std::size_t kPairOffset = 26;

CameraCalibration read_camera(ByteView b, std::size_t o) {
  CameraCalibration c;
  c.intrinsics.fx = le_double(b, o);
  c.intrinsics.fy = le_double(b, o + 1);
  c.intrinsics.cx = le_double(b, o + 2);
  c.intrinsics.cy = le_double(b, o + 3);
  for (std::size_t i = 0; i < 6; ++i) c.intrinsics.distortion[i] = le_double(b, o + 4 + i);
  for (std::size_t i = 0; i < 9; ++i) c.world_to_camera.R[i] = le_double(b, o + 10 + i);
  for (std::size_t i = 0; i < 3; ++i) c.world_to_camera.T[i] = le_double(b, o + 19 + i);
  return c;
}

Mat3 transpose(const Mat3& m) {
  return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]};
}

bool is_rotation(const Mat3& m) {
  const Mat3 t = transpose(m);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += m[i * 3 + k] * t[k * 3 + j];
      if (std::abs(s - (i == j ? 1.0 : 0.0)) > 1e-3) return false;
    }
  return std::abs(determinant(m) - 1.0) < 1e-3;
}

}  // namespace

double determinant(const Mat3& m) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
         m[2] * (m[3] * m[7] - m[4] * m[6]);
}

Vec3 Pose::apply(const Vec3& x) const {
  Vec3 y{};
  for (int i = 0; i < 3; ++i)
    y[i] = R[i * 3] * x[0] + R[i * 3 + 1] * x[1] + R[i * 3 + 2] * x[2] + T[i];
  return y;
}

Pose Pose::inverse() const {
  Pose p;
  p.R = transpose(R);
  for (int i = 0; i < 3; ++i)
    p.T[i] = -(p.R[i * 3] * T[0] + p.R[i * 3 + 1] * T[1] + p.R[i * 3 + 2] * T[2]);
  return p;
}

Pose Pose::operator*(const Pose& rhs) const {
  Pose p;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += R[i * 3 + k] * rhs.R[k * 3 + j];
      p.R[i * 3 + j] = s;
    }
  p.T = apply(rhs.T);
  return p;
}

Pose CameraModel::relative(unsigned from, unsigned to) const {
  return cameras[to].world_to_camera * cameras[from].world_to_camera.inverse();
}

const char* to_string(CalibrationError e) {
  switch (e) {
    case CalibrationError::WrongSize: return "camera model file is not 692 bytes";
    case CalibrationError::NotARotation: return "camera rotation is not orthonormal";
    case CalibrationError::MissingFile: return "calibration file missing or wrong size";
    case CalibrationError::BadDeviceBlock: return "device calibration block is missing or malformed";
  }
  return "unknown";
}

CalibrationResult parse_camera_model(ByteView b) {
  if (b.size() != kCameraModelFileSize) return {std::nullopt, CalibrationError::WrongSize};
  CameraModel m;
  for (std::size_t i = 0; i < 4; ++i) m.header[i] = le_double(b, i);
  for (std::size_t c = 0; c < 3; ++c) {
    m.cameras[c] = read_camera(b, kCameraOffsets[c]);
    if (!is_rotation(m.cameras[c].world_to_camera.R))
      return {std::nullopt, CalibrationError::NotARotation};
  }
  for (std::size_t i = 0; i < 9; ++i) m.pair_rotation_raw[i] = le_double(b, kPairOffset + i);
  for (std::size_t i = 0; i < 3; ++i) m.pair_translation[i] = le_double(b, kPairOffset + 9 + i);
  for (std::size_t i = 0; i < 4; ++i) m.tail_sizes[i] = le_i32(b, b.size() - 16 + i * 4);
  return {m, std::nullopt};
}

CalibrationResult load_camera_model(const std::string& path) {
  return parse_camera_model(read_file(path));
}

CalibrationResult parse_exstar_ccf(ByteView tex, ByteView left, ByteView right) {
  if (tex.size() != kCcfColorSize || left.size() != kCcfIrSize || right.size() != kCcfIrSize)
    return {std::nullopt, CalibrationError::MissingFile};
  CameraModel m;
  m.cameras[kColorCamera] = read_ccf(tex, false);
  m.cameras[kIrLeftCamera] = read_ccf(left, true);
  m.cameras[kIrRightCamera] = read_ccf(right, true);
  for (const auto& c : m.cameras)
    if (!is_rotation(c.world_to_camera.R)) return {std::nullopt, CalibrationError::NotARotation};
  m.pair_translation = m.relative(kIrLeftCamera, kColorCamera).T;
  m.tail_sizes = {640, 512, 320, 256};
  return {m, std::nullopt};
}

CalibrationResult parse_device_calibration(ByteView info) {
  // Offsets inside the CCF block (tagCCFData), little-endian int32 sizes.
  constexpr std::size_t kType = 0x000, kLeft = 0x108, kRight = 0x610, kTex = 0xB18, kMagic = 0xF1C;
  constexpr std::size_t kBlockSize = 0xF21;
  if (info.size() < kDeviceCcfBlockOffset + kBlockSize) return {std::nullopt, CalibrationError::BadDeviceBlock};
  const ByteView block(info.data() + kDeviceCcfBlockOffset, kBlockSize);
  if (std::memcmp(block.data() + kMagic, "FQFQ", 4) != 0 || le_i32(block, kType) != 4)
    return {std::nullopt, CalibrationError::BadDeviceBlock};
  auto slot = [&](std::size_t off, std::size_t expected) -> std::optional<ByteView> {
    if (le_i32(block, off) != static_cast<std::int32_t>(expected)) return std::nullopt;
    return ByteView(block.data() + off + 4, expected);
  };
  const auto left = slot(kLeft, kCcfIrSize), right = slot(kRight, kCcfIrSize), tex = slot(kTex, kCcfColorSize);
  if (!left || !right || !tex) return {std::nullopt, CalibrationError::BadDeviceBlock};
  return parse_exstar_ccf(*tex, *left, *right);
}

CalibrationResult load_calibration(const std::string& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) return load_exstar_ccf(path);
  const auto bytes = read_file(path);
  return bytes.size() == kDeviceInfoSize ? parse_device_calibration(bytes) : parse_camera_model(bytes);
}

CalibrationResult load_exstar_ccf(const std::string& dir) {
  return parse_exstar_ccf(read_file(dir + "/TexCCF.txt"), read_file(dir + "/LeftCCF.txt"),
                          read_file(dir + "/RightCCF.txt"));
}

}  // namespace einstar
