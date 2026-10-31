// Parser for EXStar's per-project camera model file (*.data_cm, 692 bytes).
//
// Layout (little-endian float64 unless noted). Camera roles were established
// by matching EXStar's cached TexCCF/LeftCCF/RightCCF calibration files, and
// the pair translation by cross-checking the per-camera poses:
//   [0..3]    header, unverified (looks like rectified cx, cy, f, f)
//   [4..25]   camera 0 (colour)   : fx fy cx cy, d0..d5, R (row-major), T (mm)
//   [26..37]  pair block: R (semantics unverified), T = camera1 -> camera0 (mm)
//   [38..59]  camera 1 (IR "left", LeftCCF)  : same 22-value layout
//   [60..81]  camera 2 (IR "right", RightCCF): same 22-value layout
//   [82..]    tail, unverified; ends with int32 640 512 320 256
// Distortion d0..d5 is k1 k2 p1 p2 k3 and one further term (unverified).
// Per-camera R,T map a shared calibration frame into the camera:
//   x_cam = R * x_world + T.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "einstar/span.hpp"

namespace einstar {

inline constexpr std::size_t kCameraModelFileSize = 692;

// Calibration camera indices.
inline constexpr unsigned kColorCamera = 0;
inline constexpr unsigned kIrLeftCamera = 1;
inline constexpr unsigned kIrRightCamera = 2;

// Calibration index of a stream camera (FrameInfo::camera_index()), observed
// on hardware: stream 0 is the right IR camera, stream 1 the left IR camera
// and stream 2 the colour camera.
constexpr unsigned calibration_index_for_stream(unsigned stream_camera) {
  return stream_camera == 2 ? kColorCamera : (stream_camera == 0 ? kIrRightCamera : kIrLeftCamera);
}

// How a raw stream image relates to its calibration (observed on hardware):
// streams 0 and 2 arrive rotated by 180 degrees, stream 1 as calibrated.
enum class StreamOrientation { AsCalibrated, Rotated180 };
constexpr StreamOrientation stream_orientation(unsigned stream_camera) {
  return stream_camera == 1 ? StreamOrientation::AsCalibrated : StreamOrientation::Rotated180;
}

using Mat3 = std::array<double, 9>;  // row-major
using Vec3 = std::array<double, 3>;

struct Pose {
  Mat3 R{};
  Vec3 T{};
  Pose inverse() const;
  Pose operator*(const Pose& rhs) const;  // this ∘ rhs
  Vec3 apply(const Vec3& x) const;
};

struct CameraIntrinsics {
  double fx = 0, fy = 0, cx = 0, cy = 0;
  // Ordering assumed to follow OpenCV (k1 k2 p1 p2 k3 ...); d5 is unverified.
  std::array<double, 6> distortion{};
};

struct CameraCalibration {
  CameraIntrinsics intrinsics;
  Pose world_to_camera;
};

struct CameraModel {
  std::array<double, 4> header{};
  std::array<CameraCalibration, 3> cameras{};
  Mat3 pair_rotation_raw{};
  Vec3 pair_translation{};  // camera1 -> camera0, mm
  std::array<std::int32_t, 4> tail_sizes{};
  int image_width = 1280;
  int image_height = 1024;

  // Transform taking points from camera `from` into camera `to` (mm).
  Pose relative(unsigned from, unsigned to) const;
};

enum class CalibrationError { WrongSize, NotARotation, MissingFile, BadDeviceBlock };
const char* to_string(CalibrationError e);

struct CalibrationResult {
  std::optional<CameraModel> model;
  std::optional<CalibrationError> error;
};

CalibrationResult parse_camera_model(ByteView bytes);
CalibrationResult load_camera_model(const std::string& path);

// EXStar's per-scanner calibration cache: TexCCF.txt, LeftCCF.txt and
// RightCCF.txt (binary float64 despite the name), typically in
// EXStar.app/Contents/Resources/res/Einscan-E10/200x150/. Left/Right values
// carry a fixed integer offset per field, removed here; the result is
// rejected unless every rotation is orthonormal. Header and tail fields of
// the .data_cm format are not available from these files.
inline constexpr std::size_t kCcfColorSize = 264;
inline constexpr std::size_t kCcfIrSize = 520;
CalibrationResult parse_exstar_ccf(ByteView tex, ByteView left, ByteView right);
CalibrationResult load_exstar_ccf(const std::string& directory);

// The scanner's own calibration: the 6568-byte custom-info blob read from
// flash (Device::read_custom_info). EXStar copies a CCF block from offset
// 0x39B (Sn3DCalibDevFile: setDevData / readQuickCCFData) and writes its
// Left/Right/Tex slots to the CCF files verbatim; the block must carry type
// 4 and end with "FQFQ".
inline constexpr std::size_t kDeviceInfoSize = 6568;
inline constexpr std::size_t kDeviceCcfBlockOffset = 0x39B;
CalibrationResult parse_device_calibration(ByteView custom_info);

// A directory is read as a CCF cache, a 6568-byte file as a custom-info dump
// (einstar-probe --dump-flash), anything else as a .data_cm file.
CalibrationResult load_calibration(const std::string& path);

double determinant(const Mat3& m);

}  // namespace einstar
