# Add SHINING 3D Einstar camera driver (libeinstar)

Draft description for the upstream RTAB-Map pull request of
`0001-Add-SHINING-3D-Einstar-camera-driver-libeinstar.patch`.

---

Adds `CameraEinstar`, an RGB-D driver for the SHINING 3D Einstar handheld
3D scanner, backed by [libeinstar](https://github.com/hardparking/libeinstar)
(BSD 3-Clause), an open-source userspace driver for the scanner.

The Einstar has two IR cameras around a speckle projector and a colour
camera. libeinstar runs stereo matching on the IR pair and registers depth
to the undistorted colour camera, so the driver delivers ordinary RGB-D
frames (1280x1024, depth in mm). The factory calibration is read from the
scanner, so no calibration file is needed.

## Changes

- `CameraEinstar` (corelib), built when `WITH_EINSTAR` is on and
  `find_package(einstar)` finds libeinstar with its OpenCV depth component.
  `RTABMAP_EINSTAR` in `Version.h`; `find_dependency(einstar)` in
  `RTABMapConfig.cmake`.
- GUI: Preferences > Source > RGB-D > SHINING 3D Einstar, and
  Detection > Select source > RGB-D camera > SHINING 3D Einstar > Einstar.
- `rtabmap-camera` driver 18, with `-calib` and `-replay` to play back
  captures recorded with libeinstar's tools.
- IR exposure follows the scene (libeinstar `AutoExposure`); the scanner's
  light pulses with the exposure, so the controller keeps it steady.

The driver only sends volatile settings (exposure, gain, lighting, scan
trigger); libeinstar has no flash write or reboot commands.

## Testing

- macOS 27.2 (Apple silicon), Release, Qt 6: built with and without
  libeinstar (`WITH_EINSTAR=OFF` compiles the touched files with the driver
  stubbed out).
- Live with an Einstar (firmware `EinScan10_01_SC130_FX3_V2.14_FPGA_V3.7_EN`)
  in the RTAB-Map app: mounted turntable scans of a box. Recordings were
  replayed through the same driver with libeinstar's `einstar-rtabmap`.
- Not tested: `rtabmap-camera` driver 18 itself, Linux and Windows builds,
  other firmware versions.

## Known limitations

- ~6-7 frames/s: depth is computed on the CPU (OpenCV SGBM).
- The colour image is not white-balanced yet (greenish).
- Mapping works as for other RGB-D cameras. For scanning small objects on a
  turntable, RTAB-Map's still-world assumption is a poor fit; libeinstar
  documents a KinectFusion-style alternative.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
