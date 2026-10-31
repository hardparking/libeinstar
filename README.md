# libeinstar

An open-source userspace driver for the SHINING 3D Einstar handheld scanner
(internal name EinScan10 / "E10"), built for interoperability with open
3D-scanning software. The first integration target is
[RTAB-Map](https://github.com/introlab/rtabmap).

**Status: tested on hardware.** The protocol was reconstructed from static
analysis of EXStar 3.1.3 and EXStar's own runtime logs, then checked against a
real Einstar (firmware `EinScan10_01_SC130_FX3_V2.14_FPGA_V3.7_EN`) on macOS:
device info, calibration read, streaming, depth, live scanning in the RTAB-Map
app and object scans fused with Open3D. Linux, Windows and other firmware
versions are untested, and scan accuracy has not been measured against a
reference object. See [docs/PROTOCOL.md](docs/PROTOCOL.md); every fact is tagged
as verified on hardware, observed, derived, inferred or unknown.

**Testing with a scanner?** Start with [docs/HARDWARE_BRINGUP.md](docs/HARDWARE_BRINGUP.md)
or run `scripts/einstar-bringup.sh build`.

## What works without hardware

- Command codec, reply validation, vendor transfer lengths, XOR mask (`einstar::core`)
- Image stream frame reassembly
- Calibration read from the scanner itself (`Device::read_calibration`), or
  from an EXStar project (`*.data_cm`), EXStar's CCF cache
  (`EXStar.app/Contents/Resources/res/Einscan-E10/200x150`) or a
  `--dump-flash` file; `einstar-calib` exports any of them as OpenCV YAML
- `einstar::depth` (OpenCV): IR-stereo depth registered to the colour camera
- RTAB-Map camera driver plus `einstar-rtabmap`, `einstar-synth` (synthetic
  sequences rendered through a real calibration) and `eval_trajectory.py`
- Tests: `ctest --test-dir build`; set `EINSTAR_CAMERA_MODEL` and
  `EINSTAR_CCF_DIR` to also check real calibration data

## Hardware tools

- `einstar-probe` lists devices with VID `0x3267`.
- `einstar-probe --info` runs read-only queries: serial, product, firmware,
  trigger state, exposure, gain, temperature.
- `einstar-probe --dump-flash calib.bin` reads the 6568-byte calibration blob.
- `einstar-probe --scan --frames 20 --out DIR` starts EXStar's trigger
  sequence, saves frames as PGM, then stops the trigger.
- `einstar-rgbd DIR` scans and writes registered depth + colour frames;
  `tools/einstar_fusion.py DIR --follow` fuses them into a mesh while you
  scan (see [object scanning](docs/HARDWARE_BRINGUP.md#scanning-objects-with-open3d)).

## Safety

The library is read-only unless a caller opts in with `OpenOptions::allow_writes`.
Opting in only enables volatile settings: exposure, gain, trigger and lighting.
Flash writes (opcode `0x58`), reboot (system opcode `0x08`) and firmware update
are not implemented.

## Build

```sh
brew install libusb cmake ninja opencv     # macOS; use distro packages on Linux
cmake -S . -B build -G Ninja
cmake --build build && ctest --test-dir build
cmake --install build --prefix /usr/local  # provides find_package(einstar)
```

For RTAB-Map, add `-DEINSTAR_BUILD_RTABMAP=ON`. That builds against an
RTAB-Map that already contains `CameraEinstar` (the proposed upstream patch,
`WITH_EINSTAR`, also adds Einstar to Preferences > Source > RGB-D), or compiles
the identical copy in `integrations/rtabmap`.
The upstream change itself is in `integrations/rtabmap/upstream/`, made against
RTAB-Map master `8cd2f81`. Apply it with
`git am integrations/rtabmap/upstream/*.patch` in an RTAB-Map checkout.

## Provenance

This project contains no SHINING 3D code, binaries or firmware. Protocol
knowledge comes from observing the vendor software's behaviour, for
interoperability. Do not commit vendor binaries or personal calibration files.

## License

BSD-3-Clause; see [LICENSE](LICENSE).
