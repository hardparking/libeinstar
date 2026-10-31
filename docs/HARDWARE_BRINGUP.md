# Hardware bring-up

None of libeinstar's device code has run against a scanner yet. This is the
order to try it in, what each step proves, and what to report back.

## Before you start

- Quit EXStar. Its `SnSyncService` holds the USB device.
- Build with tools and RTAB-Map support (see the README). On macOS, libusb
  works without a driver; on Linux, add a udev rule for VID `3267` or run as root.

## Quick path

```sh
scripts/einstar-bringup.sh build
```

The script runs the steps below and bundles the logs, flash dump, frames and an
RTAB-Map replay into `einstar-bringup-<date>.tar.gz`.

## Steps and what they prove

| # | Command | Changes device state? | Proves |
|---|---|---|---|
| 1 | `einstar-probe` | no | VID `0x3267`, the PID and the product string (`EINSCAN10…`) |
| 2 | `einstar-probe --info` | no | handshake, command framing, serial, firmware, getter offsets |
| 2b | `einstar-probe --info --mask` | no | only if step 2 gets bad replies: the XOR mask is required |
| 3 | `einstar-probe --dump-flash f.bin` | no | bulk page reads and the on-device calibration decoder ("calibration decoded: … IR baseline ~160 mm") |
| 4 | `einstar-probe --scan --frames 30 --out dir` | trigger on/off (volatile) | trigger sequence, stream framing, camera ids |
| 5 | `einstar-rtabmap --calib f.bin --replay dir` | no | depth, registration and odometry on real frames |
| 6 | `einstar-rtabmap` | trigger on/off | live scanning, with calibration read from the scanner |
| 7 | `EINSTAR_RECORD_DIR=dir rtabmap` (GUI, source SHINING 3D Einstar) | trigger on/off | live scanning in the RTAB-Map app; with the variable set, every frame set it processes is also saved in the step 4 format (replay it with step 5) plus `record.tsv` with the IR exposure and depth coverage per frame |

Expected values for step 2: serial in the format EXStar logs
(16 hex digits), firmware `EinScan10_01_SC130_FX3_V2.14_FPGA_V3.7_EN` or
newer, sensor count 3.

## What to look at in the frames

- `frame_<id>_cam0/1.pgm` should show the IR speckle pattern, and `cam2` a
  Bayer-patterned colour view. If not, the camera field mapping
  (`FrameInfo::camera_index`) is wrong.
- If cam0 and cam1 seem swapped (depth is mostly empty in step 5), the
  stream-to-calibration mapping (`calibration_index_for_stream`) is reversed.
- Mostly empty depth can also mean the depth range is off: the default
  working range is 200–800 mm.

## Scanning in the RTAB-Map app

Start the app with settings that suited a mounted box on a turntable at
~30 cm (the distance EXStar's own frames used); RTAB-Map's defaults left
large gaps:

```
rtabmap --Odom/Strategy 1 --Vis/CorType 1 --Vis/MaxDepth 0.8 \
        --RGBD/LinearUpdate 0.01 --RGBD/AngularUpdate 0.05 \
        --Odom/ResetCountdown 1 --Rtabmap/DetectionRate 0
```

- `Odom/Strategy 1`, `Vis/CorType 1`: frame-to-frame odometry with optical
  flow tracked far better than the default frame-to-map with descriptors.
- `Odom/ResetCountdown 1`: after a lost frame, continue from the last pose
  instead of waiting to see the last tracked view again (each reset can
  leave a small jump in the map).
- `Rtabmap/DetectionRate 0`: add map nodes as fast as odometry delivers. At
  the default 1 Hz a turntable scan kept 35 of 376 frames and the cloud was
  separate patches; at 0 it kept 450 nodes and the surface was continuous.
- Visual + ICP registration (`Reg/Strategy 2`) split a turntable scan into
  misaligned halves; leave it off.

Then choose Detection > Select source > RGB-D camera > SHINING 3D Einstar,
File > New database (Start stays disabled until a database is open), and
Start. Quit with Cmd+Q so the app saves its settings. A stable mount and
slow motion mattered more than any setting: handheld scans lost tracking
hundreds of times where mounted ones lost it 0-3 times.

RTAB-Map maps places: it expects a camera moving through a still world. On
a turntable the object turns while the table and room stand still, and the
recorded turntable scans came out as smeared rings. Use it to check that
the scanner works as an RGB-D camera; scan objects with Open3D (below).

## Scanning objects with Open3D

`einstar-rgbd` writes depth and colour frames; `tools/einstar_fusion.py`
aligns each one by point-to-plane ICP to the model fused so far and fuses
it in (KinectFusion-style, as EXStar tracks), ignoring depth beyond
`--max-depth-mm` (default 300) so a still background does not hold the
tracking. In two terminals:

```
einstar-rgbd ~/scans/box                          # Ctrl-C to stop
python3 tools/einstar_fusion.py ~/scans/box --follow
```

The preview shows the colour view with the distance to the surface in the
centre box (green when in range) and the tracking state, next to the model
seen from the scanner. Press q to finish; the mesh is written to
`~/scans/box/scan_mesh.ply`. A recording (step 4 or 7 format) can be
converted with `einstar-rgbd OUT --calib calib.bin --replay DIR` and fused
without `--follow`. Needs `pip install numpy opencv-python open3d`.

On the two recorded turntable scans every frame aligned (494 and 823
frames) and the box came out with faces flat to 0.5-0.8 mm RMS; one scan
needed a 300 mm limit to drop a still surface at ~350 mm. Live, the scanner
delivered ~6 frames/s and fusion kept up: a box turned 3-4 times on a
turntable aligned in all 591 fused frames.

Scan one slow turn rather than several. On that live scan the long side of
a box face measured 179 mm (175-186 mm) in single depth frames, but 190 mm
after fusing the first 100 frames and 207-220 mm after all of them: small
alignment errors add up over turns and smear the model. Rejecting large
steps or coarse-to-fine ICP did not reduce this measurably. Accuracy
against a measured object is not yet checked.

## If something fails

| Symptom | Likely cause | Where to look |
|---|---|---|
| `no Einstar … found` but step 1 lists a device | product string differs | `starts_with_einscan10` in `src/device.cpp` |
| `libusb_claim_interface: ACCESS` | EXStar still running, or Linux permissions | quit EXStar / udev rule |
| `handshake reply` timeout | interface or endpoint mismatch | `kEpCommand*` in `protocol.hpp` |
| `bad reply: command group mismatch` / `opcode mismatch` | XOR mask on, or wrong framing | try `--mask` |
| stream frames never complete | stream header offsets | `stream.hpp`; save a raw transfer |
| colours look wrong | Bayer pattern | `COLOR_BayerRG2BGR` in `CameraEinstar.cpp` |

## Best evidence: a USB capture of EXStar

If the steps disagree with the code, one capture of EXStar connecting and
scanning for a few seconds settles most open questions (`docs/PROTOCOL.md`,
last section). On macOS, Wireshark can capture USB through the `XHC*`
interfaces once they are brought up with `sudo ifconfig XHC20 up`; the
interface name depends on the Mac. Filter on the scanner's address and save
a `.pcapng`. On Linux, use `usbmon`.
