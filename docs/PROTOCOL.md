# Einstar (EinScan10 / "E10") USB protocol

Reconstructed for interoperability from static analysis of EXStar 3.1.3
(macOS arm64) and EXStar's runtime logs, then checked against a real Einstar
(firmware `EinScan10_01_SC130_FX3_V2.14_FPGA_V3.7_EN`) with libeinstar.

Tags: **[hw]** verified on hardware · **[obs]** read directly from code or
logs · **[der]** derived from disassembly or decompilation · **[inf]**
inference · **[?]** unknown.

## Verified on hardware [hw]

- USB: VID `0x3267`, PID `0x0003`, vendor string "Shining3d", product string
  **"SHINING-CAM"**, no USB serial. Seen at 480 Mb/s (USB 2.0) on the test Mac.
- Handshake reply bytes 9..16 are ASCII **"Shining3"**. The device serial
  comes from system opcode `0x04`; system `0x01` returns "EinScan10_01".
- **Replies**: byte 1 = `0x02`, bytes 0/2/3 echo the request, byte 4 =
  status, **byte 8 = value length N**, value at 9..9+N (big-endian). Max
  width and height are 16-bit (1280 × 1024); exposure is 32-bit (default 2184,
  range 1–10000); gain is 16-bit (default 500, range 1–800); pixel bits 8;
  trigger period default 66666.
- **Flash pages** answer only when exchanged as the vendor does: 1 KiB bulk
  transfers out and back (`CEinScan10::bulkIO`). The calibration block decodes
  identically to EXStar's CCF cache.
- **Trigger**: `0x49` 200000 → `0x41` `0x11` → `0x49` period starts frames;
  `0x41` `0x00` stops them. The frame id counter persists across starts.
  The first frame set after starting is overexposed; discard it.
- **Lighting** is off by default. EXStar-like values that light the IR
  speckle: strobe route 0 = 1000, route 1 = 300, laser 100, IR exposure
  4000 µs / gain 105, colour 5000 µs / gain 180. Which route drives which
  emitter is [?].
- **Streams**: stream 0 = IR right (RightCCF), stream 1 = IR left (LeftCCF),
  stream 2 = colour (TexCCF). Relative to the calibration, **streams 0 and 2
  arrive rotated by 180°** and stream 1 as calibrated (established by depth
  coverage across all pairings and flips, and by depth/colour overlays).
- Stream framing (32-byte header, last-packet flag, payload at 0x20) produces
  coherent 1280×1024 frames.

## Software stack [obs]

```
EXStar (Qt GUI)
  libSn3DDeviceEinStar.dylib --qttunnel (local hub, 127.0.0.1:1883)--+
                                                                      v
SnSyncService.app (Frameworks/syncservice/)
  logic/3DDigitalSyncInterface.2.64.8-alpha.5.dylib   InternalE10Device: property names -> device calls
  device/einscan10.2.0.0.dylib                        CEinScan10: packet builders, ImageStreamThread
  Frameworks/libsn3DCommunicateLayerM.2.1.8.dylib     CCommunicateManager: libusb transport
Images reach the GUI through a 2 GiB shared-memory block named "demo".
```

Reconstruction (speckle fusion, tracking, meshing) is in proprietary
`libSn3DSpeckleFusion`, `libSn3DRealtimeScan`, `libSn3DScanSlam` and others;
an open stack has to supply its own.

## USB transport

| Item | Value | Tag |
|---|---|---|
| Vendor ID | `0x3267`, hard-coded in `moniteDeviceThread` | [obs] |
| Product ID | every PID under the VID is enumerated; the plugin keeps devices whose product string starts with `EINSCAN10` | [der] |
| Interface | 0 | [obs] |
| Command out / in | EP `0x01` / `0x81`, interrupt | [obs] |
| Flash pages out / in | EP `0x02` / `0x82`, bulk | [obs] |
| Image stream | EP `0x83`, async bulk, `0xA400` bytes per transfer | [obs] |
| Hardware | Cypress FX3 + FPGA, firmware `EinScan10_01_SC130_FX3_V2.14_FPGA_V3.7_EN`, 3 sensors | [obs, log] |

**Attach** (`CCommunicateManager::attach`) [der]: open by VID/PID, claim
interface 0, drain up to 5 interrupt and 30 bulk replies, clear halts, then
send a 100-byte buffer with `buf[0] = 0xFF`. Reply bytes 9..16 are the serial,
printed `%02X` × 8 (matches the serial format in EXStar's log).

**Optional XOR mask** [der]: key `k = n | (n << 4)` for a random nibble `n`,
XORed over the whole packet. The receiver recovers `n` from the high nibble of
byte 1, which is zero in plaintext. Whether the Einstar enables it is **[?]**.

## Command packets [der]

```
off 0     packet id      rolling 0..253 (mod 0xFE)
off 1     0x00
off 2     group          0x10 camera/config, 0x00 system
off 3     opcode
off 4     status         0 in requests; replies must be 0
off 5..6  0
off 7     payload length
off 8..   payload        big-endian; getters echo the selector at 8, value at 9
```

The request is zero-padded to a per-command transfer length (10, 16, 50, 64
or 100 bytes). A reply is accepted when bytes 0, 2 and 3 echo the request and
byte 4 is 0. Otherwise the plugin clears the command pipe, sleeps 20 ms and
retries, up to 5 times.

### Camera group (byte 2 = `0x10`)

| Op | Meaning | Payload / reply | Wire | Tag |
|---|---|---|---|---|
| 0x00 | vendor id | | 16 | [der] |
| 0x01 | product id | | 16 | [der] |
| 0x16 / 0x17 | max width / height | sel = camera mask | 16 | [der] |
| 0x20 / 0x21 | max / min exposure | sel | 16 | [der] |
| 0x22 / 0x23 | get / **set exposure** | set: `mask, u32` (len 5); get: u32 at 9 | 16 | [der] |
| 0x24 / 0x25 | max / min gain | sel | 16 | [der] |
| 0x26 / 0x27 | get / **set gain** | set: `mask, u16` (len 3); get: u16 at 9 | 16 | [der] |
| 0x2e | pixel bits | sel | 16 | [der] |
| 0x40 / 0x41 | get / **set trigger counts** | one byte: RGB count << 4 \| mono count | 10 | [der] |
| 0x48 / 0x49 | get / **set trigger period** | u32 (len 4) | 50 | [der] |
| 0x50 | temperature | BE16 at 9, `>> 7` (LM75 style, units [?]) | 16 | [der] |
| 0x51 | sensor count | u8 at 9 (log: 3) | 16 | [der] |
| 0x57 / 0x58 | flash page read / **write** | via bulk, see below | | [der] |
| 0x5d | colour mode | sel | 16 | [der] |
| 0x62 | distance/state indication | 2 bytes | 16 | [der] |
| 0x67 / 0x68 | get / set laser/LED brightness | u8, clamped to 100 | 50 | [der] |
| 0x6f / 0x70 | get / set strobe brightness | `route, u16` (len 3) | 50 | [der] |
| 0x7b | heartbeat / button state | | 10 | [der] |

Camera mask: camera 0 = 1, camera 1 = 2, camera 2 = 4 (the plugin computes
`cam == 2 ? 4 : cam + 1`).

### System group (byte 2 = `0x00`)

| Op | Meaning | Reply | Wire | Tag |
|---|---|---|---|---|
| 0x01 | product name | NUL-terminated string at 9 | 100 | [der] |
| 0x04 | serial | 8 bytes at 9 | 64 | [der] |
| 0x05 | firmware version | NUL-terminated string at 9 | 64 | [der] |
| 0x08 | **reboot** | | 10 | [der] |

libeinstar refuses `0x58` (flash write) and system `0x08` (reboot).

## Property names to device calls [der]

`InternalE10Device` (logic plugin) maps EXStar property names onto
`CEinScan10`:

| Property | Device effect |
|---|---|
| `CameraExposure` [cam, µs] | 0x23 |
| `CameraGain` [cam, gain] | 0x27 |
| `ScanTriggerPeriod` value | 0x49 |
| `TriggerMode` {MONOTRIGCOUNT, RGBTRIGCOUNT} | 0x49 = 200000, then 0x41 = `rgb << 4 \| mono`; 1 s settle when rgb = 0 |
| `StrobeLuminance` {ROUTEINDEX, LUMINANCE} | 0x70 |
| `LDRGBLuminance` value | 0x68 (clamped to 100) [inf: key/opcode pairing from call order] |
| `Indication` {DEVICESTATE, DISTANCE} | 0x62 |
| `HeartBeatPeriod`, `EnableRGB`, `YGamma` {DIAT1, DIAT3} | no direct device command found [inf: host-side] |

**Connect sequence** (`initFunc`): set HeartBeatPeriod, set
ScanTriggerPeriod, read Serial, `startStream`. **`startStream` sends no
command**: it only starts the host thread that queues bulk reads on EP 0x83.
**Frames flow once the trigger is enabled** (TriggerMode 1/1), and
`closeTrigger` sets both counts to 0.

Observed EXStar session (log): openDevices{E10} → Serial, FirmwareVersion,
SensorCount → ReadData(0, 6568) → ProtectCapture → EnableRGB → YGamma →
TriggerMode{1,1} → ScanTriggerPeriod 90000 → ImageStream on →
exposure 1500–5000, gain 120–400, strobe and indication updates.
90000 is likely µs (~11 Hz), consistent with the scanner's rated frame rate
[inf].

## Flash pages (bulk EP 0x02 → 0x82) [der]

`0x1400`-byte transfers. Request: header with opcode `0x57`, length 2 and a
BE16 page index at offset 8. The reply carries 4096 bytes at offset 9. Writes
use `0x58` with data at offset 10. `readCustomInfo(offset, size)` reads pages
`offset >> 12` through `(offset + size - 1) >> 12`. At connect EXStar reads
6568 bytes from offset 0, which is probably the factory calibration. That
blob's layout is **[?]**.

## Image stream (EP 0x83) [der]

The vendor comm layer stores each transfer behind a 4-byte big-endian length
prefix (`captureImageTransferCallBack`). The offsets below are relative to the
raw USB transfer:

| Off | Meaning |
|---|---|
| 0x02 | bits 3..5 camera field (1, 2 or 4 = camera 0, 1, 2); bits 0..2 sub-index |
| 0x08 | `1` on a frame's last transfer |
| 0x0c..0x0f | BE32, read from the last transfer (frame id [inf]) |
| 0x10..0x17 | BE64, read from the first transfer (timestamp [inf]) |
| 0x20.. | payload |

Frames are 1280×1024, 8-bit [obs, log]. A capture group has two IR images and
one colour image. The colour camera delivers raw Bayer data, converted with
OpenCV `COLOR_BayerRG2BGR` (code 0x30) [der].

## Calibration

### Flash blob [der]
EXStar reads 6568 bytes of custom info at connect. `Sn3DEinStarDeviceData::
setCaliData` base64-decodes the string it receives and copies it straight
into a `tagDevData` struct, so the blob is a raw C struct. It also carries
activation data (`serialNum`, `dataVersion`, `activeState`, `activeTime`).

**Calibration block** (`libSn3DCalibDevFile`: `setDevData` copies the 0x19A8-byte
struct to handler +0x1420, and `readQuickCCFData` copies 0xF21 bytes from
handler +0x17BB, so the block starts at **blob offset 0x39B**) [der]:

| Block offset | Content |
|---|---|
| 0x000 | int32 type, must be 4 |
| 0x108 / 0x10C | int32 size / LeftCCF bytes |
| 0x610 / 0x614 | int32 size / RightCCF bytes |
| 0xB18 / 0xB1C | int32 size / TexCCF bytes |
| 0xF1C | `"FQFQ"` |

The slots are written to the CCF files verbatim (`writeCCFFile`). The
calibration date (`yyyy-MM-dd hh:mm`) is stored at handler +0xE69. The other
regions (white balance, colour correction, activation) are **[?]**.
`DeviceDataManager::readDevCalDataToDisk` writes the files under
`EXStar.app/Contents/Resources/res/Einscan-E10/200x150/`:

| File | Size | Content |
|---|---|---|
| `TexCCF.txt` | 264 B | colour camera, 33 LE float64, plain |
| `LeftCCF.txt`, `RightCCF.txt` | 520 B | IR cameras, 65 LE float64, each value offset by a small fixed integer (obfuscation) |
| `wbdata*.txt`, `ParaCCM_*.txt`, `ParaDark_*.txt`, `activeBox.dat` | small | white balance, colour correction, dark level, working volume |

CCF layout (from `TexCCF`): `fx fy 0 0 cx cy 0 0 d5 0 k1 k2 p1 p2 k3 0 0 0 0 0 Tx Ty Tz R(9)`.

### EXStar project camera model (`*.data_cm`, 692 bytes) [der]
Little-endian float64. Camera roles were established by matching the CCF files:

| Index | Content |
|---|---|
| 0..3 | header [?] (looks like rectified cx, cy, f, f) |
| 4..25 | **camera 0 = colour** (TexCCF): fx fy cx cy, k1 k2 p1 p2 k3 d5, R (row-major), T (mm) |
| 26..37 | pair block: R [?], T = camera 1 → camera 0 (mm; matches the poses exactly) |
| 38..59 | **camera 1 = IR "left"** (LeftCCF, de-obfuscated) |
| 60..81 | **camera 2 = IR "right"** (RightCCF, de-obfuscated) |
| tail | floats [?], then int32 `640 512 320 256` |

Per-camera R, T map a shared calibration frame into the camera:
`x_cam = R x_world + T`. In the calibration examined, the IR pair has a 160 mm
baseline and converges by ≈ 22.5°. The colour camera is 37 mm from IR right
and 123 mm from IR left.

### Stream cameras vs calibration cameras [hw]
Stream 0 is IR right, stream 1 is IR left and stream 2 is colour
(`calibration_index_for_stream`). Streams 0 and 2 arrive rotated by 180°
relative to the calibration (`stream_orientation`). Debayer the colour stream
(`COLOR_BayerRG2BGR`) before rotating it. The RTAB-Map driver rotates its output
back to the scanner's natural orientation and folds the turn into the camera
model.

## Open questions

1. Whether other Einstar units use the same PID and whether any firmware enables the XOR mask (off here).
2. Stream header bytes 0..1 and the timestamp interpretation.
3. Units of exposure (µs assumed), gain, trigger period and temperature (reads 0 while idle).
4. Which strobe route drives which emitter, and the meaning of `LDRGBLuminance`.
5. Whether the LeftCCF/RightCCF offset key is the same on every scanner.
