// Enumerates Einstar scanners and runs read-only queries.
//
//   einstar-probe                      list devices with VID 0x3267
//   einstar-probe --info               open the first Einstar and query it
//   einstar-probe --dump-flash FILE    also save the 6568-byte custom info blob
//   einstar-probe --frames N --out DIR listen for N frames (up to --seconds S,
//                                      default 10) and save them as PGM
//   einstar-probe --scan --frames N    also start the scan trigger first and
//                                      stop it afterwards (volatile settings only)
//   einstar-probe --raw GROUP OPCODE [PAYLOAD_HEX]
//                                      send one read-only command, hex-dump the reply
//   with --scan: EXStar's scan lighting/exposure is applied (ScanSettings) and
//                lighting is switched off afterwards; override with
//                --strobe ROUTE:LUM  --laser PCT  --exposure CAM:US  --gain CAM:G
//                (repeatable), or --no-lights to leave lighting untouched;
//                IR auto-exposure runs unless --fixed-exposure
//   --mask                             XOR-mask commands (unverified requirement)
//
// Without --scan nothing changes device state. --scan sends exactly EXStar's
// trigger sequence; flash writes and reboot are never available.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#include "einstar/device.hpp"
#include "einstar/exposure.hpp"

using namespace einstar;

namespace {

constexpr std::size_t kCustomInfoSize = 6568;  // EXStar reads this at connect

template <typename T>
void report(const char* what, const Result<T>& r) {
  if (r) std::printf("  %-22s %llu\n", what, static_cast<unsigned long long>(r.value()));
  else std::printf("  %-22s error: %s\n", what, r.error().message.c_str());
}

bool write_pgm(const std::string& path, const Frame& f, int w, int h) {
  std::ofstream o(path, std::ios::binary);
  o << "P5\n" << w << " " << h << "\n255\n";
  o.write(reinterpret_cast<const char*>(f.data.data()), static_cast<std::streamsize>(f.data.size()));
  o.close();
  return static_cast<bool>(o);
}

}  // namespace

int main(int argc, char** argv) {
  bool info = false;
  bool mask = false;
  bool scan = false;
  std::string dump_path, out_dir = ".";
  int frames = 0;
  int seconds = 10;
  int raw_group = -1, raw_opcode = -1;
  std::vector<std::pair<unsigned, unsigned>> strobes, exposures, gains;
  int laser = -1;
  bool lights = true;
  bool auto_exposure = true;
  auto pair_arg = [](const char* a) {
    unsigned x = 0, y = 0;
    std::sscanf(a, "%u:%u", &x, &y);
    return std::make_pair(x, y);
  };
  std::vector<std::uint8_t> raw_payload;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--info")) info = true;
    else if (!std::strcmp(argv[i], "--mask")) mask = true;
    else if (!std::strcmp(argv[i], "--scan")) scan = true;
    else if (!std::strcmp(argv[i], "--dump-flash") && i + 1 < argc) { dump_path = argv[++i]; info = true; }
    else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) { frames = std::atoi(argv[++i]); info = true; }
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--strobe") && i + 1 < argc) strobes.push_back(pair_arg(argv[++i]));
    else if (!std::strcmp(argv[i], "--exposure") && i + 1 < argc) exposures.push_back(pair_arg(argv[++i]));
    else if (!std::strcmp(argv[i], "--gain") && i + 1 < argc) gains.push_back(pair_arg(argv[++i]));
    else if (!std::strcmp(argv[i], "--laser") && i + 1 < argc) laser = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-lights")) lights = false;
    else if (!std::strcmp(argv[i], "--fixed-exposure")) auto_exposure = false;
    else if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--raw") && i + 2 < argc) {
      raw_group = static_cast<int>(std::strtol(argv[++i], nullptr, 0));
      raw_opcode = static_cast<int>(std::strtol(argv[++i], nullptr, 0));
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string hex = argv[++i];
        for (std::size_t k = 0; k + 1 < hex.size(); k += 2)
          raw_payload.push_back(static_cast<std::uint8_t>(std::stoi(hex.substr(k, 2), nullptr, 16)));
      }
      info = true;
    }
    else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }

  auto list = Device::enumerate();
  if (!list) {
    std::fprintf(stderr, "enumerate failed: %s\n", list.error().message.c_str());
    return 1;
  }
  std::printf("%zu device(s) with VID 0x%04x\n", list.value().size(), kVendorId);
  for (const auto& d : list.value())
    std::printf("  bus %u port %u  pid 0x%04x  product \"%s\"%s\n", d.bus, d.port, d.product_id,
                d.product.c_str(), d.looks_like_einstar ? "  [einstar]" : "");
  if (!info) return 0;

  OpenOptions opts;
  opts.mask_commands = mask;
  opts.allow_writes = scan;
  auto dev = Device::open(std::nullopt, opts);
  if (!dev) {
    std::fprintf(stderr, "open failed: %s\n", dev.error().message.c_str());
    return 1;
  }
  Device& d = *dev.value();
  if (raw_opcode >= 0) {
    const Command c{static_cast<Group>(raw_group), static_cast<std::uint8_t>(raw_opcode)};
    auto r = d.transact(c, raw_payload);  // non-read-only commands are refused without --scan
    if (!r) {
      std::fprintf(stderr, "raw command failed: %s\n", r.error().message.c_str());
      return 1;
    }
    std::printf("reply (%zu bytes):", r.value().size());
    for (std::size_t k = 0; k < r.value().size(); ++k) std::printf("%s%02x", k % 16 ? " " : "\n  ", r.value()[k]);
    std::printf("\n");
    return 0;
  }
  std::printf("serial %s\n", d.serial().c_str());
  if (auto name = d.product_name()) std::printf("  %-22s %s\n", "product", name.value().c_str());
  if (auto fw = d.firmware_version()) std::printf("  %-22s %s\n", "firmware", fw.value().c_str());
  if (auto tc = d.trigger_counts())
    std::printf("  %-22s mono %u, rgb %u\n", "trigger counts", tc.value().first, tc.value().second);
  report("sensor count", d.sensor_count());
  report("trigger period", d.trigger_period());
  auto t = d.temperature_raw();
  if (t) std::printf("  %-22s %d (raw, likely 0.5 degC units)\n", "temperature", t.value());
  for (unsigned c = 0; c < 3; ++c) {
    std::printf(" camera %u\n", c);
    report("max width", d.max_width(c));
    report("max height", d.max_height(c));
    report("exposure us", d.exposure_us(c));
    report("gain", d.gain(c));
  }

  if (!dump_path.empty()) {
    auto blob = d.read_custom_info(kCustomInfoSize);
    if (!blob) {
      std::fprintf(stderr, "flash read failed: %s\n", blob.error().message.c_str());
      return 1;
    }
    std::ofstream(dump_path, std::ios::binary)
        .write(reinterpret_cast<const char*>(blob.value().data()), static_cast<std::streamsize>(blob.value().size()));
    std::printf("wrote %zu bytes to %s\n", blob.value().size(), dump_path.c_str());
    auto calib = parse_device_calibration(blob.value());
    if (calib.model) {
      const Pose lr = calib.model->relative(kIrLeftCamera, kIrRightCamera);
      std::printf("calibration decoded: colour fx %.1f, IR baseline %.1f mm\n",
                  calib.model->cameras[kColorCamera].intrinsics.fx,
                  std::sqrt(lr.T[0] * lr.T[0] + lr.T[1] * lr.T[1] + lr.T[2] * lr.T[2]));
    } else {
      std::printf("calibration not decoded: %s\n", to_string(*calib.error));
    }
  }

  if (frames > 0) {
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
      std::fprintf(stderr, "cannot create %s: %s\n", out_dir.c_str(), ec.message().c_str());
      return 1;
    }
    std::mutex m;
    std::atomic<int> saved{0};
    std::atomic<bool> write_failed{false};
    // Device frame ids are 8-bit (observed); number files by a wrap-aware sequence.
    std::uint32_t epoch = 0, last_id = 0;
    bool first_frame = true;
    // The callback proposes IR exposures; the main loop sends them.
    std::atomic<std::uint32_t> ir_exposure{ScanSettings{}.ir_exposure_us}, wanted_exposure{0};
    const AutoExposure ae;
    // Latest IR image per camera; both views of a frame set judge the shared exposure together.
    std::vector<std::uint8_t> last_ir[2];
    std::uint32_t last_ir_id[2] = {0, 0};
    auto s = d.start_stream([&](Frame&& f) {
      const unsigned cam = f.info.camera_index().value_or(2);
      if (scan && auto_exposure && cam < 2) {
        last_ir[cam].assign(f.data.begin(), f.data.end());
        last_ir_id[cam] = f.info.frame_id;
        if (last_ir_id[0] == last_ir_id[1] && !last_ir[1 - cam].empty())
          if (auto next = ae.update(last_ir[0], last_ir[1], ir_exposure.load())) wanted_exposure = *next;
      }
      std::lock_guard<std::mutex> lock(m);
      if (saved >= frames) return;
      char name[512];
      const std::uint32_t id = f.info.frame_id & 0xFF;
      std::uint32_t frame_epoch = epoch;
      if (!first_frame && id + 128 < last_id) frame_epoch = ++epoch;            // wrapped
      else if (!first_frame && id > last_id + 128 && epoch > 0) frame_epoch = epoch - 1;  // late frame from before the wrap
      first_frame = false;
      if (frame_epoch == epoch) last_id = id;
      std::snprintf(name, sizeof name, "%s/frame_%06u_cam%u.pgm", out_dir.c_str(), frame_epoch * 256 + id,
                    f.info.camera_index().value_or(9));
      if (!write_pgm(name, f, 1280, 1024)) {
        write_failed = true;
        return;
      }
      ++saved;
    });
    if (!s) {
      std::fprintf(stderr, "stream failed: %s\n", s.error().message.c_str());
      return 1;
    }
    auto check = [](const char* what, const Status& st) {
      if (!st) std::fprintf(stderr, "%s failed: %s\n", what, st.error().message.c_str());
    };
    if (scan && lights) check("scan settings", d.apply(ScanSettings{}));
    if (scan) {
      for (auto [cam, us] : exposures) check("exposure", d.set_exposure_us(cam, us));
      for (auto [cam, g] : gains) check("gain", d.set_gain(cam, static_cast<std::uint16_t>(g)));
      for (auto [route, lum] : strobes)
        check("strobe", d.set_strobe_brightness(static_cast<std::uint8_t>(route), static_cast<std::uint16_t>(lum)));
      if (laser >= 0) check("laser", d.set_laser_brightness(static_cast<std::uint8_t>(laser)));
    }
    if (scan) {
      if (auto st = d.start_scanning(); !st) {
        std::fprintf(stderr, "start scanning failed: %s\n", st.error().message.c_str());
        d.stop_stream();
        return 1;
      }
    }
    for (int i = 0; i < seconds * 10 && saved < frames && !write_failed; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (const std::uint32_t want = wanted_exposure.exchange(0)) {
        if (d.set_exposure_us(0, want) && d.set_exposure_us(1, want)) ir_exposure = want;
      }
    }
    if (scan) {
      if (auto st = d.stop_scanning(); !st)
        std::fprintf(stderr, "stop scanning failed: %s\n", st.error().message.c_str());
      if (lights || !strobes.empty() || laser >= 0) check("lights off", d.lights_off());
    }
    d.stop_stream();
    if (write_failed) {
      std::fprintf(stderr, "writing frames to %s failed after %d frame(s)\n", out_dir.c_str(), saved.load());
      return 1;
    }
    std::printf("saved %d frame(s)%s\n", saved.load(), scan ? "" : "; use --scan to start the trigger");
  }
  return 0;
}
