// libusb-backed access to an Einstar scanner.
//
// Safety: a Device is read-only unless OpenOptions::allow_writes is set, and
// writes are limited to volatile settings (exposure, gain, trigger, lighting).
// Flash writes (opcode 0x58), reboot and firmware update are deliberately not
// implemented; they can overwrite factory calibration or interrupt the device.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "einstar/calibration.hpp"
#include "einstar/protocol.hpp"
#include "einstar/result.hpp"
#include "einstar/stream.hpp"
#include "einstar/transport.hpp"

namespace einstar {

struct DeviceInfo {
  std::uint16_t vendor_id = 0;
  std::uint16_t product_id = 0;
  std::uint8_t bus = 0;
  std::uint8_t port = 0;
  std::string product;       // USB product string, may be empty if unreadable
  bool looks_like_einstar = false;  // product string "SHINING-CAM" (observed) or "EINSCAN10*"
};

// Lighting and exposure for scanning. Defaults are values EXStar used while
// scanning (observed in its logs) and light the IR speckle on hardware.
struct ScanSettings {
  std::uint16_t strobe_route0 = 1000;
  std::uint16_t strobe_route1 = 300;
  std::uint8_t laser_percent = 100;
  std::uint32_t ir_exposure_us = 4000;
  std::uint16_t ir_gain = 105;
  std::uint32_t color_exposure_us = 5000;
  std::uint16_t color_gain = 180;
};

struct OpenOptions {
  bool allow_writes = false;
  bool mask_commands = false;  // XOR obfuscation; whether firmware expects it is unverified
  unsigned timeout_ms = 1000;
  int retries = 5;
};

class Device {
 public:
  static Result<std::vector<DeviceInfo>> enumerate();
  // Opens the device matching `info` (bus/port), else an Einstar-looking
  // device, else the first device with vendor id 0x3267.
  static Result<std::unique_ptr<Device>> open(std::optional<DeviceInfo> info = {},
                                              OpenOptions options = {});
  // Attaches over an existing transport (another USB backend, or a simulated
  // scanner in tests) and performs the handshake.
  static Result<std::unique_ptr<Device>> open(std::unique_ptr<Transport> transport, OpenOptions options = {},
                                              DeviceInfo info = {});
  ~Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  const std::string& serial() const { return serial_; }
  const DeviceInfo& info() const { return info_; }

  // Raw request/reply with vendor retry semantics and transfer lengths.
  // State-changing commands require allow_writes; forbidden ones always fail.
  Result<std::vector<std::uint8_t>> transact(Command c, ByteView payload = {});
  Result<std::vector<std::uint8_t>> transact(Opcode op, ByteView payload = {}) {
    return transact(command(op), payload);
  }

  Result<std::string> product_name();
  Result<std::string> firmware_version();

  Result<unsigned> sensor_count();
  Result<std::uint32_t> exposure_us(unsigned camera);
  Result<std::uint16_t> gain(unsigned camera);
  Result<std::uint32_t> max_width(unsigned camera);
  Result<std::uint32_t> max_height(unsigned camera);
  Result<std::uint32_t> trigger_period();
  Result<int> temperature_raw();
  // Frames captured per trigger: {mono (IR), rgb}. {0, 0} means not scanning.
  Result<std::pair<unsigned, unsigned>> trigger_counts();
  // Reads one 4 KiB flash page over the bulk endpoints.
  Result<std::vector<std::uint8_t>> read_page(std::uint16_t index);
  Result<std::vector<std::uint8_t>> read_custom_info(std::size_t size);
  // Factory calibration stored on the scanner (see parse_device_calibration).
  Result<CameraModel> read_calibration();

  Status set_exposure_us(unsigned camera, std::uint32_t us);
  Status set_gain(unsigned camera, std::uint16_t gain);
  Status set_trigger_period(std::uint32_t period);
  Status set_strobe_brightness(std::uint8_t route, std::uint16_t luminance);
  // 0..100, the vendor clamps to 100 (opcode 0x68).
  Status set_laser_brightness(std::uint8_t percent);
  // Mirrors the vendor TriggerMode sequence: trigger period 200000, then the
  // packed counts (opcode 0x41), then a 1 s settle when RGB is disabled.
  Status set_trigger_counts(unsigned mono, unsigned rgb);
  // EXStar's scan start: one IR and one colour frame per trigger, period
  // 90000 (likely microseconds, ~11 Hz). Requires allow_writes.
  Status start_scanning(std::uint32_t trigger_period = 90000);
  Status stop_scanning();
  // Exposure, gain and lighting for all cameras; lights_off() sets the
  // strobe routes and laser back to 0.
  Status apply(const ScanSettings& settings);
  Status lights_off();

  using FrameCallback = std::function<void(Frame&&)>;
  Status start_stream(FrameCallback cb, std::size_t frame_size = 1280 * 1024);
  void stop_stream();

 private:
  Device() = default;
  Status require_writes(Command c) const;
  Status handshake();
  Error transfer_error(const char* what, int rc) const;
  Result<std::uint32_t> query_uint(Opcode op, ByteView payload = {});
  Result<std::vector<std::uint8_t>> bulk_exchange(const std::vector<std::uint8_t>& request, std::size_t reply_size);
  Result<std::uint32_t> query_camera(Opcode op, unsigned camera);
  Status set(Opcode op, ByteView payload);
  void stream_loop(FrameCallback cb, std::size_t frame_size);

  std::unique_ptr<Transport> transport_;
  DeviceInfo info_;
  OpenOptions options_;
  PacketIdCounter ids_;
  std::recursive_mutex command_mutex_;  // commands may come from several threads
  std::string serial_;
  std::atomic<bool> streaming_{false};
  std::thread stream_thread_;
};

}  // namespace einstar
