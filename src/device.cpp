#include "einstar/device.hpp"

#include <libusb.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>

namespace einstar {
namespace {

Error usb_error(const char* what, int rc) {
  return {ErrorCode::UsbError, std::string(what) + ": " + libusb_error_name(rc)};
}

// libusb-backed transport; owns the context and the claimed interface.
class LibusbTransport final : public Transport {
 public:
  LibusbTransport(libusb_context* ctx, libusb_device_handle* handle) : ctx_(ctx), handle_(handle) {}
  ~LibusbTransport() override {
    if (handle_) {
      libusb_release_interface(handle_, kInterface);
      libusb_close(handle_);
    }
    if (ctx_) libusb_exit(ctx_);
  }
  int interrupt_transfer(std::uint8_t ep, std::uint8_t* data, std::size_t length, std::size_t& transferred,
                         unsigned timeout_ms) override {
    int got = 0;
    const int rc = libusb_interrupt_transfer(handle_, ep, data, static_cast<int>(length), &got, timeout_ms);
    transferred = static_cast<std::size_t>(got);
    return rc;
  }
  int bulk_transfer(std::uint8_t ep, std::uint8_t* data, std::size_t length, std::size_t& transferred,
                    unsigned timeout_ms) override {
    int got = 0;
    const int rc = libusb_bulk_transfer(handle_, ep, data, static_cast<int>(length), &got, timeout_ms);
    transferred = static_cast<std::size_t>(got);
    return rc;
  }
  void clear_halt(std::uint8_t ep) override { libusb_clear_halt(handle_, ep); }
  std::string describe(int status) const override { return libusb_error_name(status); }

 private:
  libusb_context* ctx_;
  libusb_device_handle* handle_;
};

std::string product_string(libusb_device* dev, const libusb_device_descriptor& d) {
  if (d.iProduct == 0) return {};
  libusb_device_handle* h = nullptr;
  if (libusb_open(dev, &h) != 0) return {};
  unsigned char buf[256] = {};
  const int n = libusb_get_string_descriptor_ascii(h, d.iProduct, buf, sizeof buf);
  libusb_close(h);
  return n > 0 ? std::string(reinterpret_cast<char*>(buf), static_cast<std::size_t>(n)) : std::string{};
}

// The Einstar's USB product string is "SHINING-CAM" (observed); EXStar's
// "EINSCAN10" check applies to the name returned by the device protocol.
bool looks_like_einstar(const std::string& s) {
  return s == "SHINING-CAM" || s.rfind("EINSCAN10", 0) == 0 || s.rfind("EinScan10", 0) == 0;
}

}  // namespace

const char* to_string(ErrorCode c) {
  switch (c) {
    case ErrorCode::UsbError: return "usb error";
    case ErrorCode::NotFound: return "device not found";
    case ErrorCode::NotOpen: return "device not open";
    case ErrorCode::Timeout: return "timeout";
    case ErrorCode::BadReply: return "bad reply";
    case ErrorCode::WritesDisabled: return "writes disabled";
    case ErrorCode::InvalidArgument: return "invalid argument";
  }
  return "unknown";
}

Result<std::vector<DeviceInfo>> Device::enumerate() {
  libusb_context* ctx = nullptr;
  if (int rc = libusb_init(&ctx); rc != 0) return usb_error("libusb_init", rc);
  libusb_device** list = nullptr;
  const ssize_t n = libusb_get_device_list(ctx, &list);
  if (n < 0) {
    libusb_exit(ctx);
    return usb_error("libusb_get_device_list", static_cast<int>(n));
  }
  std::vector<DeviceInfo> out;
  for (ssize_t i = 0; i < n; ++i) {
    libusb_device_descriptor d{};
    if (libusb_get_device_descriptor(list[i], &d) != 0 || d.idVendor != kVendorId) continue;
    DeviceInfo info;
    info.vendor_id = d.idVendor;
    info.product_id = d.idProduct;
    info.bus = libusb_get_bus_number(list[i]);
    info.port = libusb_get_port_number(list[i]);
    info.product = product_string(list[i], d);
    info.looks_like_einstar = looks_like_einstar(info.product);
    out.push_back(std::move(info));
  }
  libusb_free_device_list(list, 1);
  libusb_exit(ctx);
  return out;
}

Result<std::unique_ptr<Device>> Device::open(std::optional<DeviceInfo> want, OpenOptions options) {
  libusb_context* ctx = nullptr;
  if (int rc = libusb_init(&ctx); rc != 0) return usb_error("libusb_init", rc);

  libusb_device** list = nullptr;
  const ssize_t n = libusb_get_device_list(ctx, &list);
  if (n < 0) {
    libusb_exit(ctx);
    return usb_error("libusb_get_device_list", static_cast<int>(n));
  }
  libusb_device* chosen = nullptr;
  DeviceInfo info;
  for (ssize_t i = 0; i < n; ++i) {
    libusb_device_descriptor d{};
    if (libusb_get_device_descriptor(list[i], &d) != 0 || d.idVendor != kVendorId) continue;
    const auto bus = libusb_get_bus_number(list[i]);
    const auto port = libusb_get_port_number(list[i]);
    if (want) {
      if (want->bus == bus && want->port == port) {
        chosen = list[i];
        info = *want;
      }
      continue;
    }
    // Prefer a device whose product string looks like an Einstar, else take
    // the first device with SHINING 3D's vendor id.
    const std::string product = product_string(list[i], d);
    const bool einstar = looks_like_einstar(product);
    if (einstar || !chosen) {
      chosen = list[i];
      info = {d.idVendor, d.idProduct, bus, port, product, einstar};
    }
    if (einstar) break;
  }
  libusb_device_handle* handle = nullptr;
  Status status = Ok{};
  if (!chosen) {
    status = Error{ErrorCode::NotFound, "no SHINING 3D device (VID 0x3267) found"};
  } else if (int rc = libusb_open(chosen, &handle); rc != 0) {
    status = usb_error("libusb_open", rc);
  }
  libusb_free_device_list(list, 1);
  if (!status) {
    libusb_exit(ctx);
    return status.error();
  }

  libusb_set_auto_detach_kernel_driver(handle, 1);
  if (int rc = libusb_claim_interface(handle, kInterface); rc != 0) {
    libusb_close(handle);
    libusb_exit(ctx);
    return usb_error("libusb_claim_interface", rc);
  }
  return open(std::make_unique<LibusbTransport>(ctx, handle), options, info);
}

Result<std::unique_ptr<Device>> Device::open(std::unique_ptr<Transport> transport, OpenOptions options,
                                             DeviceInfo info) {
  if (!transport) return Error{ErrorCode::InvalidArgument, "no transport"};
  std::unique_ptr<Device> dev(new Device());
  dev->transport_ = std::move(transport);
  dev->options_ = options;
  dev->info_ = std::move(info);
  if (Status s = dev->handshake(); !s) return s.error();
  auto serial = dev->transact(command(SystemOpcode::Serial));
  if (!serial) return serial.error();
  auto text = reply_serial(serial.value());
  if (!text) return Error{ErrorCode::BadReply, "serial reply too short"};
  dev->serial_ = *text;
  return dev;
}

Device::~Device() { stop_stream(); }

Error Device::transfer_error(const char* what, int rc) const {
  return {rc == kTransferTimeout ? ErrorCode::Timeout : ErrorCode::UsbError,
          std::string(what) + ": " + transport_->describe(rc)};
}

Status Device::handshake() {
  // Mirrors CCommunicateManager::attach: drain stale replies, clear halts,
  // then exchange the 0xFF handshake whose reply carries the serial.
  std::uint8_t scratch[1024];
  std::size_t got = 0;
  for (int i = 0; i < 5; ++i)
    if (transport_->interrupt_transfer(kEpCommandIn, scratch, sizeof scratch, got, 10) != kTransferOk) break;
  for (int i = 0; i < 30; ++i)
    if (transport_->bulk_transfer(kEpBulkIn, scratch, sizeof scratch, got, 10) != kTransferOk) break;
  // attach() resets the command-in and bulk-in pipes.
  transport_->clear_halt(kEpCommandIn);
  transport_->clear_halt(kEpBulkIn);

  std::uint8_t hs[kHandshakeSize] = {};
  hs[0] = kHandshakeByte;
  if (int rc = transport_->interrupt_transfer(kEpCommandOut, hs, sizeof hs, got, options_.timeout_ms); rc != kTransferOk)
    return transfer_error("handshake send", rc);
  std::uint8_t reply[kHandshakeSize] = {};
  if (int rc = transport_->interrupt_transfer(kEpCommandIn, reply, sizeof reply, got, options_.timeout_ms);
      rc != kTransferOk)
    return transfer_error("handshake reply", rc);
  if (!is_handshake_reply(ByteView(reply, got)))
    return Error{ErrorCode::BadReply, "handshake reply lacks the \"Shining3\" marker"};
  return Ok{};
}

Status Device::require_writes(Command c) const {
  if (is_forbidden(c))
    return Error{ErrorCode::WritesDisabled, "flash writes and reboot are not supported by libeinstar"};
  if (!is_read_only(c) && !options_.allow_writes)
    return Error{ErrorCode::WritesDisabled, "command changes device state; open with allow_writes"};
  return Ok{};
}

Result<std::vector<std::uint8_t>> Device::transact(Command c, ByteView payload) {
  if (!transport_) return Error{ErrorCode::NotOpen, "device not open"};
  if (Status s = require_writes(c); !s) return s.error();
  std::lock_guard<std::recursive_mutex> lock(command_mutex_);
  static thread_local std::mt19937 rng{std::random_device{}()};
  const std::size_t size = wire_size(c);

  Error last{ErrorCode::Timeout, "no attempts made"};
  for (int attempt = 0; attempt < options_.retries; ++attempt) {
    auto req = build_command(ids_.next(), c, payload);
    if (!req) return Error{ErrorCode::InvalidArgument, "payload too large"};
    std::vector<std::uint8_t> wire(size, 0);
    std::copy_n(req->begin(), std::min(size, req->size()), wire.begin());
    if (options_.mask_commands) mask_encode(wire, static_cast<std::uint8_t>(rng() & 0x0F));
    std::size_t got = 0;
    int rc = transport_->interrupt_transfer(kEpCommandOut, wire.data(), wire.size(), got, options_.timeout_ms);
    if (rc == kTransferOk) {
      std::vector<std::uint8_t> reply(1024, 0);
      rc = transport_->interrupt_transfer(kEpCommandIn, reply.data(), reply.size(), got, options_.timeout_ms);
      if (rc == kTransferOk) {
        reply.resize(got);
        if (options_.mask_commands) mask_decode(reply);
        if (auto err = validate_reply(*req, reply); !err) return reply;
        else last = {ErrorCode::BadReply, to_string(*err)};
      } else {
        last = transfer_error("command reply", rc);
      }
    } else {
      last = transfer_error("command send", rc);
    }
    transport_->clear_halt(kEpCommandOut);
    transport_->clear_halt(kEpCommandIn);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return last;
}

Status Device::set(Opcode op, ByteView payload) {
  auto r = transact(op, payload);
  if (!r) return r.error();
  return Ok{};
}

Result<std::string> Device::product_name() {
  auto r = transact(command(SystemOpcode::ProductName));
  if (!r) return r.error();
  return reply_string(r.value()).value_or("");
}

Result<std::string> Device::firmware_version() {
  auto r = transact(command(SystemOpcode::FirmwareVersion));
  if (!r) return r.error();
  return reply_string(r.value()).value_or("");
}

Result<std::uint32_t> Device::query_uint(Opcode op, ByteView payload) {
  auto r = transact(op, payload);
  if (!r) return r.error();
  auto v = reply_uint(r.value());
  if (!v) return Error{ErrorCode::BadReply, "reply value missing or longer than 4 bytes"};
  return *v;
}

Result<std::uint32_t> Device::query_camera(Opcode op, unsigned camera) {
  const std::uint8_t p[] = {camera_mask(camera)};
  return query_uint(op, p);
}

Result<unsigned> Device::sensor_count() {
  auto v = query_uint(Opcode::SensorCount);
  if (!v) return v.error();
  return static_cast<unsigned>(v.value());
}

Result<std::uint32_t> Device::exposure_us(unsigned camera) { return query_camera(Opcode::GetExposure, camera); }

Result<std::uint16_t> Device::gain(unsigned camera) {
  auto v = query_camera(Opcode::GetGain, camera);
  if (!v) return v.error();
  return static_cast<std::uint16_t>(v.value());
}

Result<std::uint32_t> Device::max_width(unsigned camera) { return query_camera(Opcode::MaxWidth, camera); }

Result<std::uint32_t> Device::max_height(unsigned camera) { return query_camera(Opcode::MaxHeight, camera); }

Result<std::uint32_t> Device::trigger_period() { return query_uint(Opcode::GetTriggerPeriod); }

Result<int> Device::temperature_raw() {
  auto r = transact(Opcode::Temperature);
  if (!r) return r.error();
  if (reply_value(r.value()).size() != 2) return Error{ErrorCode::BadReply, "temperature is not a 16-bit value"};
  return decode_temperature_raw(r.value());
}

Result<std::pair<unsigned, unsigned>> Device::trigger_counts() {
  auto v = query_uint(Opcode::GetTriggerSwitch);
  if (!v) return v.error();
  return unpack_trigger_counts(static_cast<std::uint8_t>(v.value()));
}

// Bulk request/reply as CEinScan10::bulkIO -> sendDataAndWaitReply does it:
// 1 KiB transfers (1 ms apart, 2 s timeout) out, 1 KiB transfers (600 ms
// timeout, up to 3 tries 80 ms apart) back, whole exchange retried 5 times
// with the bulk-in pipe reset in between.
Result<std::vector<std::uint8_t>> Device::bulk_exchange(const std::vector<std::uint8_t>& request,
                                                        std::size_t reply_size) {
  constexpr std::size_t kChunk = 0x400;
  Error last{ErrorCode::Timeout, "no attempts made"};
  for (int attempt = 0; attempt < options_.retries; ++attempt) {
    std::vector<std::uint8_t> out = request;
    bool sent = true;
    for (std::size_t off = 0; off + kChunk <= out.size(); off += kChunk) {
      std::size_t got = 0;
      if (int rc = transport_->bulk_transfer(kEpBulkOut, out.data() + off, kChunk, got, 2000); rc != kTransferOk) {
        last = transfer_error("bulk request", rc);
        sent = false;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (sent) {
      for (int read_try = 0; read_try < 3; ++read_try) {
        std::vector<std::uint8_t> reply(reply_size, 0);
        std::size_t total = 0;
        int rc = kTransferOk;
        for (std::size_t off = 0; off + kChunk <= reply.size(); off += kChunk) {
          std::size_t got = 0;
          rc = transport_->bulk_transfer(kEpBulkIn, reply.data() + off, kChunk, got, 600);
          if (rc != kTransferOk) {
            if (rc == kTransferTimeout) transport_->clear_halt(kEpBulkIn);
            break;
          }
          total += got;
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (rc == kTransferOk) {
          reply.resize(total);
          return reply;
        }
        last = transfer_error("bulk reply", rc);
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
      }
    }
    transport_->clear_halt(kEpBulkIn);
  }
  return last;
}

Result<std::vector<std::uint8_t>> Device::read_page(std::uint16_t index) {
  if (!transport_) return Error{ErrorCode::NotOpen, "device not open"};
  if (Status s = require_writes(command(Opcode::PageRead)); !s) return s.error();
  std::lock_guard<std::recursive_mutex> lock(command_mutex_);
  for (int attempt = 0; attempt < options_.retries; ++attempt) {
    std::uint8_t page[2];
    write_be16(page, 0, index);
    auto req = build_command(ids_.next(), Opcode::PageRead, page);
    std::vector<std::uint8_t> request(kPageTransferSize, 0);
    std::copy(req->begin(), req->end(), request.begin());
    auto reply = bulk_exchange(request, kPageTransferSize);
    if (!reply) return reply.error();
    const auto& r = reply.value();
    if (auto err = validate_reply(*req, r)) {
      if (err == ReplyError::DeviceStatus) return Error{ErrorCode::BadReply, to_string(*err)};
      transport_->clear_halt(kEpBulkIn);
      continue;  // stale or mismatched reply: retry like bulkIO
    }
    if (r.size() < kOffReplyValue + kPageSize) return Error{ErrorCode::BadReply, "page reply shorter than 4 KiB"};
    return std::vector<std::uint8_t>(r.begin() + kOffReplyValue, r.begin() + kOffReplyValue + kPageSize);
  }
  return Error{ErrorCode::BadReply, "page reply did not echo the request"};
}

Result<std::vector<std::uint8_t>> Device::read_custom_info(std::size_t size) {
  std::vector<std::uint8_t> out;
  out.reserve(size);
  for (std::uint16_t page = 0; out.size() < size; ++page) {
    auto p = read_page(page);
    if (!p) return p.error();
    const std::size_t take = std::min(size - out.size(), p.value().size());
    out.insert(out.end(), p.value().begin(), p.value().begin() + take);
  }
  return out;
}

Result<CameraModel> Device::read_calibration() {
  auto info = read_custom_info(kDeviceInfoSize);
  if (!info) return info.error();
  auto parsed = parse_device_calibration(info.value());
  if (!parsed.model) return Error{ErrorCode::BadReply, to_string(*parsed.error)};
  return *parsed.model;
}

Status Device::set_exposure_us(unsigned camera, std::uint32_t us) {
  std::uint8_t p[5] = {camera_mask(camera)};
  write_be32(p, 1, us);
  return set(Opcode::SetExposure, p);
}

Status Device::set_gain(unsigned camera, std::uint16_t g) {
  std::uint8_t p[3] = {camera_mask(camera)};
  write_be16(p, 1, g);
  return set(Opcode::SetGain, p);
}

Status Device::set_trigger_period(std::uint32_t period) {
  std::uint8_t p[4];
  write_be32(p, 0, period);
  return set(Opcode::SetTriggerPeriod, p);
}

Status Device::set_strobe_brightness(std::uint8_t route, std::uint16_t luminance) {
  std::uint8_t p[3] = {route};
  write_be16(p, 1, luminance);
  return set(Opcode::SetStrobeBrightness, p);
}

Status Device::set_laser_brightness(std::uint8_t percent) {
  const std::uint8_t p[1] = {std::min<std::uint8_t>(percent, 100)};
  return set(Opcode::SetLaserBrightness, p);
}

Status Device::set_trigger_counts(unsigned mono, unsigned rgb) {
  if (Status s = set_trigger_period(200000); !s) return s;
  const std::uint8_t p[1] = {pack_trigger_counts(mono, rgb)};
  if (Status s = set(Opcode::SetTriggerSwitch, p); !s) return s;
  if (rgb == 0) std::this_thread::sleep_for(std::chrono::seconds(1));
  return Ok{};
}

Status Device::start_scanning(std::uint32_t trigger_period) {
  if (Status s = set_trigger_counts(1, 1); !s) return s;
  return set_trigger_period(trigger_period);
}

Status Device::stop_scanning() { return set_trigger_counts(0, 0); }

Status Device::apply(const ScanSettings& s) {
  for (unsigned cam : {0u, 1u}) {
    if (Status st = set_exposure_us(cam, s.ir_exposure_us); !st) return st;
    if (Status st = set_gain(cam, s.ir_gain); !st) return st;
  }
  if (Status st = set_exposure_us(2, s.color_exposure_us); !st) return st;
  if (Status st = set_gain(2, s.color_gain); !st) return st;
  if (Status st = set_strobe_brightness(0, s.strobe_route0); !st) return st;
  if (Status st = set_strobe_brightness(1, s.strobe_route1); !st) return st;
  return set_laser_brightness(s.laser_percent);
}

Status Device::lights_off() {
  if (Status st = set_strobe_brightness(0, 0); !st) return st;
  if (Status st = set_strobe_brightness(1, 0); !st) return st;
  return set_laser_brightness(0);
}

Status Device::start_stream(FrameCallback cb, std::size_t frame_size) {
  if (!transport_) return Error{ErrorCode::NotOpen, "device not open"};
  if (streaming_.exchange(true)) return Error{ErrorCode::InvalidArgument, "already streaming"};
  stream_thread_ = std::thread([this, cb = std::move(cb), frame_size] { stream_loop(cb, frame_size); });
  return Ok{};
}

void Device::stop_stream() {
  streaming_ = false;
  if (stream_thread_.joinable()) stream_thread_.join();
}

void Device::stream_loop(FrameCallback cb, std::size_t frame_size) {
  FrameAssembler assembler(frame_size);
  std::vector<std::uint8_t> buf(kStreamPackageSize);
  while (streaming_) {
    std::size_t got = 0;
    const int rc = transport_->bulk_transfer(kEpStreamIn, buf.data(), buf.size(), got, 200);
    if (rc == kTransferTimeout) continue;
    if (rc != kTransferOk) {
      transport_->clear_halt(kEpStreamIn);
      assembler.reset();
      continue;
    }
    if (assembler.push(ByteView(buf.data(), got)) == StreamEvent::FrameComplete)
      cb(assembler.take());
  }
}

}  // namespace einstar
