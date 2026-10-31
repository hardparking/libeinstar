// Simulated Einstar for Device tests: speaks the reverse-engineered protocol
// (handshake, command replies, flash pages, stream transfers) over the
// Transport interface and records what it receives.
#pragma once

#include <chrono>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "einstar/protocol.hpp"
#include "einstar/transport.hpp"

namespace einstar::test {

struct ReceivedCommand {
  std::uint8_t group = 0;
  std::uint8_t opcode = 0;
  std::vector<std::uint8_t> payload;
  std::size_t wire_size = 0;
};

class FakeScanner final : public Transport {
 public:
  // Configuration (set before handing the scanner to Device::open).
  std::array<std::uint8_t, 8> serial{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
  std::string firmware = "EinScan10_01_SC130_FX3_V2.14_FPGA_V3.7_EN";
  std::map<std::uint8_t, std::uint32_t> exposure{{1, 5000}, {2, 5000}, {4, 1000}};  // by camera mask
  std::map<std::uint8_t, std::uint16_t> gain{{1, 120}, {2, 120}, {4, 200}};
  std::uint8_t trigger_byte = 0;
  std::uint32_t trigger_period = 90000;
  bool require_mask = false;
  int corrupt_replies = 0;  // corrupt the packet id of the next N replies
  std::vector<std::vector<std::uint8_t>> pages;  // 4 KiB each

  std::vector<ReceivedCommand> received() {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_;
  }
  int clear_halts() {
    std::lock_guard<std::mutex> lock(mutex_);
    return clear_halts_;
  }
  void queue_stream_transfer(std::vector<std::uint8_t> transfer) {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_.push_back(std::move(transfer));
  }

  int interrupt_transfer(std::uint8_t ep, std::uint8_t* data, std::size_t length, std::size_t& transferred,
                         unsigned) override {
    std::lock_guard<std::mutex> lock(mutex_);
    transferred = 0;
    if (ep == kEpCommandOut) {
      std::vector<std::uint8_t> packet(data, data + length);
      transferred = length;
      if (length == kHandshakeSize && packet[0] == kHandshakeByte) {
        std::vector<std::uint8_t> reply(kHandshakeSize, 0);
        std::copy(kHandshakeMarker, kHandshakeMarker + 8, reply.begin() + 9);
        interrupt_replies_.push_back(std::move(reply));
        return kTransferOk;
      }
      if (require_mask) mask_decode(packet);
      handle_command(packet);
      return kTransferOk;
    }
    if (ep == kEpCommandIn) {
      if (interrupt_replies_.empty()) return kTransferTimeout;
      auto reply = std::move(interrupt_replies_.front());
      interrupt_replies_.pop_front();
      transferred = std::min(length, reply.size());
      std::memcpy(data, reply.data(), transferred);
      return kTransferOk;
    }
    return -1;
  }

  int bulk_transfer(std::uint8_t ep, std::uint8_t* data, std::size_t length, std::size_t& transferred,
                    unsigned) override {
    transferred = 0;
    if (ep == kEpStreamIn) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (stream_.empty()) {
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return kTransferTimeout;
      }
      auto t = std::move(stream_.front());
      stream_.pop_front();
      transferred = std::min(length, t.size());
      std::memcpy(data, t.data(), transferred);
      return kTransferOk;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (ep == kEpBulkOut) {
      // Requests may arrive in several transfers; act once a full one is in.
      transferred = length;
      bulk_in_progress_.insert(bulk_in_progress_.end(), data, data + length);
      if (bulk_in_progress_.size() < kPageTransferSize) return kTransferOk;
      const ByteView req(bulk_in_progress_.data(), bulk_in_progress_.size());
      const std::uint16_t index = read_be16(req, kOffPayload);
      record(req, bulk_in_progress_.size());
      std::vector<std::uint8_t> reply(kPageTransferSize, 0);
      std::copy(bulk_in_progress_.begin(), bulk_in_progress_.begin() + kOffPayload, reply.begin());
      reply[1] = kReplyMarker;
      reply[kOffStatus] = index < pages.size() ? 0 : 1;
      if (index < pages.size()) std::copy(pages[index].begin(), pages[index].end(), reply.begin() + kOffReplyValue);
      bulk_reply_bytes_.insert(bulk_reply_bytes_.end(), reply.begin(), reply.end());
      bulk_in_progress_.clear();
      return kTransferOk;
    }
    if (ep == kEpBulkIn) {
      if (bulk_reply_bytes_.empty()) return kTransferTimeout;
      transferred = std::min(length, bulk_reply_bytes_.size());
      std::copy_n(bulk_reply_bytes_.begin(), transferred, data);
      bulk_reply_bytes_.erase(bulk_reply_bytes_.begin(), bulk_reply_bytes_.begin() + static_cast<long>(transferred));
      return kTransferOk;
    }
    return -1;
  }

  void clear_halt(std::uint8_t) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++clear_halts_;
  }
  std::string describe(int status) const override { return status == kTransferTimeout ? "timeout" : "error"; }

 private:
  void record(ByteView packet, std::size_t wire) {
    ReceivedCommand c;
    c.group = packet[kOffGroup];
    c.opcode = packet[kOffOpcode];
    c.payload.assign(packet.begin() + kOffPayload, packet.begin() + kOffPayload + packet[kOffLength]);
    c.wire_size = wire;
    received_.push_back(std::move(c));
  }

  void handle_command(const std::vector<std::uint8_t>& p) {
    record(p, p.size());
    // Reply layout observed on hardware: byte 1 = 0x02, byte 8 = value length.
    std::vector<std::uint8_t> reply(p.begin(), p.begin() + kOffPayload);
    reply[1] = kReplyMarker;
    reply[kOffStatus] = 0;
    reply[kOffLength] = 0;
    const std::uint8_t sel = p[kOffPayload];
    auto value = [&](std::uint32_t v, int bytes) {
      reply.push_back(static_cast<std::uint8_t>(bytes));
      for (int i = bytes - 1; i >= 0; --i) reply.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    };
    auto bytes = [&](const std::uint8_t* b, std::size_t n) {
      reply.push_back(static_cast<std::uint8_t>(n));
      reply.insert(reply.end(), b, b + n);
    };
    const ByteView req(p.data(), p.size());
    if (p[kOffGroup] == static_cast<std::uint8_t>(Group::System)) {
      if (p[kOffOpcode] == static_cast<std::uint8_t>(SystemOpcode::FirmwareVersion))
        bytes(reinterpret_cast<const std::uint8_t*>(firmware.data()), firmware.size());
      else if (p[kOffOpcode] == static_cast<std::uint8_t>(SystemOpcode::Serial)) {
        std::uint8_t twelve[12] = {};
        std::copy(serial.begin(), serial.end(), twelve);
        bytes(twelve, sizeof twelve);
      } else {
        value(0, 0);
      }
    } else {
      switch (static_cast<Opcode>(p[kOffOpcode])) {
        case Opcode::GetExposure: value(exposure[sel], 4); break;
        case Opcode::SetExposure: exposure[sel] = read_be32(req, kOffPayload + 1); value(0, 0); break;
        case Opcode::GetGain: value(gain[sel], 2); break;
        case Opcode::SetGain: gain[sel] = read_be16(req, kOffPayload + 1); value(0, 0); break;
        case Opcode::SensorCount: value(3, 1); break;
        case Opcode::Temperature: value(0x1900, 2); break;
        case Opcode::GetTriggerSwitch: value(trigger_byte, 1); break;
        case Opcode::SetTriggerSwitch: trigger_byte = p[kOffPayload]; value(0, 0); break;
        case Opcode::GetTriggerPeriod: value(trigger_period, 4); break;
        case Opcode::SetTriggerPeriod: trigger_period = read_be32(req, kOffPayload); value(0, 0); break;
        default: value(0, 0); break;
      }
    }
    if (corrupt_replies > 0) {
      --corrupt_replies;
      reply[kOffId] ^= 0xFF;
    }
    if (require_mask) mask_encode(reply, static_cast<std::uint8_t>(received_.size() & 0x0F));
    interrupt_replies_.push_back(std::move(reply));
  }

  std::mutex mutex_;
  std::deque<std::vector<std::uint8_t>> interrupt_replies_, stream_;
  std::vector<std::uint8_t> bulk_in_progress_, bulk_reply_bytes_;
  std::vector<ReceivedCommand> received_;
  int clear_halts_ = 0;
};

}  // namespace einstar::test
