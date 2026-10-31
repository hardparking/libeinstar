#include "einstar/protocol.hpp"

#include <algorithm>

namespace einstar {

bool is_read_only(Command c) {
  if (c.group == Group::System)
    return c.code != static_cast<std::uint8_t>(SystemOpcode::Reboot);
  switch (static_cast<Opcode>(c.code)) {
    case Opcode::VendorId:
    case Opcode::ProductId:
    case Opcode::MaxWidth:
    case Opcode::MaxHeight:
    case Opcode::MaxExposure:
    case Opcode::MinExposure:
    case Opcode::GetExposure:
    case Opcode::MaxGain:
    case Opcode::MinGain:
    case Opcode::GetGain:
    case Opcode::PixelBits:
    case Opcode::GetTriggerSwitch:
    case Opcode::GetTriggerPeriod:
    case Opcode::Temperature:
    case Opcode::SensorCount:
    case Opcode::PageRead:
    case Opcode::ColorMode:
    case Opcode::GetLaserBrightness:
    case Opcode::GetStrobeBrightness:
    case Opcode::HeartBeat:
      return true;
    default:
      return false;
  }
}

bool is_forbidden(Command c) {
  if (c.group == Group::System) return c.code == static_cast<std::uint8_t>(SystemOpcode::Reboot);
  return c.code == static_cast<std::uint8_t>(Opcode::PageWrite);
}

std::size_t wire_size(Command c) {
  if (c.group == Group::System) {
    switch (static_cast<SystemOpcode>(c.code)) {
      case SystemOpcode::ProductName: return 100;
      case SystemOpcode::Serial:
      case SystemOpcode::FirmwareVersion: return 64;
      case SystemOpcode::Reboot: return 10;
    }
    return kCommandSize;
  }
  switch (static_cast<Opcode>(c.code)) {
    case Opcode::GetTriggerSwitch:
    case Opcode::SetTriggerSwitch:
    case Opcode::HeartBeat:
      return 10;
    case Opcode::GetTriggerPeriod:
    case Opcode::SetTriggerPeriod:
    case Opcode::GetLaserBrightness:
    case Opcode::SetLaserBrightness:
    case Opcode::GetStrobeBrightness:
    case Opcode::SetStrobeBrightness:
      return 50;
    default:
      return kCommandSize;
  }
}

std::optional<Packet> build_command(std::uint8_t packet_id, Command c, ByteView payload) {
  if (payload.size() > kMaxPayload) return std::nullopt;
  Packet p{};
  p[kOffId] = packet_id;
  p[kOffGroup] = static_cast<std::uint8_t>(c.group);
  p[kOffOpcode] = c.code;
  p[kOffLength] = static_cast<std::uint8_t>(payload.size());
  std::copy(payload.begin(), payload.end(), p.begin() + kOffPayload);
  return p;
}

const char* to_string(ReplyError e) {
  switch (e) {
    case ReplyError::TooShort: return "reply too short";
    case ReplyError::IdMismatch: return "packet id mismatch";
    case ReplyError::GroupMismatch: return "command group mismatch";
    case ReplyError::OpcodeMismatch: return "opcode mismatch";
    case ReplyError::DeviceStatus: return "device reported non-zero status";
  }
  return "unknown";
}

std::optional<ReplyError> validate_reply(const Packet& request,
                                         ByteView reply) {
  if (reply.size() <= kOffStatus) return ReplyError::TooShort;
  if (reply[kOffId] != request[kOffId]) return ReplyError::IdMismatch;
  if (reply[kOffGroup] != request[kOffGroup]) return ReplyError::GroupMismatch;
  if (reply[kOffOpcode] != request[kOffOpcode]) return ReplyError::OpcodeMismatch;
  if (reply[kOffStatus] != 0) return ReplyError::DeviceStatus;
  return std::nullopt;
}

std::uint8_t camera_mask(unsigned camera_index) {
  return static_cast<std::uint8_t>(1u << camera_index);
}

std::uint8_t pack_trigger_counts(unsigned mono, unsigned rgb) {
  return static_cast<std::uint8_t>(((rgb & 0x0F) << 4) | (mono & 0x0F));
}

std::pair<unsigned, unsigned> unpack_trigger_counts(std::uint8_t byte) {
  return {byte & 0x0Fu, byte >> 4};
}

std::optional<std::string> reply_string(ByteView reply) {
  if (reply.size() <= kOffReplyValue) return std::nullopt;
  std::string s;
  for (auto b : reply_value(reply)) {
    if (b == 0) break;
    s.push_back(static_cast<char>(b));
  }
  return s;
}

std::uint8_t PacketIdCounter::next() {
  const std::uint8_t id = value_;
  value_ = static_cast<std::uint8_t>((value_ + 1) % kPacketIdModulus);
  return id;
}

std::uint16_t read_be16(ByteView b, std::size_t off) {
  return static_cast<std::uint16_t>((b[off] << 8) | b[off + 1]);
}

std::uint32_t read_be32(ByteView b, std::size_t off) {
  return (std::uint32_t{b[off]} << 24) | (std::uint32_t{b[off + 1]} << 16) |
         (std::uint32_t{b[off + 2]} << 8) | std::uint32_t{b[off + 3]};
}

std::uint64_t read_be64(ByteView b, std::size_t off) {
  return (std::uint64_t{read_be32(b, off)} << 32) | read_be32(b, off + 4);
}

void write_be16(MutableByteView b, std::size_t off, std::uint16_t v) {
  b[off] = static_cast<std::uint8_t>(v >> 8);
  b[off + 1] = static_cast<std::uint8_t>(v);
}

void write_be32(MutableByteView b, std::size_t off, std::uint32_t v) {
  b[off] = static_cast<std::uint8_t>(v >> 24);
  b[off + 1] = static_cast<std::uint8_t>(v >> 16);
  b[off + 2] = static_cast<std::uint8_t>(v >> 8);
  b[off + 3] = static_cast<std::uint8_t>(v);
}

ByteView reply_value(ByteView reply) {
  if (reply.size() <= kOffReplyValue) return {};
  const std::size_t n = std::min<std::size_t>(reply[kOffReplyLength], reply.size() - kOffReplyValue);
  return ByteView(reply.data() + kOffReplyValue, n);
}

std::optional<std::uint32_t> reply_uint(ByteView reply) {
  const ByteView v = reply_value(reply);
  if (v.empty() || v.size() > 4) return std::nullopt;
  std::uint32_t out = 0;
  for (auto b : v) out = (out << 8) | b;
  return out;
}

bool is_handshake_reply(ByteView reply) {
  return reply.size() >= kOffReplyValue + 8 &&
         std::equal(kHandshakeMarker, kHandshakeMarker + 8, reply.begin() + kOffReplyValue);
}

std::optional<std::string> reply_serial(ByteView reply) {
  const ByteView v = reply_value(reply);
  if (v.size() < 8) return std::nullopt;
  static constexpr char kHex[] = "0123456789ABCDEF";
  std::string s;
  for (std::size_t i = 0; i < 8; ++i) {
    s.push_back(kHex[v[i] >> 4]);
    s.push_back(kHex[v[i] & 0x0F]);
  }
  return s;
}

int decode_temperature_raw(ByteView reply) {
  const auto raw = static_cast<std::int16_t>(read_be16(reply, kOffReplyValue));
  return raw >> 7;
}

void mask_encode(MutableByteView data, std::uint8_t nibble) {
  const std::uint8_t key = static_cast<std::uint8_t>((nibble & 0x0F) | (nibble << 4));
  for (auto& b : data) b ^= key;
}

void mask_decode(MutableByteView data) {
  if (data.size() < 2) return;
  const std::uint8_t n = data[1] >> 4;
  const std::uint8_t key = static_cast<std::uint8_t>(n | (n << 4));
  for (auto& b : data) b ^= key;
}

}  // namespace einstar
