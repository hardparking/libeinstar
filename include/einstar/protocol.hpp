// Wire-level constants and the command packet codec.
//
// Everything here is pure (no I/O) so it can be unit-tested without hardware.
// See docs/PROTOCOL.md for provenance of each constant.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "einstar/span.hpp"

namespace einstar {

inline constexpr std::uint16_t kVendorId = 0x3267;
inline constexpr int kInterface = 0;

inline constexpr std::uint8_t kEpCommandOut = 0x01;  // interrupt
inline constexpr std::uint8_t kEpCommandIn = 0x81;   // interrupt
inline constexpr std::uint8_t kEpBulkOut = 0x02;
inline constexpr std::uint8_t kEpBulkIn = 0x82;
inline constexpr std::uint8_t kEpStreamIn = 0x83;

inline constexpr std::size_t kCommandSize = 16;
inline constexpr std::size_t kHandshakeSize = 100;
inline constexpr std::uint8_t kHandshakeByte = 0xFF;
inline constexpr std::size_t kPageTransferSize = 0x1400;
inline constexpr std::size_t kPageSize = 0x1000;
inline constexpr std::size_t kStreamPackageSize = 0xA400;
inline constexpr std::uint8_t kPacketIdModulus = 0xFE;

inline constexpr std::size_t kOffId = 0;
inline constexpr std::size_t kOffGroup = 2;
inline constexpr std::size_t kOffOpcode = 3;
inline constexpr std::size_t kOffStatus = 4;
inline constexpr std::size_t kOffLength = 7;
inline constexpr std::size_t kOffPayload = 8;
// Replies (observed on hardware): byte 1 = 0x02, byte 8 = value length N,
// value at 9..9+N (big-endian).
inline constexpr std::size_t kOffReplyLength = 8;
inline constexpr std::size_t kOffReplyValue = 9;
inline constexpr std::uint8_t kReplyMarker = 0x02;
inline constexpr std::size_t kMaxPayload = kCommandSize - kOffPayload;

// Byte 2 selects a command group.
enum class Group : std::uint8_t {
  System = 0x00,  // device identity and control
  Camera = 0x10,  // sensors, trigger, lighting, flash pages
};

// System-group opcodes (byte 2 = 0x00).
enum class SystemOpcode : std::uint8_t {
  ProductName = 0x01,      // NUL-terminated string at reply offset 9
  Serial = 0x04,           // 8 bytes at reply offset 9
  FirmwareVersion = 0x05,  // NUL-terminated string at reply offset 9
  Reboot = 0x08,           // never sent by this library
};

// Camera-group opcodes (byte 2 = 0x10).
enum class Opcode : std::uint8_t {
  VendorId = 0x00,
  ProductId = 0x01,
  MaxWidth = 0x16,
  MaxHeight = 0x17,
  MaxExposure = 0x20,
  MinExposure = 0x21,
  GetExposure = 0x22,
  SetExposure = 0x23,
  MaxGain = 0x24,
  MinGain = 0x25,
  GetGain = 0x26,
  SetGain = 0x27,
  PixelBits = 0x2e,
  GetTriggerSwitch = 0x40,
  SetTriggerSwitch = 0x41,
  GetTriggerPeriod = 0x48,
  SetTriggerPeriod = 0x49,
  Temperature = 0x50,
  SensorCount = 0x51,
  PageRead = 0x57,
  PageWrite = 0x58,  // overwrites device flash; never sent by this library
  ColorMode = 0x5d,
  SetIndication = 0x62,
  GetLaserBrightness = 0x67,
  SetLaserBrightness = 0x68,
  GetStrobeBrightness = 0x6f,
  SetStrobeBrightness = 0x70,
  HeartBeat = 0x7b,
};

struct Command {
  Group group;
  std::uint8_t code;
};
constexpr Command command(Opcode op) { return {Group::Camera, static_cast<std::uint8_t>(op)}; }
constexpr Command command(SystemOpcode op) { return {Group::System, static_cast<std::uint8_t>(op)}; }

// True for commands that only query state.
bool is_read_only(Command c);
inline bool is_read_only(Opcode op) { return is_read_only(command(op)); }
// Commands this library refuses to send at all (flash writes, reboot).
bool is_forbidden(Command c);

// Interrupt transfer length the vendor plugin uses for each command (10, 16,
// 50, 64 or 100 bytes); the request is zero-padded to it.
std::size_t wire_size(Command c);

using Packet = std::array<std::uint8_t, kCommandSize>;

// Builds a request; returns nullopt when the payload does not fit.
std::optional<Packet> build_command(std::uint8_t packet_id, Command c, ByteView payload = {});
inline std::optional<Packet> build_command(std::uint8_t packet_id, Opcode op, ByteView payload = {}) {
  return build_command(packet_id, command(op), payload);
}

enum class ReplyError {
  TooShort,
  IdMismatch,
  GroupMismatch,
  OpcodeMismatch,
  DeviceStatus,
};

const char* to_string(ReplyError e);

// Validates a reply exactly as the vendor plugin does: id, group and opcode
// must echo the request and the status byte must be zero.
std::optional<ReplyError> validate_reply(const Packet& request,
                                         ByteView reply);

// Camera selector bitmask used by exposure/gain setters (cam0=1, cam1=2, cam2=4).
std::uint8_t camera_mask(unsigned camera_index);

// Trigger counts share one byte (opcodes 0x40/0x41): RGB frames per trigger in
// the high nibble, mono (IR) frames in the low nibble.
std::uint8_t pack_trigger_counts(unsigned mono, unsigned rgb);
std::pair<unsigned, unsigned> unpack_trigger_counts(std::uint8_t byte);  // {mono, rgb}

// String reply (product name, firmware version): N value bytes, stopping
// early at a NUL.
std::optional<std::string> reply_string(ByteView reply);

// Rolling packet id generator (0..253).
class PacketIdCounter {
 public:
  std::uint8_t next();

 private:
  std::uint8_t value_ = 0;
};

std::uint16_t read_be16(ByteView b, std::size_t off);
std::uint32_t read_be32(ByteView b, std::size_t off);
std::uint64_t read_be64(ByteView b, std::size_t off);
void write_be16(MutableByteView b, std::size_t off, std::uint16_t v);
void write_be32(MutableByteView b, std::size_t off, std::uint32_t v);

// Value bytes of a reply (bounded by the length byte and the reply size).
ByteView reply_value(ByteView reply);
// Big-endian unsigned value of 1..4 bytes; nullopt for other lengths.
std::optional<std::uint32_t> reply_uint(ByteView reply);

// The 0xFF handshake reply carries ASCII "Shining3" at bytes 9..16
// (observed); the device serial comes from SystemOpcode::Serial.
inline constexpr char kHandshakeMarker[] = "Shining3";
bool is_handshake_reply(ByteView reply);
// Serial reply: the first 8 value bytes, rendered %02X (matches EXStar).
std::optional<std::string> reply_serial(ByteView reply);

// LM75-style temperature decoding observed in readTemperature: big-endian
// 16-bit value at reply offset 9, arithmetic shift right by 7. The vendor
// stores the shifted integer; units (likely 0.5 degC) are unverified.
int decode_temperature_raw(ByteView reply);

// Optional XOR obfuscation. Key byte is n|(n<<4) for a nibble n; the decoder
// recovers n from the high nibble of byte 1, which is zero in plaintext.
void mask_encode(MutableByteView data, std::uint8_t nibble);
void mask_decode(MutableByteView data);

}  // namespace einstar
