#include <algorithm>
#include "check.hpp"
#include "einstar/exposure.hpp"
#include "einstar/protocol.hpp"

using namespace einstar;

// Byte layout taken from CEinScan10::setExposureTime (einscan10.2.0.0.dylib
// 0xb3d8..0xb41c): header word 0x00231000 at offset 1, len 5 at offset 7,
// camera mask at 8, exposure big-endian at 9..12.
TEST(set_exposure_packet_matches_vendor_layout) {
  const std::uint8_t payload[] = {camera_mask(1), 0x00, 0x00, 0x13, 0x88};  // 5000 us
  auto p = build_command(7, Opcode::SetExposure, payload);
  CHECK(p.has_value());
  const Packet expected = {7, 0x00, 0x10, 0x23, 0, 0, 0, 5, 0x02, 0x00, 0x00, 0x13, 0x88, 0, 0, 0};
  CHECK(*p == expected);
}

TEST(set_gain_packet_matches_vendor_layout) {
  const std::uint8_t payload[] = {camera_mask(2), 0x01, 0x90};  // 400
  auto p = build_command(0, Opcode::SetGain, payload);
  const Packet expected = {0, 0x00, 0x10, 0x27, 0, 0, 0, 3, 0x04, 0x01, 0x90, 0, 0, 0, 0, 0};
  CHECK(*p == expected);
}

TEST(payload_too_large_is_rejected) {
  const std::uint8_t payload[9] = {};
  CHECK(!build_command(0, Opcode::SetGain, payload).has_value());
}

TEST(camera_mask_matches_vendor_csinc) {
  // Vendor: selector = (cam == 2) ? 4 : cam + 1
  CHECK(camera_mask(0) == 1);
  CHECK(camera_mask(1) == 2);
  CHECK(camera_mask(2) == 4);
}

TEST(reply_validation_follows_interruptIO) {
  auto req = *build_command(9, Opcode::GetGain, std::array<std::uint8_t, 1>{1});
  Packet reply = req;
  reply[kOffReplyValue] = 0x00;
  reply[kOffReplyValue + 1] = 0x78;
  CHECK(!validate_reply(req, reply).has_value());
  CHECK(read_be16(reply, kOffReplyValue) == 120);

  Packet bad = reply;
  bad[kOffId] = 10;
  CHECK(validate_reply(req, bad) == ReplyError::IdMismatch);
  bad = reply;
  bad[kOffOpcode] = 0x27;
  CHECK(validate_reply(req, bad) == ReplyError::OpcodeMismatch);
  bad = reply;
  bad[kOffStatus] = 1;
  CHECK(validate_reply(req, bad) == ReplyError::DeviceStatus);
  CHECK(validate_reply(req, ByteView(reply.data(), 3)) == ReplyError::TooShort);
}

TEST(packet_ids_wrap_at_254) {
  PacketIdCounter c;
  std::uint8_t last = 0;
  for (int i = 0; i < 254; ++i) last = c.next();
  CHECK(last == 253);
  CHECK(c.next() == 0);
}

TEST(mask_roundtrip_recovers_key_from_byte1) {
  auto p = *build_command(3, Opcode::SetTriggerPeriod, std::array<std::uint8_t, 4>{0, 1, 0x5f, 0x90});
  const Packet original = p;
  for (std::uint8_t n = 0; n < 16; ++n) {
    Packet q = original;
    mask_encode(q, n);
    if (n) CHECK(q != original);
    CHECK((q[1] >> 4) == n);
    mask_decode(q);
    CHECK(q == original);
  }
}

TEST(handshake_reply_carries_shining3_marker) {
  std::array<std::uint8_t, 100> reply{};
  std::copy(kHandshakeMarker, kHandshakeMarker + 8, reply.begin() + 9);
  CHECK(is_handshake_reply(reply));
  reply[12] = 'x';
  CHECK(!is_handshake_reply(reply));
  CHECK(!is_handshake_reply(ByteView(reply.data(), 16)));
}

// Replies captured from an Einstar (firmware EinScan10_01_SC130_FX3_V2.14_FPGA_V3.7_EN).
TEST(observed_replies_decode) {
  const std::vector<std::uint8_t> width = {0x00, 0x02, 0x10, 0x16, 0, 0, 0, 0, 0x02, 0x05, 0x00};
  const std::vector<std::uint8_t> exposure = {0x00, 0x02, 0x10, 0x22, 0, 0, 0, 0, 0x04, 0x00, 0x00, 0x08, 0x88};
  const std::vector<std::uint8_t> sensors = {0x00, 0x02, 0x10, 0x51, 0, 0, 0, 0, 0x01, 0x03};
  const std::vector<std::uint8_t> serial = {0x00, 0x02, 0x00, 0x04, 0, 0, 0, 0, 0x0c, 0x12, 0x34, 0x56,
                                            0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x44, 0x00, 0x00, 0x00};
  const std::vector<std::uint8_t> name = {0x00, 0x02, 0x00, 0x01, 0, 0, 0, 0, 0x0c, 'E', 'i', 'n', 'S',
                                          'c', 'a', 'n', '1', '0', '_', '0', '1'};
  CHECK(reply_uint(width) == 1280u);
  CHECK(reply_uint(exposure) == 2184u);
  CHECK(reply_uint(sensors) == 3u);
  CHECK(reply_serial(serial) == std::string("123456789ABCDEF0"));
  CHECK(reply_string(name) == std::string("EinScan10_01"));
  CHECK(!reply_uint(serial).has_value());  // 12-byte value is not an integer
  // A length byte larger than the reply is clamped to the bytes present.
  std::vector<std::uint8_t> truncated = width;
  truncated[kOffReplyLength] = 9;
  CHECK(reply_value(truncated).size() == 2);
}

TEST(temperature_is_signed_be16_shifted_by_7) {
  Packet r{};
  write_be16(r, kOffReplyValue, 0x1900);  // 25.0 degC in LM75 encoding -> 50 half-degrees
  CHECK(decode_temperature_raw(r) == 50);
  write_be16(r, kOffReplyValue, 0xFF80);  // -0.5 degC -> -1
  CHECK(decode_temperature_raw(r) == -1);
}

TEST(only_queries_are_read_only) {
  CHECK(is_read_only(Opcode::GetExposure));
  CHECK(is_read_only(Opcode::PageRead));
  CHECK(!is_read_only(Opcode::SetExposure));
  CHECK(!is_read_only(Opcode::PageWrite));
  CHECK(!is_read_only(Opcode::SetIndication));
}

TEST(system_commands_use_group_zero) {
  auto p = build_command(5, command(SystemOpcode::FirmwareVersion));
  const Packet expected = {5, 0, 0x00, 0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  CHECK(*p == expected);
  Packet reply = *p;
  reply[kOffGroup] = 0x10;
  CHECK(validate_reply(*p, reply) == ReplyError::GroupMismatch);
}

TEST(wire_sizes_match_vendor_calls) {
  CHECK(wire_size(command(Opcode::SetExposure)) == 16);
  CHECK(wire_size(command(Opcode::SetTriggerSwitch)) == 10);
  CHECK(wire_size(command(Opcode::HeartBeat)) == 10);
  CHECK(wire_size(command(Opcode::SetTriggerPeriod)) == 50);
  CHECK(wire_size(command(Opcode::SetStrobeBrightness)) == 50);
  CHECK(wire_size(command(Opcode::SetLaserBrightness)) == 50);
  CHECK(wire_size(command(SystemOpcode::ProductName)) == 100);
  CHECK(wire_size(command(SystemOpcode::FirmwareVersion)) == 64);
}

TEST(reboot_and_flash_writes_are_forbidden) {
  CHECK(is_forbidden(command(SystemOpcode::Reboot)));
  CHECK(is_forbidden(command(Opcode::PageWrite)));
  CHECK(!is_forbidden(command(Opcode::SetTriggerSwitch)));
  CHECK(!is_read_only(command(SystemOpcode::Reboot)));
  CHECK(is_read_only(command(SystemOpcode::Serial)));
}

TEST(trigger_counts_pack_rgb_high_mono_low) {
  // CEinScan10::triggerSwitchIO: read splits byte >> 4 / byte & 0xF.
  CHECK(pack_trigger_counts(1, 1) == 0x11);
  CHECK(pack_trigger_counts(2, 0) == 0x02);
  CHECK(pack_trigger_counts(0, 3) == 0x30);
  CHECK(unpack_trigger_counts(0x21) == std::make_pair(1u, 2u));
}

TEST(reply_string_stops_at_nul) {
  std::array<std::uint8_t, 64> reply{};
  const char fw[] = "EinScan10_01_SC130_FX3_V2.14_FPGA_V3.7_EN";
  std::copy(std::begin(fw), std::end(fw), reply.begin() + kOffReplyValue);
  reply[kOffReplyLength] = sizeof fw;
  CHECK(reply_string(reply) == std::string(fw));
  CHECK(!reply_string(ByteView(reply.data(), 9)).has_value());
}

TEST(auto_exposure_shortens_on_saturation_and_lengthens_when_dark) {
  const AutoExposure ae;
  std::vector<std::uint8_t> bright(10000, 255), dark(10000, 20), fine(10000, 220), dim(10000, 150);
  CHECK(ae.update(bright, 4000) == 2800u);
  CHECK(ae.update(dark, 1000) == 1250u);
  CHECK(!ae.update(fine, 1000).has_value());
  CHECK(ae.update(dim, 1000) == 1211u);        // half of the log correction: sqrt(220 / 150)
  CHECK(!ae.update(bright, 200).has_value());  // already at the minimum
  CHECK(ae.update(dark, 5900) == 6000u);       // clamped to the maximum
}

TEST(auto_exposure_pair_follows_the_brighter_view) {
  const AutoExposure ae;
  std::vector<std::uint8_t> in_band(10000, 220), dim(10000, 150), clipped(10000, 255);
  CHECK(!ae.update(in_band, dim, 2000).has_value());  // the dimmer view must not lengthen
  CHECK(!ae.update(dim, in_band, 2000).has_value());
  CHECK(ae.update(clipped, dim, 2000) == 1400u);       // either view clipping shortens
  CHECK(ae.update(dim, dim, 1000) == 1211u);           // both dim: lengthen
}

// Closed loop on a scanned object against a dark background, with a new
// exposure only visible after a delay (as on hardware): the exposure must
// settle and then stop changing, since every change blinks the light.
TEST(auto_exposure_settles_with_delayed_response) {
  const AutoExposure ae;
  std::vector<double> reflectance(20000);
  for (std::size_t i = 0; i < reflectance.size(); ++i)
    reflectance[i] = i % 10 == 0 ? 0.08 + 0.04 * ((i * 7919) % 13) / 13.0  // bright speckled object, 10%
                                 : 0.004;                                    // dark background
  // The right view sees the speckle ~35% dimmer, as measured on hardware.
  for (double right_gain : {1.0, 0.74})
    for (int delay : {1, 2}) {
      std::vector<std::uint32_t> applied(80, 4000);
      std::uint32_t current = 4000;
      int late_changes = 0;
      for (int frame = 0; frame < 80; ++frame) {
        const std::uint32_t seen = applied[std::max(0, frame - delay)];
        std::vector<std::uint8_t> left(reflectance.size()), right(reflectance.size());
        for (std::size_t i = 0; i < left.size(); ++i) {
          left[i] = static_cast<std::uint8_t>(std::min(255.0, reflectance[i] * seen));
          right[i] = static_cast<std::uint8_t>(std::min(255.0, reflectance[i] * seen * right_gain));
        }
        if (auto next = ae.update(left, right, current)) {
          current = *next;
          if (frame >= 40) ++late_changes;
        }
        applied[frame] = current;
      }
      CHECK(late_changes == 0);
      CHECK(current > 1300 && current < 2300);  // brighter view's speckle near the target of 220
    }
}
