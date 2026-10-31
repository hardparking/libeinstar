#include "check.hpp"
#include "einstar/protocol.hpp"
#include "einstar/stream.hpp"

using namespace einstar;

namespace {

std::vector<std::uint8_t> transfer(std::size_t chunk, bool last, std::uint8_t camera_field,
                                   std::uint32_t frame_id, std::uint64_t ts, std::uint8_t fill) {
  std::vector<std::uint8_t> t(kStreamHeaderSize + chunk, 0);
  t[kOffStreamCamera] = static_cast<std::uint8_t>((camera_field << 3) | 1);
  t[kOffStreamLastFlag] = last ? 1 : 0;
  write_be32(t, kOffStreamFrameId, frame_id);
  write_be32(t, kOffStreamTimestamp, static_cast<std::uint32_t>(ts >> 32));
  write_be32(t, kOffStreamTimestamp + 4, static_cast<std::uint32_t>(ts));
  for (std::size_t i = kStreamPayloadOffset; i < t.size(); ++i) t[i] = fill;
  return t;
}

}  // namespace

TEST(frame_completes_on_last_transfer_with_exact_size) {
  FrameAssembler a(64);
  CHECK(a.push(transfer(32, false, 2, 0, 0x1122334455667788ull, 0xAA)) == StreamEvent::Accepted);
  CHECK(a.push(transfer(32, true, 2, 42, 0, 0xBB)) == StreamEvent::FrameComplete);
  Frame f = a.take();
  CHECK(f.data.size() == 64);
  CHECK(f.info.frame_id == 42);
  CHECK(f.info.timestamp == 0x1122334455667788ull);  // from the first transfer
  CHECK(f.info.camera_field == 2);
  CHECK(f.info.camera_index() == 1u);
  CHECK(f.info.sub_index == 1);
  CHECK(f.data[0] == 0xAA);
  CHECK(f.data[63] == 0xBB);
}

TEST(short_last_transfer_is_size_mismatch_and_resets) {
  FrameAssembler a(64);
  CHECK(a.push(transfer(16, false, 1, 0, 0, 1)) == StreamEvent::Accepted);
  CHECK(a.push(transfer(16, true, 1, 1, 0, 1)) == StreamEvent::SizeMismatch);
  // A fresh, correctly sized frame still assembles afterwards.
  CHECK(a.push(transfer(64, true, 1, 2, 0, 1)) == StreamEvent::FrameComplete);
}

TEST(header_only_transfer_is_rejected) {
  FrameAssembler a(64);
  std::vector<std::uint8_t> t(kStreamHeaderSize, 0);
  CHECK(a.push(t) == StreamEvent::TooShort);
}

TEST(overflow_restarts_assembly) {
  FrameAssembler a(64);
  CHECK(a.push(transfer(48, false, 1, 0, 0, 1)) == StreamEvent::Accepted);
  CHECK(a.push(transfer(48, false, 1, 0, 0, 1)) == StreamEvent::Overflow);
}

TEST(camera_field_maps_bitmask_to_index) {
  FrameInfo i;
  i.camera_field = 4;
  CHECK(i.camera_index() == 2u);
  i.camera_field = 3;
  CHECK(!i.camera_index().has_value());
}
