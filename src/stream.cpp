#include "einstar/stream.hpp"

#include <algorithm>

#include "einstar/protocol.hpp"

namespace einstar {

std::optional<unsigned> FrameInfo::camera_index() const {
  switch (camera_field) {
    case 1: return 0u;
    case 2: return 1u;
    case 4: return 2u;
    default: return std::nullopt;
  }
}

FrameAssembler::FrameAssembler(std::size_t frame_size)
    : frame_size_(frame_size), buffer_(frame_size) {}

void FrameAssembler::reset() { filled_ = 0; }

StreamEvent FrameAssembler::push(ByteView t) {
  if (t.size() <= kStreamHeaderSize) return StreamEvent::TooShort;
  const bool last = t[kOffStreamLastFlag] == 1;
  const std::size_t chunk = t.size() - kStreamHeaderSize;

  if (last && filled_ + chunk != frame_size_) {
    reset();
    return StreamEvent::SizeMismatch;
  }
  if (filled_ == 0) timestamp_ = read_be64(t, kOffStreamTimestamp);

  StreamEvent result = StreamEvent::Accepted;
  if (filled_ + chunk > frame_size_) {
    reset();
    result = StreamEvent::Overflow;
  }
  const std::size_t n = std::min(frame_size_ - filled_, chunk);
  std::copy_n(t.begin() + kStreamPayloadOffset, n, buffer_.begin() + filled_);
  filled_ += chunk;

  if (filled_ == frame_size_ && last) {
    ready_.info.timestamp = timestamp_;
    ready_.info.frame_id = read_be32(t, kOffStreamFrameId);
    ready_.info.camera_field = (t[kOffStreamCamera] >> 3) & 0x7;
    ready_.info.sub_index = t[kOffStreamCamera] & 0x7;
    ready_.data = buffer_;
    reset();
    return StreamEvent::FrameComplete;
  }
  if (last) reset();
  return result;
}

Frame FrameAssembler::take() { return std::move(ready_); }

}  // namespace einstar
