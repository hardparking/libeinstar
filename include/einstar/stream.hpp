// Reassembles image frames from EP 0x83 stream transfers.
//
// Mirrors ImageStreamThread::run in the vendor plugin: every transfer starts
// with a 32-byte header; payload bytes are appended until the expected frame
// size is reached on a transfer flagged as the frame's last.
//
// Offsets are relative to the raw USB transfer. The vendor comm layer stores
// each transfer behind a 4-byte big-endian length prefix
// (captureImageTransferCallBack), so the plugin's offsets are these plus 4.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "einstar/span.hpp"

namespace einstar {

inline constexpr std::size_t kStreamHeaderSize = 0x20;
inline constexpr std::size_t kStreamPayloadOffset = kStreamHeaderSize;
inline constexpr std::size_t kOffStreamCamera = 0x02;
inline constexpr std::size_t kOffStreamLastFlag = 0x08;
inline constexpr std::size_t kOffStreamFrameId = 0x0c;
inline constexpr std::size_t kOffStreamTimestamp = 0x10;

struct FrameInfo {
  std::uint64_t timestamp = 0;  // header 0x10..0x17 of the first transfer
  std::uint32_t frame_id = 0;   // header 0x0c..0x0f of the last transfer
  std::uint8_t camera_field = 0;  // header 0x02 bits 3..5 (1, 2 or 4)
  std::uint8_t sub_index = 0;     // header 0x02 bits 0..2
  // 0 and 1 are the IR cameras, 2 the colour camera; nullopt if unrecognised.
  std::optional<unsigned> camera_index() const;
};

struct Frame {
  FrameInfo info;
  std::vector<std::uint8_t> data;
};

enum class StreamEvent {
  Accepted,
  FrameComplete,
  TooShort,        // transfer not larger than its header
  SizeMismatch,    // last-flagged transfer does not complete the frame
  Overflow,        // payload would exceed the frame; assembly restarted
};

class FrameAssembler {
 public:
  explicit FrameAssembler(std::size_t frame_size);

  // Feeds one transfer. On FrameComplete the frame is available via take().
  StreamEvent push(ByteView transfer);
  Frame take();
  void reset();
  std::size_t frame_size() const { return frame_size_; }

 private:
  std::size_t frame_size_;
  std::size_t filled_ = 0;
  std::uint64_t timestamp_ = 0;
  Frame ready_;
  std::vector<std::uint8_t> buffer_;
};

}  // namespace einstar
