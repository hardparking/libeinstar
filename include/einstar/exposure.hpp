// IR auto-exposure for scanning.
//
// The fixed exposure EXStar starts with (4000 us) saturates close objects
// (half of a frame at ~25 cm on hardware), which destroys the speckle stereo
// matching needs; EXStar itself kept adjusting exposure (1200-5000 us in its
// logs). This keeps the brightest speckle just below saturation.
//
// The scanner's light pulses follow the exposure, so every change is visible
// as a change in brightness. Steering on the bright end of the histogram
// ignores the dark background around a scanned object, a band around the
// target holds the exposure steady, and taking half of each needed
// correction keeps the loop stable although a new exposure only shows up a
// frame or two later. (The previous whole-frame mean/saturation rule changed
// the exposure on 98% of live frames, which blinked the light.)
#pragma once

#include <cstdint>
#include <optional>

#include "einstar/span.hpp"

namespace einstar {

struct AutoExposureParams {
  double percentile = 0.99;      // steer on this brightness percentile
  double target = 220;           // wanted value of that percentile
  double band_low = 195;         // no change while the percentile is within the band
  double band_high = 240;
  double clipped = 250;          // percentile at or above this: saturated, shorten
  double unlit = 40;             // percentile below this: nothing lit, lengthen
  double max_shorten = 0.7;      // per-update limits on the exposure ratio
  double max_lengthen = 1.25;
  std::uint32_t min_us = 200;
  std::uint32_t max_us = 6000;
  std::size_t sample_step = 31;  // sample every Nth pixel
};

class AutoExposure {
 public:
  explicit AutoExposure(AutoExposureParams params = {}) : p_(params) {}
  // Returns a new exposure when `image` (8-bit IR) calls for one.
  std::optional<std::uint32_t> update(ByteView image, std::uint32_t current_us) const;
  // Both IR cameras share one exposure and see the speckle at different
  // brightness (~35% apart on hardware): returns the shorter of the two
  // wanted exposures, where a camera within its band wants the current one,
  // so the dimmer view cannot lengthen the exposure past the brighter
  // view's band.
  std::optional<std::uint32_t> update(ByteView left, ByteView right, std::uint32_t current_us) const;

 private:
  std::uint32_t wanted(ByteView image, std::uint32_t current_us) const;
  AutoExposureParams p_;
};

}  // namespace einstar
