#include "einstar/exposure.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace einstar {

std::uint32_t AutoExposure::wanted(ByteView image, std::uint32_t current_us) const {
  if (image.empty() || p_.sample_step == 0) return current_us;
  std::array<std::size_t, 256> histogram{};
  std::size_t n = 0;
  for (std::size_t i = 0; i < image.size(); i += p_.sample_step) {
    ++histogram[image[i]];
    ++n;
  }
  const auto rank = static_cast<std::size_t>(p_.percentile * static_cast<double>(n - 1));
  std::size_t seen = 0;
  int level = 0;
  for (; level < 255; ++level) {
    seen += histogram[level];
    if (seen > rank) break;
  }
  double ratio = 1.0;
  if (level >= p_.clipped) ratio = p_.max_shorten;
  else if (level < p_.unlit) ratio = p_.max_lengthen;
  else if (level < p_.band_low || level > p_.band_high)
    ratio = std::clamp(std::sqrt(p_.target / level), p_.max_shorten, p_.max_lengthen);
  if (ratio == 1.0) return current_us;
  return static_cast<std::uint32_t>(
      std::clamp(std::lround(current_us * ratio), static_cast<long>(p_.min_us), static_cast<long>(p_.max_us)));
}

std::optional<std::uint32_t> AutoExposure::update(ByteView image, std::uint32_t current_us) const {
  const std::uint32_t next = wanted(image, current_us);
  if (next == current_us) return std::nullopt;
  return next;
}

std::optional<std::uint32_t> AutoExposure::update(ByteView left, ByteView right, std::uint32_t current_us) const {
  const std::uint32_t next = std::min(wanted(left, current_us), wanted(right, current_us));
  if (next == current_us) return std::nullopt;
  return next;
}

}  // namespace einstar
