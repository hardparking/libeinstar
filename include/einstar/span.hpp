// Minimal C++17 stand-in for std::span, so the public API builds in C++17
// projects such as RTAB-Map.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace einstar {

template <typename T>
class Span {
 public:
  constexpr Span() = default;
  constexpr Span(T* data, std::size_t size) : data_(data), size_(size) {}
  template <std::size_t N>
  constexpr Span(T (&array)[N]) : data_(array), size_(N) {}
  // Any contiguous container (std::array, std::vector, ...). Temporaries are
  // accepted for call-duration use.
  template <typename C,
            typename = std::enable_if_t<!std::is_base_of_v<Span, std::decay_t<C>> &&
                                        std::is_convertible_v<decltype(std::declval<C&>().data()), T*>>>
  constexpr Span(C&& c) : data_(c.data()), size_(c.size()) {}
  template <typename U, typename = std::enable_if_t<std::is_same_v<const U, T>>>
  constexpr Span(Span<U> other) : data_(other.data()), size_(other.size()) {}

  constexpr T* data() const { return data_; }
  constexpr std::size_t size() const { return size_; }
  constexpr bool empty() const { return size_ == 0; }
  constexpr T* begin() const { return data_; }
  constexpr T* end() const { return data_ + size_; }
  constexpr T& operator[](std::size_t i) const { return data_[i]; }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

using ByteView = Span<const std::uint8_t>;
using MutableByteView = Span<std::uint8_t>;

}  // namespace einstar
