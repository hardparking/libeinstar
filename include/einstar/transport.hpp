// USB transfer interface used by Device. The libusb implementation lives in
// device.cpp; tests drive Device through a simulated scanner instead.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace einstar {

// Transfer status codes (values match libusb so its codes pass through).
inline constexpr int kTransferOk = 0;
inline constexpr int kTransferTimeout = -7;

class Transport {
 public:
  virtual ~Transport() = default;
  // Synchronous transfers on one endpoint (direction from the address bit).
  // Return kTransferOk, kTransferTimeout or another negative error, and set
  // `transferred` to the bytes moved.
  virtual int interrupt_transfer(std::uint8_t endpoint, std::uint8_t* data, std::size_t length,
                                 std::size_t& transferred, unsigned timeout_ms) = 0;
  virtual int bulk_transfer(std::uint8_t endpoint, std::uint8_t* data, std::size_t length,
                            std::size_t& transferred, unsigned timeout_ms) = 0;
  virtual void clear_halt(std::uint8_t endpoint) = 0;
  virtual std::string describe(int status) const = 0;
};

}  // namespace einstar
