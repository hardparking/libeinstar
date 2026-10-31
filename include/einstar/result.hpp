// Minimal expected-style result used across the public API.
#pragma once

#include <string>
#include <utility>
#include <variant>

namespace einstar {

enum class ErrorCode {
  UsbError,
  NotFound,
  NotOpen,
  Timeout,
  BadReply,
  WritesDisabled,
  InvalidArgument,
};

const char* to_string(ErrorCode c);

struct Error {
  ErrorCode code;
  std::string message;
};

template <typename T>
class Result {
 public:
  Result(T value) : v_(std::move(value)) {}
  Result(Error error) : v_(std::move(error)) {}

  bool ok() const { return std::holds_alternative<T>(v_); }
  explicit operator bool() const { return ok(); }
  const T& value() const { return std::get<T>(v_); }
  T& value() { return std::get<T>(v_); }
  const Error& error() const { return std::get<Error>(v_); }

 private:
  std::variant<T, Error> v_;
};

struct Ok {};
using Status = Result<Ok>;

}  // namespace einstar
