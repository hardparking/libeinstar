// Tiny self-registering test harness; avoids a test-framework dependency.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace einstar::test {

struct Case {
  const char* name;
  std::function<void()> fn;
};

std::vector<Case>& registry();
extern int failures;

struct Register {
  Register(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

}  // namespace einstar::test

#define EINSTAR_CAT2(a, b) a##b
#define EINSTAR_CAT(a, b) EINSTAR_CAT2(a, b)
#define TEST(name)                                                              \
  static void name();                                                           \
  static ::einstar::test::Register EINSTAR_CAT(reg_, name)(#name, name);        \
  static void name()

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
      ++::einstar::test::failures;                                              \
    }                                                                           \
  } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::abs((a) - (b)) <= (tol))
