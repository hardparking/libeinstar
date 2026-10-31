#include "check.hpp"

namespace einstar::test {
std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}
int failures = 0;
}  // namespace einstar::test

int main() {
  using namespace einstar::test;
  for (auto& c : registry()) {
    const int before = failures;
    c.fn();
    std::printf("%s %s\n", failures == before ? "PASS" : "FAIL", c.name);
  }
  std::printf("%zu tests, %d failed checks\n", registry().size(), failures);
  return failures == 0 ? 0 : 1;
}
