#pragma once

#include <string>

namespace c10::backend {

struct DeviceProp {
  std::string name{};
  size_t totalGlobalMem = 0;
};

} // namespace c10::backend
