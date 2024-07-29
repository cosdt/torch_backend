#pragma once

#include <string>

namespace c10_npu {

struct NPUDeviceProp {
  std::string name{};
  size_t totalGlobalMem = 0;
};

} // namespace c10_npu
