#pragma once

#include "csrc/npu/NPUDeviceProp.h"
#include "csrc/npu/NPUFunctions.h"

namespace c10::npu {

// NPU is available if we compiled with NPU.
inline bool is_available() {
  return c10::npu::device_count() > 0;
}

NPUDeviceProp* getDeviceProperties(c10::DeviceIndex device);

} // namespace c10::npu
