#pragma once

#include "csrc/backend/DeviceProp.h"
#include "csrc/backend/Functions.h"

namespace c10::backend {

// NPU is available if we compiled with NPU.
inline bool is_available() {
  return c10::backend::device_count() > 0;
}

DeviceProp* getDeviceProperties(c10::DeviceIndex device);

} // namespace c10::backend
