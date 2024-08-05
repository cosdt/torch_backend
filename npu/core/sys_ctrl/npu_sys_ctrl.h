#pragma once

#include <npu/acl/include/acl/acl.h>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "c10/macros/Export.h"
#include "csrc/core/Macros.h"
#include "npu/core/npu_log.h"
#include "c10/core/Device.h"

namespace c10_npu {

void TryInitDevice(c10::DeviceIndex device_id = -1);
class NpuSysCtrl {
 public:
  virtual ~NpuSysCtrl();

  friend void TryInitDevice(c10::DeviceIndex device_id);
 private:
  NpuSysCtrl(c10::DeviceIndex device_id);
  bool need_finalize_;
};

} // namespace c10_npu
