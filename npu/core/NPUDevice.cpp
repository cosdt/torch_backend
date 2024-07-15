#include "NPUDevice.h"
#include "npu/core/npu_log.h"

namespace c10_npu::acl {

c10::DeviceIndex device_count_impl(uint32_t* count) noexcept {
  aclError error = aclrtGetDeviceCount(count);
  if (error != ACL_ERROR_NONE) {
    ASCEND_LOGE("get device count of NPU failed");
    return 0;
  }

  return static_cast<c10::DeviceIndex>(*count);
}

} // namespace c10_npu::acl
