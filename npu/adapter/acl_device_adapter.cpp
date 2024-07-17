#include "npu/adapter/acl_device_adapter.h"
#include <mutex>
#include "npu/core/NPUException.h"
#include "npu/core/npu_log.h"

namespace acl_adapter {

static std::unordered_map<c10::DeviceIndex, aclrtContext> used_devices;
std::mutex mtx;

aclError aclrtGetDevice(int32_t* deviceId) {
  auto err = ::aclrtGetDevice(deviceId);
  if (err == ACL_ERROR_NONE) {
    return ACL_ERROR_NONE;
  }

  // If ::aclrtSetDevice has never been called, then device is set to 0
  if (err == ACL_ERROR_RT_CONTEXT_NULL && aclrtSetDevice(0) == ACL_ERROR_NONE) {
    *deviceId = 0;
    return ACL_ERROR_NONE;
  }

  return err;
}

// ::aclrtSetDevice will create a context implicitly. Save it when setting
// device.
aclError aclrtSetDevice(int32_t deviceId) {
  aclError err = ::aclrtSetDevice(deviceId);
  if (err == ACL_ERROR_NONE) {
    auto device = static_cast<c10::DeviceIndex>(deviceId);
    if (used_devices.find(device) == used_devices.end()) {
      std::lock_guard<std::mutex> lock(mtx);
      if (used_devices.find(device) == used_devices.end()) {
        NPU_CHECK_ERROR(aclrtGetCurrentContext(&used_devices[device]));
      }
    }
  }
  return err;
}

bool hasPrimaryContext(c10::DeviceIndex device_index) {
  return used_devices.find(device_index) != used_devices.end();
}

aclrtContext GetDeviceContext(c10::DeviceIndex device) {
  if (used_devices.find(device) == used_devices.end()) {
    ASCEND_LOGE(
        "NPU device %d has been initialized! Can not get context", device);
    return nullptr;
  }
  return used_devices[device];
}

aclError ResetUsedDevices() {
  for (const auto it : used_devices) {
    aclError err = aclrtResetDevice(it.first);
    if (err != ACL_ERROR_NONE) {
      return err;
    }
  }
  used_devices.clear();
  return ACL_ERROR_NONE;
}

aclError DestroyUsedStreams() {
  c10::DeviceIndex cur_device = 0;
  NPU_CHECK_ERROR(GetDevice(&cur_device));
  for (const auto it : used_devices) {
    NPU_CHECK_ERROR(SetDevice(it.first));
    NPUStream stream = getCurrentNPUStream(it.first);
    aclError acl_ret = acl::AclrtDestroyStreamForce(stream);
    if (acl_ret != ACL_ERROR_NONE) {
      return acl_ret;
    }
  }
  NPU_CHECK_ERROR(SetDevice(cur_device));
  return ACL_ERROR_NONE;
}

aclError SynchronizeUsedDevices() {
  c10::DeviceIndex cur_device = 0;
  NPU_CHECK_ERROR(GetDevice(&cur_device));
  for (const auto it : used_devices) {
    NPU_CHECK_ERROR(SetDevice(it.first));
    aclError acl_ret = aclrtSynchronizeDevice();
    if (acl_ret != ACL_ERROR_NONE) {
      return acl_ret;
    }
  }
  NPU_CHECK_ERROR(SetDevice(cur_device));
  return ACL_ERROR_NONE;
}

} // namespace acl_adapter
