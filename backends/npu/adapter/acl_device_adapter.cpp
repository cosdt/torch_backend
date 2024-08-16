#include <mutex>
#include "acl/include/acl/acl.h"
#include "core/NPUException.h"
#include "core/npu_log.h"
#include "csrc/adapter/device_adapter.h"

namespace acl_adapter {

static std::unordered_map<c10::DeviceIndex, aclrtContext> used_devices;
std::mutex mtx;

DeviceError Init() {
  auto init_ret = aclInit(nullptr);
  if (init_ret == ACL_ERROR_REPEAT_INITIALIZE) {
    // do nothing.
  } else if (init_ret != ACL_ERROR_NONE) {
    NPU_CHECK_ERROR(init_ret, "aclInit");
  }
  return init_ret;
}

DeviceError Finalize() {
  NPU_CHECK_WARN(aclFinalize());
}

aclError GetDevice(int32_t* deviceId) {
  auto err = ::aclrtGetDevice(deviceId);
  if (err == ACL_ERROR_NONE) {
    return ACL_ERROR_NONE;
  }

  // If ::SetDevice() has never been called, then device is set to 0.
  if (err == ACL_ERROR_RT_CONTEXT_NULL && SetDevice(0) == ACL_ERROR_NONE) {
    *deviceId = 0;
    return ACL_ERROR_NONE;
  }

  return err;
}

// ::SetDevice() will create a context implicitly. Save it when setting
// device.
aclError SetDevice(int32_t deviceId) {
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

// Each device that has been set has a primary context.
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

std::vector<c10::DeviceIndex> GetUsedDevices() {
  std::vector<c10::DeviceIndex> device_idx_vec;
  for (auto pair : used_devices) {
    device_idx_vec.push_back(pair.first);
  }
  return device_idx_vec;
}

void SynchronizeAllDevice() {
  int32_t cur_device = 0;
  NPU_CHECK_ERROR(GetDevice(&cur_device));
  std::vector<c10::DeviceIndex> device_idx_vec = acl_adapter::GetUsedDevices();
  for (const auto deviceId : device_idx_vec) {
    NPU_CHECK_ERROR(SetDevice(deviceId));
    NPU_CHECK_ERROR(aclrtSynchronizeDevice());
  }
  NPU_CHECK_ERROR(SetDevice(cur_device));
}

void CreateStream(aclrtStream* stream, uint32_t priority, uint32_t configFlag) {
  NPU_CHECK_SUPPORTED_OR_ERROR(
      aclrtCreateStreamWithConfig(stream, priority, configFlag));
}

DeviceError GetDeviceCount(uint32_t* dev_count) {
  return aclrtGetDeviceCount(dev_count);
}

void SynchronizeDevice() {
  NPU_CHECK_ERROR(aclrtSynchronizeDevice());
}

} // namespace acl_adapter
