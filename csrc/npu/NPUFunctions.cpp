#include "csrc/npu/NPUFunctions.h"
#include <mutex>
#include <unordered_map>
#include "npu/core/register/OptionsManager.h"

namespace c10_npu {

static thread_local c10::DeviceIndex local_device = -1;
// TODO: remove used_devices
static std::unordered_map<int8_t, aclrtContext> used_devices;
std::mutex mtx;

int device_count_impl() {
  int count = 0;
  NPU_CHECK_ERROR(GetDeviceCount(&count));
  return count;
}

c10::DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl();
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<c10::DeviceIndex>::max(),
          "Too many NPU devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      // msg() returns the message without the stack trace
      TORCH_WARN("NPU initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<c10::DeviceIndex>(count);
}

c10::DeviceIndex device_count_ensure_non_zero() {
  // Call the implementation every time to throw the exception
  int count = device_count_impl();
  // Zero npus doesn't produce a warning in `device_count` but we fail here
  TORCH_CHECK(count, "No NPUs are available", PTA_ERROR(ErrCode::UNAVAIL));
  TORCH_INTERNAL_ASSERT(
      count <= std::numeric_limits<c10::DeviceIndex>::max(),
      "Too many NPU devices, DeviceIndex overflowed");
  return static_cast<c10::DeviceIndex>(count);
}

c10::DeviceIndex current_device() {
  c10::DeviceIndex cur_device = -1;
  NPU_CHECK_ERROR(c10_npu::GetDevice(&cur_device));
  return cur_device;
}

void set_device(c10::DeviceIndex device) {
  NPU_CHECK_ERROR(c10_npu::SetDevice(device));
}

void device_synchronize() {
  NPU_CHECK_ERROR(aclrtSynchronizeDevice());
}

// this function has to be called from callers performing npu synchronizing
// operations, to raise proper error or warning
void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(
        false, "called a synchronizing NPU operation", PTA_ERROR(ErrCode::ACL));
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_NPU_WARN("called a synchronizing NPU operation");
  }
}

// Wrappers for raw CUDA device management functions
aclError GetDeviceCount(int* dev_count) {
  return aclrtGetDeviceCount(reinterpret_cast<uint32_t*>(dev_count));
}

aclError GetDevice(c10::DeviceIndex* device) {
  if (local_device >= 0) {
    *device = local_device;
    return ACL_ERROR_NONE;
  }
  int tmp_device = -1;
  auto err = aclrtGetDevice(&tmp_device);
  if (err == ACL_ERROR_NONE) {
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<c10::DeviceIndex>::max(),
        "aclrtGetDevice returns invalid device ",
        tmp_device);
    *device = static_cast<c10::DeviceIndex>(tmp_device);
    local_device = *device;
  } else if (
      err == ACL_ERROR_RT_CONTEXT_NULL && aclrtSetDevice(0) == ACL_ERROR_NONE) {
    *device = 0;
    local_device = 0;
    if (used_devices.find(local_device) == used_devices.end()) {
      std::lock_guard<std::mutex> lock(mtx);
      if (used_devices.find(local_device) == used_devices.end()) {
        NPU_CHECK_ERROR(aclrtGetCurrentContext(&used_devices[local_device]));
      }
    }
    return ACL_ERROR_NONE;
  }
  return err;
}

aclError SetDevice(c10::DeviceIndex device) {
  TORCH_CHECK(
      device >= 0, "device id must be positive!", PTA_ERROR(ErrCode::VALUE));

  if (local_device == device) {
    return ACL_ERROR_NONE;
  }

  aclError err = aclrtSetDevice(device);
  if (err == ACL_ERROR_NONE) {
    local_device = device;
    if (used_devices.find(local_device) == used_devices.end()) {
      std::lock_guard<std::mutex> lock(mtx);
      if (used_devices.find(local_device) == used_devices.end()) {
        NPU_CHECK_ERROR(aclrtGetCurrentContext(&used_devices[local_device]));
      }
    }
  }
  return err;
}

aclrtContext GetDeviceContext(int32_t device) {
  if (used_devices.find(device) == used_devices.end()) {
    ASCEND_LOGE(
        "NPU device %d has been initialized! Can not get context", device);
    return nullptr;
  }
  return used_devices[device];
}

int GetLocalDevice() {
  return local_device;
}

} // namespace c10_npu
