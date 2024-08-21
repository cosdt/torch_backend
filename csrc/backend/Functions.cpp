#include "csrc/backend/Functions.h"
#include <mutex>
#include <unordered_map>
#include "csrc/backend/Stream.h"

namespace c10::backend {

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
          "Too many devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      // msg() returns the message without the stack trace
      TORCH_WARN("Device initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<c10::DeviceIndex>(count);
}

c10::DeviceIndex device_count_ensure_non_zero() {
  // Call the implementation every time to throw the exception
  int count = device_count_impl();
  // Zero devices doesn't produce a warning in `device_count` but we fail here
  TORCH_CHECK(count, "No devices are available", PTA_ERROR(ErrCode::UNAVAIL));
  TORCH_INTERNAL_ASSERT(
      count <= std::numeric_limits<c10::DeviceIndex>::max(),
      "Too many devices devices, DeviceIndex overflowed");
  return static_cast<c10::DeviceIndex>(count);
}

c10::DeviceIndex current_device() {
  c10::DeviceIndex cur_device = -1;
  NPU_CHECK_ERROR(c10::backend::GetDevice(&cur_device));
  return cur_device;
}

void set_device(c10::DeviceIndex device) {
  NPU_CHECK_ERROR(c10::backend::SetDevice(device));
}

void device_synchronize() {
  DEVICE_NAMESPACE::SynchronizeDevice();
}

// this function has to be called from callers performing device synchronizing
// operations, to raise proper error or warning
void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(
        false,
        "called a synchronizing device operation",
        PTA_ERROR(ErrCode::ACL));
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_NPU_WARN("called a synchronizing device operation");
  }
}

std::optional<c10::DeviceIndex> getDeviceIndexWithPrimaryContext() {
  // check current device first
  auto current_device_index = current_device();
  if (current_device_index >= 0) {
    if (hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (const auto device_index : c10::irange(device_count())) {
    if (device_index == current_device_index)
      continue;
    if (hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return c10::nullopt;
}

bool hasPrimaryContext(c10::DeviceIndex device_index) {
  return DEVICE_NAMESPACE::hasPrimaryContext(device_index);
}

// Wrappers for raw CUDA device management functions
deviceError_t GetDeviceCount(int* dev_count) {
  return DEVICE_NAMESPACE::GetDeviceCount(
      reinterpret_cast<uint32_t*>(dev_count));
}

thread_local c10::DeviceIndex targetDeviceIndex = -1;

deviceError_t InitDevice() {
  return DEVICE_NAMESPACE::Init();
}

void FinalizeDevice() {
  DEVICE_NAMESPACE::Finalize();
}

deviceError_t GetDevice(c10::DeviceIndex* device) {
  if (targetDeviceIndex >= 0) {
    *device = targetDeviceIndex;
    return ACL_ERROR_NONE;
  }
  int tmp_device = -1;
  auto err = DEVICE_NAMESPACE::GetDevice(&tmp_device);
  if (err == ACL_ERROR_NONE) {
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<c10::DeviceIndex>::max(),
        "GetDevice returns invalid device ",
        tmp_device);
    *device = static_cast<c10::DeviceIndex>(tmp_device);
  }
  return err;
}

deviceError_t SetDevice(c10::DeviceIndex device) {
  TORCH_CHECK(
      device >= 0, "device id must be positive!", PTA_ERROR(ErrCode::VALUE));
  targetDeviceIndex = -1;
  int cur_device = -1;
  NPU_CHECK_ERROR(DEVICE_NAMESPACE::GetDevice(&cur_device));
  if (device == cur_device) {
    return ACL_ERROR_NONE;
  }
  return DEVICE_NAMESPACE::SetDevice(device);
}

deviceError_t MaybeSetDevice(c10::DeviceIndex device) {
  if (hasPrimaryContext(device)) {
    return c10::backend::SetDevice(device);
  }
  targetDeviceIndex = device;
  return ACL_ERROR_NONE;
}

// This function always initializes the context
// on to_device
c10::DeviceIndex ExchangeDevice(c10::DeviceIndex to_device) {
  auto cur_device = targetDeviceIndex;
  targetDeviceIndex = -1;
  if (cur_device < 0) {
    int tmp_device = -1;
    NPU_CHECK_ERROR(DEVICE_NAMESPACE::GetDevice(&tmp_device));
    cur_device = static_cast<c10::DeviceIndex>(tmp_device);
    if (to_device == cur_device) {
      return cur_device;
    }
  }
  NPU_CHECK_ERROR(DEVICE_NAMESPACE::SetDevice(to_device));
  return cur_device;
}

// This function does not initialize the context
// on to_device if it does not already exist
c10::DeviceIndex MaybeExchangeDevice(c10::DeviceIndex to_device) {
  int tmp_cur_device = -1;
  NPU_CHECK_ERROR(DEVICE_NAMESPACE::GetDevice(&tmp_cur_device));
  TORCH_INTERNAL_ASSERT(
      tmp_cur_device >= 0 &&
          tmp_cur_device <= std::numeric_limits<c10::DeviceIndex>::max(),
      "GetDevice returns invalid device ",
      tmp_cur_device);
  auto cur_device = static_cast<c10::DeviceIndex>(tmp_cur_device);
  if (to_device == tmp_cur_device) {
    return cur_device;
  }
  if (hasPrimaryContext(to_device)) {
    NPU_CHECK_ERROR(DEVICE_NAMESPACE::SetDevice(to_device));
  } else {
    targetDeviceIndex = to_device;
  }
  return cur_device;
}

void SetTargetDevice() {
  if (targetDeviceIndex >= 0) {
    NPU_CHECK_ERROR(c10::backend::SetDevice(targetDeviceIndex));
  }
}

DeviceContext GetDeviceContext(c10::DeviceIndex device) {
  return DEVICE_NAMESPACE::GetDeviceContext(device);
}

std::mutex* getFreeMutex() {
  static std::mutex free_mutex;
  return &free_mutex;
}

void get_device_properties(
    c10::backend::DeviceProp* device_prop,
    c10::DeviceIndex device) {
  const char* device_name;
  device_name = aclrtGetSocName();
  if (device_name == nullptr) {
    device_prop->name = " ";
  } else {
    device_prop->name = std::string(device_name);
  }
}

} // namespace c10::backend
