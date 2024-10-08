#pragma once

// This header provides C++ wrappers around commonly used AscendCL API
// functions. The benefit of using C++ here is that we can raise an exception in
// the event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of
// torch.backend

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <mutex>
#include <optional>
#include "csrc/backend/DeviceProp.h"
#include "csrc/core/Macros.h"

// TODO(FFFrog):
// Remove later
#include "core/NPUException.h"
#include "csrc/adapter/device_adapter.h"

namespace c10::backend {

C10_BACKEND_API c10::DeviceIndex device_count() noexcept;

// Version of device_count that throws is no devices are detected
C10_BACKEND_API c10::DeviceIndex device_count_ensure_non_zero();

C10_BACKEND_API c10::DeviceIndex current_device();

C10_BACKEND_API void set_device(c10::DeviceIndex device);

C10_BACKEND_API void device_synchronize();

// this function has to be called from callers performing deivce synchronizing
// operations, to raise proper error or warning
C10_BACKEND_API void warn_or_error_on_sync();

// Raw CUDA device management functions
C10_BACKEND_API deviceError_t GetDeviceCount(int* dev_count);

C10_BACKEND_API deviceError_t InitDevice();

C10_BACKEND_API void FinalizeDevice();

C10_BACKEND_API deviceError_t GetDevice(c10::DeviceIndex* device);

C10_BACKEND_API deviceError_t SetDevice(c10::DeviceIndex device);

C10_BACKEND_API deviceError_t MaybeSetDevice(c10::DeviceIndex device);

C10_BACKEND_API c10::DeviceIndex ExchangeDevice(c10::DeviceIndex device);

C10_BACKEND_API c10::DeviceIndex MaybeExchangeDevice(c10::DeviceIndex device);

C10_BACKEND_API void SetTargetDevice();

C10_BACKEND_API aclrtContext GetDeviceContext(c10::DeviceIndex device);

enum class SyncDebugMode { L_DISABLED = 0, L_WARN, L_ERROR };

// it's used to store device synchronization state
// through this global state to determine the synchronization debug mode
class WarningState {
 public:
  void set_sync_debug_mode(SyncDebugMode l) {
    sync_debug_mode = l;
  }

  SyncDebugMode get_sync_debug_mode() {
    return sync_debug_mode;
  }

 private:
  SyncDebugMode sync_debug_mode = SyncDebugMode::L_DISABLED;
};

C10_BACKEND_API __inline__ WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}

C10_BACKEND_API bool hasPrimaryContext(c10::DeviceIndex device_index);
C10_BACKEND_API std::optional<c10::DeviceIndex>
getDeviceIndexWithPrimaryContext();

C10_BACKEND_API std::mutex* getFreeMutex();

C10_BACKEND_API void get_device_properties(
    c10::backend::DeviceProp* device_prop,
    c10::DeviceIndex device);

} // namespace c10::backend
