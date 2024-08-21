#include "core/NpuDeviceRAII.h"
#include "acl/include/acl/acl_op_compiler.h"
#include "core/NpuVariables.h"
#include "core/register/OptionRegister.h"
#include "csrc/adapter/device_adapter.h"
#include "csrc/backend/CachingHostAllocator.h"
#include "csrc/backend/DeviceCachingAllocator.h"
#include "csrc/backend/Functions.h"
#include "csrc/backend/Stream.h"
#include "framework/interface/AclOpCompileInterface.h"

namespace c10::npu {

class NPUDeviceRAII {
 public:
  virtual ~NPUDeviceRAII();

  friend void TryInitDevice();

 private:
  NPUDeviceRAII();
  bool need_finalize_;
};

void TryInitDevice() {
  static NPUDeviceRAII device;
}

NPUDeviceRAII::NPUDeviceRAII() : need_finalize_(true) {
  aclError ret = c10::backend::InitDevice();
  if (ret == ACL_ERROR_REPEAT_INITIALIZE) {
    need_finalize_ = false;
  }
  // Init allocator
  c10::backend::Allocator::init(c10::backend::CachingAllocator::get());

  c10::DeviceIndex device_id;
  ret = c10::backend::GetDevice(&device_id);
  if (ret != ACL_ERROR_NONE) {
    device_id = (device_id == -1) ? 0 : device_id;
    NPU_CHECK_ERROR(c10::backend::SetDevice(device_id));
  }

  // set default jit_Compile value from Get acl defalut value
  c10::npu::option::SetOption("jitCompile", "disable");
}

NPUDeviceRAII::~NPUDeviceRAII() {
  c10::backend::HostAllocator::emptyCache();
  c10::backend::Allocator::emptyCache();

  NPU_CHECK_WARN(c10::backend::DestroyUsedStreams());
  NPU_CHECK_WARN(DEVICE_NAMESPACE::ResetUsedDevices());
  // Maintain a basic point of view, who applies for the resource, the
  // resource is released by whom. If aclInit is not a PTA call, then
  // aclFinalize should not be a PTA call either.

  // TODO: The order of destruction cannot be guaranteed. Finalize is not
  // performed to ensure that the program runs normally.
  // if (need_finalize_) {
  //   c10::backend::FinalizeDevice();
  // }
}

} // namespace c10::npu
