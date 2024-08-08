#include "npu/core/NpuDeviceRAII.h"
#include "csrc/backend/NPUCachingAllocator.h"
#include "csrc/backend/NPUCachingHostAllocator.h"
#include "csrc/backend/NPUFunctions.h"
#include "csrc/backend/NPUStream.h"
#include "npu/acl/include/acl/acl_op_compiler.h"
#include "npu/adapter/acl_device_adapter.h"
#include "npu/core/NpuVariables.h"
#include "npu/core/register/OptionRegister.h"
#include "npu/framework/interface/AclOpCompileInterface.h"

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
  c10::npu::NPUCachingAllocator::init(c10::backend::CachingAllocator::get());

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
  NPUCachingHostAllocator_emptyCache();
  c10::npu::NPUCachingAllocator::emptyCache();

  NPU_CHECK_WARN(c10::backend::DestroyUsedStreams());
  NPU_CHECK_WARN(acl_adapter::ResetUsedDevices());
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
