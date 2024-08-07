#include "npu/core/sys_ctrl/npu_sys_ctrl.h"
#include "csrc/npu/NPUCachingAllocator.h"
#include "csrc/npu/NPUCachingHostAllocator.h"
#include "csrc/npu/NPUFunctions.h"
#include "csrc/npu/NPUStream.h"
#include "npu/acl/include/acl/acl_op_compiler.h"
#include "npu/core/NpuVariables.h"
#include "npu/framework/interface/AclOpCompileInterface.h"
#include "npu/adapter/acl_device_adapter.h"
#include "npu/core/register/OptionRegister.h"

namespace c10::npu {

void TryInitDevice(c10::DeviceIndex device_id) {
  static NpuSysCtrl device(device_id);
}

NpuSysCtrl::NpuSysCtrl(c10::DeviceIndex device_id) : need_finalize_(true) {
  aclError ret = c10::npu::InitDevice();
  if (ret == ACL_ERROR_REPEAT_INITIALIZE) {
    need_finalize_ = false;
  }
  // Init allocator
  c10::npu::NPUCachingAllocator::init(c10::backend::CachingAllocator::get());

  ret = c10::npu::GetDevice(&device_id);
  if (ret != ACL_ERROR_NONE) {
    device_id = (device_id == -1) ? 0 : device_id;
    NPU_CHECK_ERROR(c10::npu::SetDevice(device_id));
  } 

  // set default jit_Compile value from Get acl defalut value
  c10::npu::option::SetOption("jitCompile", "disable");
}

NpuSysCtrl::~NpuSysCtrl() {
  NPUCachingHostAllocator_emptyCache();
  c10::npu::NPUCachingAllocator::emptyCache();

  NPU_CHECK_WARN(c10::npu::DestroyUsedStreams());
  NPU_CHECK_WARN(acl_adapter::ResetUsedDevices());
  // Maintain a basic point of view, who applies for the resource, the
  // resource is released by whom. If aclInit is not a PTA call, then
  // aclFinalize should not be a PTA call either.

  // TODO: The order of destruction cannot be guaranteed. Finalize is not
  // performed to ensure that the program runs normally.
  // if (need_finalize_) {
  //   c10::npu::FinalizeDevice();
  // }
}

} // namespace c10::npu
