#include "npu/core/sys_ctrl/npu_sys_ctrl.h"
#include "csrc/npu/NPUCachingAllocator.h"
#include "csrc/npu/NPUFunctions.h"
#include "csrc/npu/NPUStream.h"
#include "npu/acl/include/acl/acl_op_compiler.h"
#include "npu/core/NPUCachingAllocatorHelper.h"
#include "npu/core/NpuVariables.h"
#include "npu/core/npu_log.h"
#include "npu/core/register/OptionRegister.h"
#include "npu/core/register/OptionsManager.h"
#include "npu/framework/interface/AclOpCompileInterface.h"
#ifdef SUCCESS
#undef SUCCESS
#endif
#ifdef FAILED
#undef FAILED
#endif

#if defined(_MSC_VER)
#include <direct.h>
#elif defined(__unix__)
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#else
#endif

namespace c10_npu {

void TryInitDevice(c10::DeviceIndex device_id) {
  static NpuSysCtrl device(device_id);
}

NpuSysCtrl::NpuSysCtrl(c10::DeviceIndex device_id) : need_finalize_(true){
  aclError ret = c10_npu::InitDevice();
  if(ret == ACL_ERROR_REPEAT_INITIALIZE) {
    need_finalize_ = false;
  }

  // Init allocator
  static c10_npu::NPUCachingAllocator::CachingAllocatorHelper helper;
  c10_backend::CachingAllocator::registerHelper(&helper);
  const auto num_devices = c10_npu::device_count_ensure_non_zero();
  c10_backend::CachingAllocator::init(num_devices);

  c10_npu::NPUCachingAllocator::init(c10_backend::CachingAllocator::get());

  ret = c10_npu::GetDevice(&device_id);
  if (ret != ACL_ERROR_NONE) {
    device_id = (device_id == -1) ? 0 : device_id;
    NPU_CHECK_ERROR(c10_npu::SetDevice(device_id));
  } else {
    ASCEND_LOGW("Npu device %d has been set before global init.", device_id);
  }

  // set default jit_Compile value from Get acl defalut value
  c10_npu::option::SetOption("jitCompile", "disable");

  ASCEND_LOGD("Npu sys ctrl initialize successfully.");
}

NpuSysCtrl::~NpuSysCtrl() {
  NPU_CHECK_WARN(c10_npu::DestroyUsedStreams());
  NPU_CHECK_WARN(c10_npu::ResetUsedDevices());
  // Maintain a basic point of view, who applies for the resource, the
  // resource is released by whom. If aclInit is not a PTA call, then
  // aclFinalize should not be a PTA call either.

  // TODO: The order of destruction cannot be guaranteed. Finalize is not
  // performed to ensure that the program runs normally.
  // if (need_finalize_) {
  //   c10_npu::FinalizeDevice();
  // }

  ASCEND_LOGD("Npu sys ctrl finalize successfully.");
}

} // namespace c10_npu
