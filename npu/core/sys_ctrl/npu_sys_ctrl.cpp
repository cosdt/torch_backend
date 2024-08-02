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

NpuSysCtrl::NpuSysCtrl()
    : repeat_init_acl_flag_(true), init_flag_(false), device_id_(0) {}

// Get NpuSysCtrl singleton instance
NpuSysCtrl& NpuSysCtrl::GetInstance() {
  static NpuSysCtrl instance;
  return instance;
}


bool NpuSysCtrl::IsInitializeSuccess(int device_id) {
  SysStatus status = GetInstance().Initialize(device_id);
  return status == SysStatus::INIT_SUCC;
}

bool NpuSysCtrl::IsFinalizeSuccess() {
  SysStatus status = GetInstance().Finalize();
  return status == SysStatus::FINALIZE_SUCC;
}

// Environment Initialize, return Status: SUCCESS, FAILED
NpuSysCtrl::SysStatus NpuSysCtrl::Initialize(int device_id) {
  if (init_flag_) {
    return INIT_SUCC;
  }
  auto init_ret = aclInit(nullptr);

  if (init_ret != ACL_ERROR_NONE) {
    NPU_CHECK_ERROR(init_ret, "aclInit");
  }

  // Init allocator
  static c10_npu::NPUCachingAllocator::CachingAllocatorHelper helper;
  c10_backend::CachingAllocator::registerHelper(&helper);
  const auto num_devices = c10_npu::device_count_ensure_non_zero();
  c10_backend::CachingAllocator::init(num_devices);

  c10_npu::NPUCachingAllocator::init(c10_backend::CachingAllocator::get());

  // There's no need to call c10_npu::GetDevice at the start of the process,
  // because device 0 may not be needed
  auto ret = aclrtGetDevice(&device_id_);
  if (ret != ACL_ERROR_NONE) {
    device_id_ = (device_id == -1) ? 0 : device_id;
    NPU_CHECK_ERROR(c10_npu::SetDevice(device_id_));
  } else {
    ASCEND_LOGW("Npu device %d has been set before global init.", device_id_);
  }

  // set default jit_Compile value from Get acl defalut value
  c10_npu::option::SetOption("jitCompile", "disable");

  init_flag_ = true;
  ASCEND_LOGD("Npu sys ctrl initialize successfully.");

  return INIT_SUCC;
}

// Environment Finalize, return SysStatus
NpuSysCtrl::SysStatus NpuSysCtrl::Finalize() {
  if (!init_flag_) {
    return FINALIZE_SUCC;
  }

  c10_npu::NPUEventManager::GetInstance().ClearEvent();
  NPU_CHECK_WARN(c10_npu::DestroyUsedStreams());
  NPU_CHECK_WARN(c10_npu::ResetUsedDevices());
  // Maintain a basic point of view, who applies for the resource, the
  // resource is released by whom. If aclInit is not a PTA call, then
  // aclFinalize should not be a PTA call either.
  if (repeat_init_acl_flag_) {
    NPU_CHECK_WARN(aclFinalize());
  }

  init_flag_ = false;

  ASCEND_LOGD("Npu sys ctrl finalize successfully.");
  return FINALIZE_SUCC;
}

bool NpuSysCtrl::GetInitFlag() {
  return init_flag_;
}

} // namespace c10_npu
