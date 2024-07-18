#include "npu/core/sys_ctrl/npu_sys_ctrl.h"
#include "csrc/npu/NPUCachingAllocator.h"
#include "csrc/npu/NPUFunctions.h"
#include "csrc/npu/NPUStream.h"
#include "npu/acl/include/acl/acl_op_compiler.h"
#include "npu/acl/include/acl/acl_rt.h"
#include "npu/core/NPUCachingAllocatorHelper.h"
#include "npu/core/NpuVariables.h"
#include "npu/core/interface/AclInterface.h"
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
#define GetCurrentDirPath _getcwd
#define Mkdir(path, mode) _mkdir(path)
#elif defined(__unix__)
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define GetCurrentDirPath getcwd
#define Mkdir(path, mode) mkdir(path, mode)
#else
#endif

namespace {
const uint32_t kMaxOpExecuteTimeOut = 547U;
const size_t kMaxPathLen = 4096U;

void MakeCompileCacheDirAndSetOption() {
  char* compile_cache_mode_val = std::getenv("ACL_OP_COMPILER_CACHE_MODE");
  std::string compile_cache_mode = (compile_cache_mode_val == nullptr)
      ? std::string("enable")
      : std::string(compile_cache_mode_val);
  if (compile_cache_mode != "enable" && compile_cache_mode != "disable" &&
      compile_cache_mode != "force") {
    compile_cache_mode = std::string("enable");
  }
  c10_npu::option::register_options::OptionRegister::GetInstance()->Set(
      "ACL_OP_COMPILER_CACHE_MODE", compile_cache_mode);

  char* compile_cache_dir_val = std::getenv("ACL_OP_COMPILER_CACHE_DIR");
  if (compile_cache_dir_val != nullptr) {
    std::string compile_cache_dir = std::string(compile_cache_dir_val);
    // mode : 750
    auto ret = Mkdir(compile_cache_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
    if (ret == -1) {
      if (errno != EEXIST) {
        TORCH_NPU_WARN("make compile cache directory error: ", strerror(errno));
        return;
      }
    }
    c10_npu::option::register_options::OptionRegister::GetInstance()->Set(
        "ACL_OP_COMPILER_CACHE_DIR", compile_cache_dir);
  }
}

void GetAndSetDefaultJitCompileByAcl() {
  auto opt_size = at_npu::native::AclGetCompileoptSize(ACL_OP_JIT_COMPILE);
  if (!opt_size.has_value()) {
    ASCEND_LOGW(
        "Get ACL JitCompile default value size failed, use PTA default value: True");
    return;
  }
  TORCH_CHECK(
      opt_size.value() != 0,
      "AclGetCompileoptSize opt_size.value() = 0 !",
      PTA_ERROR(ErrCode::ACL));
  char value_name[opt_size.value()];
  auto ret = at_npu::native::AclGetCompileopt(
      ACL_OP_JIT_COMPILE, value_name, opt_size.value());
  // Get func success but get value failed, throw error
  TORCH_CHECK(
      ret == ACL_SUCCESS,
      "Get ACL JitCompile default value failed.",
      PTA_ERROR(ErrCode::ACL));
  std::string value_str(value_name);
  c10_npu::option::SetOption("jitCompile", value_str);
  ASCEND_LOGI("Get ACL JitCompile default value %s and set", value_str.c_str());
}

void SetHF32DefaultValue() {
  // The default value of the flag used to control whether HF32 is allowed on
  // conv is True. The default value of the flag used to control whether HF32 is
  // allowed on matmul is True, but this flag defaults to False in PyTorch 1.12
  // and later.

  // When the flag of matmul is False, and the flag of conv is True,
  // the value of option "ACL_ALLOW_HF32" should be set to "10";
  std::string allow_hf32 = "10";
  auto ret = at_npu::native::AclSetCompileopt(
      aclCompileOpt::ACL_ALLOW_HF32, allow_hf32.c_str());
  if (ret == ACL_SUCCESS) {
    ASCEND_LOGI(
        "Set ACL option ACL_ALLOW_HF32 default value to %s.",
        allow_hf32.c_str());
  } else if (ret == ACL_ERROR_INTERNAL_ERROR) {
    // Used to solve version compatibility issues, when ASCEND have not been
    // updated.
    ASCEND_LOGW(
        "Failed to set default value of ACL option ACL_ALLOW_HF32, which is unsupported by current version.");
  } else {
    TORCH_CHECK(
        0,
        "Failed to set compile option ACL_ALLOW_HF32, result = ",
        ret,
        ", set value ",
        allow_hf32,
        PTA_ERROR(ErrCode::ACL));
  }
}

std::string RealPath(const std::string& path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return "";
  }
  char realPath[PATH_MAX] = {0};
  if (realpath(path.c_str(), realPath) == nullptr) {
    return "";
  }
  return std::string(realPath);
}

std::string GetAclConfigJsonPath() {
  return "";
}
} // namespace

namespace c10_npu {

NpuSysCtrl::NpuSysCtrl()
    : repeat_init_acl_flag_(true), init_flag_(false), device_id_(0) {}

// Get NpuSysCtrl singleton instance
NpuSysCtrl& NpuSysCtrl::GetInstance() {
  static NpuSysCtrl instance;
  return instance;
}

void initAllocator() {
  static c10_backend::CachingAllocator::NPUCachingAllocatorHelper helper;
  c10_backend::CachingAllocator::registerHelper(&helper);
  const auto num_devices = c10_npu::device_count_ensure_non_zero();
  c10_backend::CachingAllocator::init(num_devices);

  c10_backend::CachingAllocator::NPU::init(
      c10_backend::CachingAllocator::get());
  ASCEND_LOGD("Npu caching allocator initialize successfully");
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
  std::string json_path = GetAclConfigJsonPath();
  const char* json_path_ptr = json_path == "" ? nullptr : json_path.c_str();
  ASCEND_LOGD("get acl json path:%s.", json_path_ptr);
  auto init_ret = aclInit(json_path_ptr);
  // The aclinit function is an important low-level aclInit interface that can
  // only be called once in PTA. However, because of the different business
  // codes, aclinit may be successfully invoked by other frameworks or
  // components. ACL_ERROR_REPEAT_INITIALIZE means that aclInit is not called by
  // PTA, so we save the flag variable to control aclFinalize.
  if (init_ret == ACL_ERROR_REPEAT_INITIALIZE) {
    repeat_init_acl_flag_ = false;
    ASCEND_LOGI("acl has allready init by other component.");
  } else if (init_ret != ACL_ERROR_NONE) {
    NPU_CHECK_ERROR(init_ret, "aclInit");
  }

  if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
    NPU_CHECK_ERROR(aclmdlInitDump());
    ASCEND_LOGD("dump init success");
  }

  initAllocator();

  // There's no need to call c10_npu::GetDevice at the start of the process,
  // because device 0 may not be needed
  auto ret = aclrtGetDevice(&device_id_);
  if (ret != ACL_ERROR_NONE) {
    device_id_ = (device_id == -1) ? 0 : device_id;
    NPU_CHECK_ERROR(c10_npu::SetDevice(device_id_));
  } else {
    ASCEND_LOGW("Npu device %d has been set before global init.", device_id_);
  }

  if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
    const char* aclConfigPath = "acl.json";
    NPU_CHECK_ERROR(aclmdlSetDump(aclConfigPath));
    ASCEND_LOGD("set dump config success");
  }

  auto soc_name = c10_npu::acl::AclGetSocName();
  // set global soc name
  c10_npu::SetSocVersion(soc_name);

  if (c10_npu::IsSupportInfNan()) {
    c10_npu::acl::AclrtSetDeviceSatMode(
        aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_INFNAN);
  } else {
    c10_npu::acl::AclrtSetDeviceSatMode(
        aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION);
  }

  // set ACL_PRECISION_MODE by SocVersion("allow_fp32_to_fp16" or
  // "must_keep_origin_dtype").
  auto precision_mode =
      c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1
      ? "must_keep_origin_dtype"
      : "allow_fp32_to_fp16";
  NPU_CHECK_ERROR(at_npu::native::AclSetCompileopt(
      aclCompileOpt::ACL_PRECISION_MODE, precision_mode));

  // set default compile cache mode and dir for users to improve op compile time
  MakeCompileCacheDirAndSetOption();
  // set default jit_Compile value from Get acl defalut value
  GetAndSetDefaultJitCompileByAcl();

  SetHF32DefaultValue();

  NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(
      aclSysParamOpt::ACL_OPT_DETERMINISTIC, 0));
  NPU_CHECK_SUPPORTED_OR_ERROR(
      c10_npu::acl::AclrtSetOpExecuteTimeOut(kMaxOpExecuteTimeOut));
  init_flag_ = true;
  ASCEND_LOGD("Npu sys ctrl initialize successfully.");

  return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::ExchangeDevice(int pre_device, int device) {
  NPU_CHECK_ERROR(c10_npu::SetDevice(device));
  device_id_ = device;
  return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::BackwardsInit() {
  NPU_CHECK_ERROR(c10_npu::SetDevice(device_id_));
  return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::OverflowSwitchEnable() {
  if (!c10_npu::IsSupportInfNan()) {
    c10_npu::acl::AclrtSetStreamOverflowSwitch(
        c10_npu::getCurrentNPUStream(), 1);
    ASCEND_LOGI("Npu overflow check switch set successfully.");
  }
  return INIT_SUCC;
}

// Environment Finalize, return SysStatus
NpuSysCtrl::SysStatus NpuSysCtrl::Finalize() {
  if (!init_flag_) {
    return FINALIZE_SUCC;
  }

  this->RegisterReleaseFn(
      [=]() -> void {
        c10_npu::NPUEventManager::GetInstance().ClearEvent();
        NPU_CHECK_WARN(c10_npu::DestroyUsedStreams());
        NPU_CHECK_WARN(c10_npu::ResetUsedDevices());
        // Maintain a basic point of view, who applies for the resource, the
        // resource is released by whom. If aclInit is not a PTA call, then
        // aclFinalize should not be a PTA call either.
        if (repeat_init_acl_flag_) {
          NPU_CHECK_WARN(aclFinalize());
        }
      },
      ReleasePriority::PriorityLast);

  init_flag_ = false;

  if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
    NPU_CHECK_WARN(aclmdlFinalizeDump());
  }

  // call release fn by priotity
  for (const auto& iter : release_fn_) {
    const auto& fn_vec = iter.second;
    for (const auto& fn : fn_vec) {
      fn();
    }
  }
  release_fn_.clear();

  ASCEND_LOGD("Npu sys ctrl finalize successfully.");
  return FINALIZE_SUCC;
}

bool NpuSysCtrl::GetInitFlag() {
  return init_flag_;
}

int NpuSysCtrl::InitializedDeviceID() {
  if (GetInitFlag()) {
    return device_id_;
  }
  TORCH_CHECK(
      false,
      "no npu device has been initialized!",
      PTA_ERROR(ErrCode::INTERNAL));
  return -1;
}

void NpuSysCtrl::RegisterReleaseFn(
    ReleaseFn release_fn,
    ReleasePriority priority) {
  const auto& iter = this->release_fn_.find(priority);
  if (iter != release_fn_.end()) {
    release_fn_[priority].emplace_back(release_fn);
  } else {
    release_fn_[priority] = (std::vector<ReleaseFn>({release_fn}));
  }
}

aclError SetCurrentDevice() {
  if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
    c10_npu::SetDevice(
        c10_npu::NpuSysCtrl::GetInstance().InitializedDeviceID());
    return ACL_SUCCESS;
  }
  TORCH_CHECK(
      false, "npu device has not been inited.", PTA_ERROR(ErrCode::INTERNAL));
}

} // namespace c10_npu
