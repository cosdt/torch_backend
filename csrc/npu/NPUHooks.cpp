#include "csrc/npu/NPUHooks.h"
#include "csrc/npu/NPUCachingHostAllocator.h"
#include "csrc/npu/NPUFunctions.h"
#include "csrc/npu/NPUStorageImpl.h"
#include "npu/aten/common/ResizeNpu.h"
#include "npu/framework/FormatHelper.h"
#include "csrc/core/Register.h"

namespace c10_npu {

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, NPUHooks, NPUHooksArgs);

AT_REGISTER_PRIVATEUSE1_HOOKS_INTERFACE(c10_npu::get_npu_hooks());

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, NPUHooks, NPUHooksArgs)

void NPUHooks::initPrivateUse1() const {
  c10_npu::NpuSysCtrl::SysStatus status =
      c10_npu::NpuSysCtrl::GetInstance().Initialize();
  TORCH_CHECK(
      status == c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC,
      "Device init failed, status:",
      status,
      PTA_ERROR(ErrCode::INTERNAL));
}

bool NPUHooks::hasPrimaryContext(c10::DeviceIndex device_index) const {
  return c10_npu::hasPrimaryContext(device_index);
}

void NPUHooks::resizePrivateUse1Bytes(
    const c10::Storage& storage,
    size_t new_bytes) const {
  auto storage_impl =
      static_cast<torch_npu::NPUStorageImpl*>(storage.unsafeGetStorageImpl());
  auto format = storage_impl->npu_desc_.npu_format_;
  if (!at_npu::native::FormatHelper::IsBaseFormatType(format)) {
    AT_ERROR("Try to resize a storage without base format");
  }

  auto itemsize = storage_impl->npu_desc_.data_type_.itemsize();
  std::vector<int64_t> new_size = {new_bytes / (ptrdiff_t)itemsize};
  at_npu::native::storage_resize_npu(*storage_impl, new_bytes, new_size);
}

bool NPUHooks::isPinnedPtr(const void* data) const {
  return NPUCachingHostAllocator_isPinndPtr(data);
}

at::Allocator* NPUHooks::getPinnedMemoryAllocator() const {
  return getNPUCachingHostAllocator();
}

at::PrivateUse1HooksInterface* get_npu_hooks() {
  static at::PrivateUse1HooksInterface* npu_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] { npu_hooks = new NPUHooks(); });
  return npu_hooks;
}
} // namespace c10_npu
