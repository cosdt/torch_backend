#include "csrc/backend/Hooks.h"
#include "csrc/backend/CachingHostAllocator.h"
#include "csrc/backend/Functions.h"
#include "csrc/backend/StorageImpl.h"
#include "csrc/core/Register.h"

// TODO(FFFrog):
// Remove later
#include "aten/common/ResizeNpu.h"
#include "framework/FormatHelper.h"

namespace c10::backend {

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, Hooks, HooksArgs);

AT_REGISTER_PRIVATEUSE1_HOOKS_INTERFACE(c10::backend::get_device_hooks());

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, Hooks, HooksArgs)

void Hooks::initPrivateUse1() const {
  c10::npu::TryInitDevice();
}

bool Hooks::hasPrimaryContext(c10::DeviceIndex device_index) const {
  return hasPrimaryContext(device_index);
}

void Hooks::resizePrivateUse1Bytes(
    const c10::Storage& storage,
    size_t new_bytes) const {
  auto storage_impl = static_cast<c10::backend::DeviceStorageImpl*>(
      storage.unsafeGetStorageImpl());
  auto format = storage_impl->storage_desc_.npu_format_;
  if (!at_npu::native::FormatHelper::IsBaseFormatType(format)) {
    AT_ERROR("Try to resize a storage without base format");
  }

  auto itemsize = storage_impl->storage_desc_.data_type_.itemsize();
  std::vector<int64_t> new_size = {new_bytes / (ptrdiff_t)itemsize};
  at_npu::native::storage_resize_npu(*storage_impl, new_bytes, new_size);
}

bool Hooks::isPinnedPtr(const void* data) const {
  return c10::backend::HostAllocator::isPinndPtr(data);
}

at::Allocator* Hooks::getPinnedMemoryAllocator() const {
  return c10::backend::HostAllocator::getAllocator();
}

at::PrivateUse1HooksInterface* get_device_hooks() {
  static at::PrivateUse1HooksInterface* device_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] { device_hooks = new Hooks(); });
  return device_hooks;
}
} // namespace c10::backend
