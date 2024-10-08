#include "csrc/backend/StorageImpl.h"
#include "csrc/core/Register.h"

namespace c10::backend {

C10_SET_STORAGE_IMPL_CREATE(&c10::backend::make_device_storage_impl);

DeviceStorageImpl::DeviceStorageImpl(
    use_byte_size_t use_byte_size,
    size_t size_bytes,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable)
    : c10::StorageImpl(
          use_byte_size,
          size_bytes,
          at::DataPtr(std::move(data_ptr)),
          allocator,
          resizable) {}

void DeviceStorageImpl::release_resources() {
  StorageImpl::release_resources();
}

c10::intrusive_ptr<c10::StorageImpl> make_device_storage_impl(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable) {
  if (data_ptr == nullptr) {
    data_ptr = allocator->allocate(size_bytes.as_int_unchecked());
  }
  // Correctly create DeviceStorageImpl object.
  c10::intrusive_ptr<c10::StorageImpl> npu_storage_impl =
      c10::make_intrusive<DeviceStorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size_bytes.as_int_unchecked(),
          std::move(data_ptr),
          allocator,
          resizable);
  // There is no need to consider the StorageDesc information, it will be
  // carried out in the subsequent processing.
  return npu_storage_impl;
}

} // namespace c10::backend
