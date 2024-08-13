#include "core/NPUBridge.h"

namespace c10::backend {

DeviceStorageImpl* NPUBridge::GetNpuStorageImpl(c10::StorageImpl* storageImpl) {
  return static_cast<DeviceStorageImpl*>(storageImpl);
}

DeviceStorageImpl* NPUBridge::GetNpuStorageImpl(c10::Storage&& storage) {
  return static_cast<DeviceStorageImpl*>(storage.unsafeGetStorageImpl());
}

DeviceStorageImpl* NPUBridge::GetNpuStorageImpl(const at::Tensor& tensor) {
  return static_cast<DeviceStorageImpl*>(
      tensor.storage().unsafeGetStorageImpl());
}

StorageDesc& NPUBridge::GetNpuStorageImplDesc(const at::Tensor& tensor) {
  return static_cast<DeviceStorageImpl*>(
             tensor.storage().unsafeGetStorageImpl())
      ->npu_desc_;
}

TensorImpl* NPUBridge::GetNpuTensorImpl(const at::Tensor& tensor) {
  return static_cast<TensorImpl*>(tensor.unsafeGetTensorImpl());
}

} // namespace c10::backend
