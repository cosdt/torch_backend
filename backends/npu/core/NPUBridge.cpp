#include "core/NPUBridge.h"

namespace c10::backend {

NPUStorageImpl* NPUBridge::GetNpuStorageImpl(c10::StorageImpl* storageImpl) {
  return static_cast<NPUStorageImpl*>(storageImpl);
}

NPUStorageImpl* NPUBridge::GetNpuStorageImpl(c10::Storage&& storage) {
  return static_cast<NPUStorageImpl*>(storage.unsafeGetStorageImpl());
}

NPUStorageImpl* NPUBridge::GetNpuStorageImpl(const at::Tensor& tensor) {
  return static_cast<NPUStorageImpl*>(tensor.storage().unsafeGetStorageImpl());
}

NPUStorageDesc& NPUBridge::GetNpuStorageImplDesc(const at::Tensor& tensor) {
  return static_cast<NPUStorageImpl*>(tensor.storage().unsafeGetStorageImpl())
      ->npu_desc_;
}

TensorImpl* NPUBridge::GetNpuTensorImpl(const at::Tensor& tensor) {
  return static_cast<TensorImpl*>(tensor.unsafeGetTensorImpl());
}

} // namespace c10::backend
