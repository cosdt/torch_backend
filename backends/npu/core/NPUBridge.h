#pragma once

#include <c10/core/StorageImpl.h>
#include "csrc/backend/StorageImpl.h"
#include "csrc/backend/TensorImpl.h"

namespace c10::backend {

class NPUBridge {
 public:
  // at::tensor to DeviceStorageImpl
  static DeviceStorageImpl* GetNpuStorageImpl(const at::Tensor& tensor);

  // c10::StorageImpl to DeviceStorageImpl
  static DeviceStorageImpl* GetNpuStorageImpl(c10::StorageImpl* storageImpl);

  // c10::Storage to DeviceStorageImpl
  static DeviceStorageImpl* GetNpuStorageImpl(c10::Storage&& storage);

  // tensor to StorageDesc
  static StorageDesc& GetNpuStorageImplDesc(const at::Tensor& tensor);

  // tensor to TensorImpl
  static TensorImpl* GetNpuTensorImpl(const at::Tensor& tensor);
};

} // namespace c10::backend
