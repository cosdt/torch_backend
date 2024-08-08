#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>

#include "csrc/aten/generated/NPUNativeFunctions.h"
#include "aten/common/ResizeNpu.h"
#include "core/NPUBridge.h"
#include "csrc/backend/NPUStorageImpl.h"
#include "csrc/backend/NPUCachingAllocator.h"
#include "framework/StorageDescHelper.h"

namespace at_npu {
namespace native {

at::Tensor set_tensor_with_storage_format(c10::Storage src);

} // namespace native
} // namespace at_npu
