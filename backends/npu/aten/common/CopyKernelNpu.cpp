#include <ATen/record_function.h>
#include <c10/core/Scalar.h>

#include "aten/OpInterface.h"
#include "aten/common/InnerNpuNativeFunction.h"
#include "core/NPUBridge.h"
#include "core/interface/AsyncTaskQueueInterface.h"
#include "csrc/backend/StorageImpl.h"
#include "csrc/backend/Stream.h"
#include "framework/StorageDescHelper.h"
#include "framework/utils/CalcuOpUtil.h"
#include "framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

// the dst and src are same dtype
// the dst and src have same elemsize
// if exceptCopySize is not defined, we will copy dst storage size
// so: caller should make sure that the storage size of src and dst are
// reasonable.
void copy_d2d_by_memcpy(
    at::Tensor& dst,
    const at::Tensor& src,
    int64_t exceptSize) {
  c10::DeviceGuard guard(src.device());
  int64_t size = exceptSize;
  auto dst_mem_size = StorageDescHelper::GetMemorySize(dst);
  if (exceptSize == 0) {
    size = dst_mem_size;
  }

  if (!dst.data_ptr()) {
    TORCH_NPU_WARN("copy_d2d_by_memcpy, dst.data_ptr() is null.");
    return;
  }

  if (!src.data_ptr()) {
    TORCH_NPU_WARN("copy_d2d_by_memcpy, src.data_ptr() is null.");
    return;
  }

  if (dst.data_ptr() == src.data_ptr() &&
      dst.element_size() == src.element_size()) {
    return;
  }

  // The current logic is only used in single op mode.
  aclError error = c10::npu::queue::LaunchAsyncCopyTask(
      dst.data_ptr(),
      size * dst.element_size(),
      src.data_ptr(),
      size * dst.element_size(),
      ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (error != ACL_ERROR_NONE) {
    AT_ERROR("async copy device to device error.");
    return;
  }
}
} // namespace native
} // namespace at_npu
