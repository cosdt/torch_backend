#include <ATen/record_function.h>
#include <c10/core/Scalar.h>

#include "npu/aten/common/InnerNpuNativeFunction.h"
#include "npu/core/NPUBridge.h"
#include "csrc/npu/NPUStorageImpl.h"
#include "csrc/npu/NPUStream.h"
#include "npu/core/interface/AsyncTaskQueueInterface.h"
#include "npu/framework/StorageDescHelper.h"
#include "npu/framework/utils/CalcuOpUtil.h"
#include "npu/framework/utils/OpAdapter.h"
#include "npu/aten/OpInterface.h"

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
  aclError error = c10_npu::queue::LaunchAsyncCopyTask(
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
