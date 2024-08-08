#include "csrc/aten/generated/NPUNativeFunctions.h"
#include "core/NPUBridge.h"
#include "framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

int64_t NPUNativeFunctions::get_storage_size(const at::Tensor& self) {
  torch_backend::utils::torch_check_npu(self);
  auto sizes = c10::backend::NPUBridge::GetNpuStorageImpl(self)
                   ->npu_desc_.storage_sizes_;
  int64_t n = 1;
  for (auto s : sizes) {
    n *= s;
  }
  return n;
}

} // namespace native
} // namespace at_npu
