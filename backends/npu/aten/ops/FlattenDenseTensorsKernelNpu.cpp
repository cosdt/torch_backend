#include "csrc/aten/generated/NPUNativeFunctions.h"
#include "framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::flatten_dense_tensors(at::TensorList tensors) {
  static auto cast_back_to_ori_format = [](const at::Tensor& t) {
    return custom_ops::npu_format_cast(
        t,
        c10::backend::NPUBridge::GetNpuStorageImpl(t)
            ->storage_desc_.origin_format_);
  };
  static auto flatten = [](const at::Tensor& t) {
    return cast_back_to_ori_format(t).contiguous().view({-1});
  };
  if (tensors.size() == 1) {
    return flatten(tensors[0]);
  }
  return at::cat(c10::fmap(tensors, flatten));
}

} // namespace native
} // namespace at_npu
