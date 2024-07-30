#include <c10/core/DispatchKey.h>
#include <torch/library.h>

namespace at_npu {
namespace native {

bool _has_compatible_shallow_copy_type(
    const at::Tensor& self,
    const at::Tensor& from) {
  auto is_dense = [](c10::DispatchKeySet ts) {
    constexpr auto dense_backends = c10::DispatchKeySet(
        {c10::BackendComponent::CPUBit, c10::BackendComponent::PrivateUse1Bit});
    constexpr auto dense_k = c10::DispatchKeySet(c10::DispatchKey::Dense);

    return ts.has_any(dense_k) && ts.has_any(dense_backends);
  };

  c10::DispatchKeySet self_key = self.key_set();
  c10::DispatchKeySet from_key = from.key_set();

  return (self_key == from_key) || (is_dense(self_key) && is_dense(from_key));
}

TORCH_LIBRARY_IMPL(aten, CatchAll, m) {
  m.impl(
      "_has_compatible_shallow_copy_type",
      TORCH_FN(_has_compatible_shallow_copy_type));
}

} // namespace native
} // namespace at_npu
