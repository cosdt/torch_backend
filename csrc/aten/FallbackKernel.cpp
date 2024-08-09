#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

namespace at::native::backend {
static void autograd_fallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  // PyTorch has separate builds, some of which don't include autograd.
  // So we define some behavior for when autograd isn't included and
  // go through a layer of indirection (VariableHooksInterface) when it is.
  // See aten/src/ATen/core/VariableHooksInterface.h for more details.
  if (!at::impl::HasVariableHooks()) {
    op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
    return;
  }

  at::impl::GetVariableHooks()->basic_autograd_not_implemented_fallback(
      op, dispatch_keys, stack);
}

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&autograd_fallback>());
}

static void cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  TORCH_CHECK(
      false,
      "CAUTION: The operator '",
      op.schema().operator_name(),
      "' is not currently supported on the current backend",
      " and will fallback to CPU");

  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}
} // namespace at::native::backend
