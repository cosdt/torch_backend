#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/MetaFunctions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Half.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/library.h>

#include "csrc/aten/generated/NPUNativeFunctions.h"
#include "csrc/aten/generated/VariableType.h"
#include "core/NPUException.h"
#include "aten/OpInterface.h"

namespace at_npu {

namespace native {

${custom_op_definitions}

namespace {

TORCH_LIBRARY(npu, m) {
  ${custom_schema_registrations}
}

} // anonymous namespace

namespace {

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  ${custom_impl_registrations}
}

} // anonymous namespace

} // namespace native

} // namespace at_npu
