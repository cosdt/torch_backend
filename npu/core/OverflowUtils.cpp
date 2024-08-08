#include <ATen/ATen.h>

#include "npu/aten/OpInterface.h"
#include "npu/core/OverflowUtils.h"
#include "npu/core/NpuDeviceRAII.h"

namespace torch_backend {
namespace utils {

OverflowUtil::OverflowUtil() {}

OverflowUtil::~OverflowUtil() {}

void OverflowUtil::EnableOverflowNpu() {
  return;
}

bool OverflowUtil::CheckOverflowNpu() {
  auto options =
      at::TensorOptions(c10::DeviceType::PrivateUse1).dtype(at::kFloat);
  at::Tensor tmp = at::empty({8}, options);
  auto floatStatus = op_plugin::npu_alloc_float_status(tmp);
  auto result = op_plugin::npu_get_float_status(floatStatus);
  if (result.cpu()[0].item().toInt() != 0) {
    return true;
  }
  return false;
}

void OverflowUtil::ClearOverflowNpu() {
  auto options =
      at::TensorOptions(c10::DeviceType::PrivateUse1).dtype(at::kFloat);
  at::Tensor tmp = at::empty({8}, options);
  auto floatStatus = op_plugin::npu_alloc_float_status(tmp);
  auto result = op_plugin::npu_clear_float_status(floatStatus);
  return;
}

} // namespace utils
} // namespace torch_backend
