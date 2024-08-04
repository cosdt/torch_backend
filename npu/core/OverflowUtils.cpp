#include <ATen/ATen.h>

#include "npu/core/OverflowUtils.h"
#include "npu/core/sys_ctrl/npu_sys_ctrl.h"
#include "npu/aten/OpInterface.h"

namespace torch_npu {
namespace utils {

OverflowUtil::OverflowUtil() {}

OverflowUtil::~OverflowUtil() {}

void OverflowUtil::EnableOverflowNpu() {
  auto result = c10_npu::NpuSysCtrl::GetInstance().OverflowSwitchEnable();
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
} // namespace torch_npu
