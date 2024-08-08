#pragma once
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include "npu/core/NpuDeviceRAII.h"

namespace torch_backend {
namespace utils {

inline bool is_npu(const at::Tensor& tensor) {
  return tensor.is_privateuseone();
}

inline bool is_npu(const at::TensorOptions& options) {
  return options.device().is_privateuseone();
}

inline bool is_npu(const at::Device& device) {
  return device.is_privateuseone();
}

inline void torch_check_npu(const at::Tensor& tensor) {
  TORCH_CHECK(
      is_npu(tensor),
      "Expected NPU tensor, please check whether the input tensor device is correct.",
      PTA_ERROR(ErrCode::PARAM));
}

inline void torch_check_npu(const at::TensorOptions& options) {
  TORCH_CHECK(
      is_npu(options),
      "Expected NPU tensor, please check whether the input tensor device is correct.",
      PTA_ERROR(ErrCode::PARAM));
}

inline void torch_check_npu(const at::Device& device) {
  TORCH_CHECK(
      is_npu(device),
      "Expected NPU tensor, please check whether the input tensor device is correct.",
      PTA_ERROR(ErrCode::PARAM));
}

inline c10::DeviceType get_npu_device_type() {
  return c10::DeviceType::PrivateUse1;
}

} // namespace utils
} // namespace torch_backend
