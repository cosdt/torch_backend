#pragma once

#include <ATen/Tensor.h>
#include <string>
#include <unordered_map>

namespace c10::backend {
// Serialize device-related information, mainly related to private formats
void device_info_serialization(
    const at::Tensor& t,
    std::unordered_map<std::string, bool>& mate_map);
// Deserialize device-related information, mainly related to private formats
void device_info_deserialization(
    const at::Tensor& t,
    std::unordered_map<std::string, bool>& mate_map);
} // namespace c10::backend
