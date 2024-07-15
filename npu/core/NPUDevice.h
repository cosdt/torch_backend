#pragma once

#include <c10/core/Device.h>
#include <npu/acl/include/acl/acl.h>

namespace c10_npu::acl {

c10::DeviceIndex device_count_impl(uint32_t* count) noexcept;

}
