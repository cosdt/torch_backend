#pragma once

/*
  Provides an adapter for CANN API to offset the differences between it and
  CUDA API. For example:

        aclError aclrtGetDevice(int32_t* deviceId)

  If aclrtSetDevice or aclrtCreateContext is not called to specify the device,
  an error is returned. In this adapter API, it will set the device to 0 first
  in this case.
*/

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <npu/acl/include/acl/acl.h>

namespace acl_adapter {

aclError aclrtGetDevice(int32_t* deviceId);

aclError aclrtSetDevice(int32_t deviceId);

bool hasPrimaryContext(c10::DeviceIndex device_index);

aclrtContext GetDeviceContext(c10::DeviceIndex device);

aclError ResetUsedDevices();

std::vector<c10::DeviceIndex> GetUsedDevices();

void synchronize_all_device();

} // namespace acl_adapter
