#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <npu/acl/include/acl/acl.h>

namespace acl_adapter {

aclError aclrtGetDevice(int32_t* deviceId);

aclError aclrtSetDevice(int32_t deviceId);

bool hasPrimaryContext(c10::DeviceIndex device_index);

aclrtContext GetDeviceContext(c10::DeviceIndex device);

aclError ResetUsedDevices();

aclError DestroyUsedStreams();

aclError SynchronizeUsedDevices();

} // namespace acl_adapter
