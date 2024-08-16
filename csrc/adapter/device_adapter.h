#pragma once

#include <c10/core/Device.h>
#include <vector>
#include "acl/include/acl/acl.h"

#ifndef DEVICE_NAMESPACE
#define DEVICE_NAMESPACE
#endif

typedef int DeviceError;
typedef void* DeviceContext;

namespace DEVICE_NAMESPACE {

DeviceError aclrtGetDevice(int32_t* deviceId);

DeviceError setDevice(int32_t deviceId);

bool hasPrimaryContext(c10::DeviceIndex device_index);

DeviceContext GetDeviceContext(c10::DeviceIndex device);

DeviceError ResetUsedDevices();

std::vector<c10::DeviceIndex> GetUsedDevices();

void synchronize_all_device();

void initSingleStream(int p, c10::DeviceIndex device_index, int i);

} // namespace DEVICE_NAMESPACE