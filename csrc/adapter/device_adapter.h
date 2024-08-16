#pragma once

#include <c10/core/Device.h>
#include <vector>

#ifndef DEVICE_NAMESPACE
#define DEVICE_NAMESPACE
#endif

typedef int DeviceError;
typedef void* DeviceContext;

namespace DEVICE_NAMESPACE {

DeviceError Init();

DeviceError Finalize();

DeviceError GetDevice(int32_t* deviceId);

DeviceError SetDevice(int32_t deviceId);

bool hasPrimaryContext(c10::DeviceIndex device_index);

DeviceContext GetDeviceContext(c10::DeviceIndex device);

DeviceError ResetUsedDevices();

std::vector<c10::DeviceIndex> GetUsedDevices();

void SynchronizeAllDevice();

void SynchronizeDevice();

void CreateStream(aclrtStream* stream, uint32_t priority, uint32_t configFlag);

DeviceError GetDeviceCount(uint32_t* dev_count);

} // namespace DEVICE_NAMESPACE