/**
 * @file device_adapter.h
 * @brief Defines the DEVICE_NAMESPACE macro and structures for device-related
 * functionality.
 *
 * This header file contains the definition for the DEVICE_NAMESPACE macro and
 * basic type definitions for device functions. The DEVICE_NAMESPACE macro
 * is intended to be replaced at compile time with a specific namespace
 * corresponding to the desired device implementation. This allows for flexible
 * routing to different implementations of device-related functions at compile
 * time.
 *
 * @note The DEVICE_NAMESPACE macro should be defined via a compile option,
 * such as `-DDEVICE_NAMESPACE=acl_adapter`, where `acl_adapter` is the
 * namespace containing the specific device implementation. This compile-time
 * substitution enables routing to the appropriate device code.
 */

#pragma once

#include <c10/core/Device.h>
#include <vector>

#ifndef DEVICE_NAMESPACE
#define DEVICE_NAMESPACE
#endif

typedef int deviceError_t;
typedef void* DeviceContext;

namespace DEVICE_NAMESPACE {

deviceError_t Init();

deviceError_t Finalize();

deviceError_t GetDevice(int32_t* deviceId);

deviceError_t SetDevice(int32_t deviceId);

bool hasPrimaryContext(c10::DeviceIndex device_index);

DeviceContext GetDeviceContext(c10::DeviceIndex device);

deviceError_t ResetUsedDevices();

std::vector<c10::DeviceIndex> GetUsedDevices();

void SynchronizeAllDevice();

void SynchronizeDevice();

void CreateStream(aclrtStream* stream, uint32_t priority, uint32_t configFlag);

deviceError_t GetDeviceCount(uint32_t* dev_count);

} // namespace DEVICE_NAMESPACE