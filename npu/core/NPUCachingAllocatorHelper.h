#pragma once

#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include "csrc/core/CachingAllocatorHelper.h"
#include "csrc/npu/NPUFunctions.h"
#include "csrc/npu/NPUStream.h"
#include "npu/acl/include/acl/acl_base.h"
#include "npu/acl/include/acl/acl_rt.h"
#include "npu/core/npu_log.h"

namespace c10_npu::NPUCachingAllocator {

class CachingAllocatorHelper
    : public c10_backend::CachingAllocator::CachingAllocatorHelper {
 public:
  void insertEventWrapper(c10::DeviceIndex device, std::function<void()> fn)
      override {
    aclrtContext compiler_ctx = aclrtContext();
    aclError ret_ctx = aclrtGetCurrentContext(&compiler_ctx);
    NPU_CHECK_ERROR(aclrtSetCurrentContext(c10_npu::GetDeviceContext(device)));
    fn();
    if (ret_ctx == ACL_ERROR_NONE) {
      NPU_CHECK_ERROR(aclrtSetCurrentContext(compiler_ctx));
    }
  }

  void* getCurrentStream(c10::DeviceIndex device_index) override {
    return c10_npu::getCurrentNPUStream(device_index);
  }

  int synchronizeStream(void* stream) override {
    return aclrtSynchronizeStream(stream);
  }

  virtual void deviceSynchronize() override {
    c10_npu::synchronize_device();
  }

  int memFree(void* devPtr) override {
    return aclrtFree(devPtr);
  }

  int memAlloc(void** devPtr, size_t size) override {
    return aclrtMalloc(
        devPtr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
  }

  int memGetInfo(size_t* free, size_t* total) override {
    return aclrtGetMemInfo(ACL_HBM_MEM, free, total);
  }

  int memAddressFree(void* ptr, size_t size) override {
    return c10_npu::acl::AclrtReleaseMemAddress(ptr);
  }

  int memAddressReserve(
      void** virPtr,
      size_t size,
      size_t alignment,
      void* expectPtr,
      uint64_t flags) override {
    return c10_npu::acl::AclrtReserveMemAddress(
        virPtr, size, alignment, expectPtr, flags);
  }

  int memAddressReserve(void** ptr, size_t size, size_t alignment, void* addr)
      override {
    return c10_npu::acl::AclrtReserveMemAddress(ptr, size, alignment, addr, 1);
  }

  int memCreate(void** handle, size_t size, int device, uint64_t flags)
      override {
    aclrtPhysicalMemProp prop = {};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.reserve = 0;
    int status = c10_npu::acl::AclrtMallocPhysical(handle, size, &prop, flags);
    return status == ACL_ERROR_RT_MEMORY_ALLOCATION ? MEM_ALLOCATION_ERROR
                                                    : status;
  }

  int memRelease(void* handle) override {
    return c10_npu::acl::AclrtFreePhysical(handle);
  }

  int memMap(
      void* ptr,
      size_t size,
      size_t offset,
      void* handle,
      uint64_t flags) override {
    return c10_npu::acl::AclrtMapMem(ptr, size, offset, handle, flags);
  }

  int memSetAccess(void* ptr, size_t size, int device) override {
    return MEM_SUCCESS;
  }

  int memUnmap(void* ptr, size_t size) override {
    return c10_npu::acl::AclrtUnmapMem(ptr);
  }
};
} // namespace c10_npu::NPUCachingAllocator
