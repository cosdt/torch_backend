#include "csrc/backend/NPUCachingAllocator.h"
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <iostream>
#include "csrc/core/allocator/CachingAllocator.h"
#include "csrc/backend/NPUFunctions.h"
#include "csrc/backend/NPUStream.h"
#include "npu/acl/include/acl/acl_base.h"
#include "npu/acl/include/acl/acl_rt.h"

namespace c10::npu::NPUCachingAllocator {
class DefaultNPUAllocator final : public NPUAllocator {
 public:
  void init(CachingAllocator* delegate) {
    this->delegate = delegate;
  }

  c10::DataPtr allocate(size_t size) override {
    return delegate->allocate(size);
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    delegate->copy_data(dest, src, count);
  }

  void* raw_alloc_with_stream(size_t nbytes, void* stream) override {
    return delegate->raw_alloc_with_stream(nbytes, stream);
  }
  void raw_delete(void* ptr) override {
    delegate->raw_delete(ptr);
  }
  void setMemoryFraction(double fraction, int device) override {
    delegate->setMemoryFraction(fraction, device);
  }
  void emptyCache(bool check_error) override {
    delegate->emptyCache(check_error);
  }
  void recordStream(const c10::DataPtr& ptr, c10::Stream stream) override {
    delegate->recordStream(ptr, stream);
  }
  DeviceStats getDeviceStats(int device) override {
    return delegate->getDeviceStats(device);
  }
  void resetAccumulatedStats(int device) override {
    delegate->resetAccumulatedStats(device);
  }
  void resetPeakStats(int device) override {
    delegate->resetPeakStats(device);
  }
  SnapshotInfo snapshot() override {
    return delegate->snapshot();
  }
  void emptyDeviceCache(int device) override {
    delegate->emptyDeviceCache(device);
  }
  std::string name() override {
    return "DefaultNPUAllocator";
  }
  bool isHistoryEnabled() override {
    return delegate->isHistoryEnabled();
  }
  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) override {
    delegate->recordHistory(
        enabled, context_recorder, alloc_trace_max_entries, when);
  }
  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    delegate->attachOutOfMemoryObserver(observer);
  }

 private:
  CachingAllocator* delegate;
};

class CachingAllocatorHelper
    : public c10::backend::CachingAllocator::CachingAllocatorHelper {
 public:
  void insertEventWrapper(c10::DeviceIndex device, std::function<void()> fn)
      override {
    aclrtContext compiler_ctx = aclrtContext();
    aclError ret_ctx = aclrtGetCurrentContext(&compiler_ctx);
    aclrtSetCurrentContext(c10::npu::GetDeviceContext(device));
    fn();
    if (ret_ctx == ACL_ERROR_NONE) {
      aclrtSetCurrentContext(compiler_ctx);
    }
  }

  void* getCurrentStream(c10::DeviceIndex device_index) override {
    return c10::backend::getCurrentNPUStream(device_index);
  }

  int synchronizeStream(void* stream) override {
    return aclrtSynchronizeStream(stream);
  }

  virtual void deviceSynchronize() override {
    c10::npu::device_synchronize();
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
    return aclrtReleaseMemAddress(ptr);
  }

  int memAddressReserve(
      void** virPtr,
      size_t size,
      size_t alignment,
      void* expectPtr,
      uint64_t flags) override {
    return aclrtReserveMemAddress(virPtr, size, alignment, expectPtr, flags);
  }

  int memAddressReserve(void** ptr, size_t size, size_t alignment, void* addr)
      override {
    return aclrtReserveMemAddress(ptr, size, alignment, addr, 1);
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
    int status = aclrtMallocPhysical(handle, size, &prop, flags);
    return status == ACL_ERROR_RT_MEMORY_ALLOCATION ? MEM_ALLOCATION_ERROR
                                                    : status;
  }

  int memRelease(void* handle) override {
    return aclrtFreePhysical(handle);
  }

  int memMap(
      void* ptr,
      size_t size,
      size_t offset,
      void* handle,
      uint64_t flags) override {
    return aclrtMapMem(ptr, size, offset, handle, flags);
  }

  int memSetAccess(void* ptr, size_t size, int device) override {
    return MEM_SUCCESS;
  }

  int memUnmap(void* ptr, size_t size) override {
    return aclrtUnmapMem(ptr);
  }
};

static DefaultNPUAllocator defaultNPUAllocator;
std::atomic<NPUAllocator*> npu_allocator = &defaultNPUAllocator;

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &defaultNPUAllocator);

void init(CachingAllocator* delegate) {
  static c10::npu::NPUCachingAllocator::CachingAllocatorHelper helper;
  c10::backend::CachingAllocator::registerHelper(&helper);
  c10::backend::CachingAllocator::init(c10::npu::device_count_ensure_non_zero());

  defaultNPUAllocator.init(delegate);
}
} // namespace c10::npu::NPUCachingAllocator
