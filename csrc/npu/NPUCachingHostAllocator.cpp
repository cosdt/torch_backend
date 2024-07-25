#include <c10/core/DeviceGuard.h>
#include <c10/util/Logging.h>
#include "csrc/core/EventPool.h"
#include "csrc/npu/NPUFunctions.h"
#include "npu/core/interface/AclInterface.h"
#include "npu/core/interface/AsyncTaskQueueInterface.h"
#include "npu/core/npu_log.h"
#include "npu/core/register/OptionsManager.h"
#include "npu/core/sys_ctrl/npu_sys_ctrl.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <ATen/core/CachingHostAllocator.h>
#include "csrc/npu/NPUEvent.h"
#include "csrc/npu/NPUCachingHostAllocator.h"

namespace c10_npu {
using Block = at::HostBlock<NPUStream>;
struct HostAllocator : public at::CachingHostAllocatorImpl<
                           NPUStream,
                           c10_backend::EventPool<NPUEvent>::Event> {
 public:
  bool isPinndPtr(void* ptr) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return pinned_ptrs.find(ptr) != pinned_ptrs.end();
  }

 private:
  void allocate_host_memory(size_t size, void** ptr) override {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    // for pin_memory in dataloader, it should be set device first when new a
    // thread
    SetCurrentDevice();
    // allocate a new block if no cached allocation is found
    NPU_CHECK_ERROR(aclrtMallocHost(ptr, size));
    pinned_ptrs.insert(*ptr);
  }

  void free_block(Block* block) override {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    NPU_CHECK_ERROR(aclrtFreeHost(block->ptr_));
    pinned_ptrs.erase(block->ptr_);
  }

  void record_stream(
      std::optional<std::vector<c10_backend::EventPool<NPUEvent>::Event>>&
          events,
      NPUStream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(c10_backend::EventPool<NPUEvent>::Event& event) override {
    return event->query();
  }

  c10_backend::EventPool<NPUEvent>::Event create_event_internal(
      c10::DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool =
        new c10_backend::EventPool<NPUEvent>(device_count(), []() {
          return std::make_unique<NPUEvent>(ACL_EVENT_CAPTURE_STREAM_PROGRESS);
        });
    return event_pool->get(idx);
  }

  std::shared_mutex mutex_{};
  std::set<void*> pinned_ptrs{};
};
} // namespace c10_npu

void raw_local_deleter(void* ptr);

struct NPUCachingHostAllocator final
    : public at::CachingHostAllocatorInterface<c10_npu::HostAllocator> {
  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }

  bool isPinnedPtr(void* ptr) {
    return impl_->isPinndPtr(ptr);
  }
};

static NPUCachingHostAllocator npu_caching_host_allocator;

aclError NPUCachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10_npu::NPUStream stream) {
  return npu_caching_host_allocator.record_event(ptr, ctx, stream);
}

bool NPUCachingHostAllocator_isPinndPtr(void* ptr) {
  return npu_caching_host_allocator.isPinnedPtr(ptr);
}

void NPUCachingHostAllocator_emptyCache() {
  npu_caching_host_allocator.empty_cache();
}

void raw_local_deleter(void* ptr) {
  npu_caching_host_allocator.free(ptr);
}

at::Allocator* getNPUCachingHostAllocator() {
  return &npu_caching_host_allocator;
}

c10::Allocator* getPinnedMemoryAllocator() {
  C10_LOG_API_USAGE_ONCE("aten.init.npu");
  if (!c10_npu::NpuSysCtrl::IsInitializeSuccess()) {
    ASCEND_LOGE("Npu init fail.");
  }
  return getNPUCachingHostAllocator();
}
