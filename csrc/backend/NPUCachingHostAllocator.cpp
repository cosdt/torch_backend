#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Logging.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "csrc/backend/NPUCachingHostAllocator.h"
#include "csrc/backend/NPUEvent.h"
#include "csrc/backend/NPUFunctions.h"
#include "csrc/core/allocator/EventPool.h"

namespace c10::npu {

using namespace c10::backend;

using Block = at::HostBlock<NPUStream>;
struct HostAllocator
    : public at::CachingHostAllocatorImpl<
          NPUStream,
          c10::backend::CachingAllocator::EventPool<NPUEvent>::Event> {
 public:
  bool isPinndPtr(const void* ptr) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return pinned_ptrs.find(ptr) != pinned_ptrs.end();
  }

 private:
  void allocate_host_memory(size_t size, void** ptr) override {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // TODO(FFFrog): implement aclrtMallocHost which don`t need explicitly
    // to create context
    c10::backend::current_device();
    NPU_CHECK_ERROR(aclrtMallocHost(ptr, size));
    pinned_ptrs.insert(*ptr);
  }

  void free_block(Block* block) override {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    NPU_CHECK_ERROR(aclrtFreeHost(block->ptr_));
    pinned_ptrs.erase(block->ptr_);
  }

  void record_stream(
      std::optional<std::vector<
          c10::backend::CachingAllocator::EventPool<NPUEvent>::Event>>& events,
      NPUStream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(c10::backend::CachingAllocator::EventPool<NPUEvent>::Event&
                       event) override {
    return event->query();
  }

  c10::backend::CachingAllocator::EventPool<NPUEvent>::Event
  create_event_internal(c10::DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool =
        new c10::backend::CachingAllocator::EventPool<NPUEvent>(
            device_count(), []() {
              return std::make_unique<NPUEvent>(
                  ACL_EVENT_CAPTURE_STREAM_PROGRESS);
            });
    return event_pool->get(idx);
  }

  std::shared_mutex mutex_{};
  std::set<const void*> pinned_ptrs{};
};
} // namespace c10::npu

void raw_local_deleter(void* ptr);

struct NPUCachingHostAllocator final
    : public at::CachingHostAllocatorInterface<c10::npu::HostAllocator> {
  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }

  bool isPinnedPtr(const void* ptr) {
    return impl_->isPinndPtr(ptr);
  }
};

static NPUCachingHostAllocator npu_caching_host_allocator;

at::Allocator* getNPUCachingHostAllocator() {
  return &npu_caching_host_allocator;
}

void raw_local_deleter(void* ptr) {
  npu_caching_host_allocator.free(ptr);
}

bool NPUCachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::backend::NPUStream stream) {
  return npu_caching_host_allocator.record_event(ptr, ctx, stream);
}

void NPUCachingHostAllocator_emptyCache() {
  npu_caching_host_allocator.empty_cache();
}

bool NPUCachingHostAllocator_isPinndPtr(const void* ptr) {
  return npu_caching_host_allocator.isPinnedPtr(ptr);
}
