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

#include "csrc/backend/Event.h"
#include "csrc/backend/Functions.h"
#include "csrc/backend/NPUCachingHostAllocator.h"
#include "csrc/core/allocator/EventPool.h"

namespace c10::backend::HostAllocator {

using Block = at::HostBlock<Stream>;
struct HostAllocator
    : public at::CachingHostAllocatorImpl<
          Stream,
          c10::backend::CachingAllocator::EventPool<Event>::Event> {
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
    aclrtMallocHost(ptr, size);
    pinned_ptrs.insert(*ptr);
  }

  void free_block(Block* block) override {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    aclrtFreeHost(block->ptr_);
    pinned_ptrs.erase(block->ptr_);
  }

  void record_stream(
      std::optional<std::vector<
          c10::backend::CachingAllocator::EventPool<Event>::Event>>& events,
      Stream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(
      c10::backend::CachingAllocator::EventPool<Event>::Event& event) override {
    return event->query();
  }

  c10::backend::CachingAllocator::EventPool<Event>::Event create_event_internal(
      c10::DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool =
        new c10::backend::CachingAllocator::EventPool<Event>(
            device_count(), []() {
              return std::make_unique<Event>(ACL_EVENT_CAPTURE_STREAM_PROGRESS);
            });
    return event_pool->get(idx);
  }

  std::shared_mutex mutex_{};
  std::set<const void*> pinned_ptrs{};
};

void raw_local_deleter(void* ptr);

struct NPUCachingHostAllocator final
    : public at::CachingHostAllocatorInterface<HostAllocator> {
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

at::Allocator* getAllocator() {
  return &npu_caching_host_allocator;
}

void raw_local_deleter(void* ptr) {
  npu_caching_host_allocator.free(ptr);
}

bool recordEvent(void* ptr, void* ctx, c10::backend::Stream stream) {
  return npu_caching_host_allocator.record_event(ptr, ctx, stream);
}

void emptyCache() {
  npu_caching_host_allocator.empty_cache();
}

bool isPinndPtr(const void* ptr) {
  return npu_caching_host_allocator.isPinnedPtr(ptr);
}

} // namespace c10::backend::HostAllocator