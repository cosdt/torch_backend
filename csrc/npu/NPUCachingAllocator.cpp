#include "csrc/npu/NPUCachingAllocator.h"
#include <iostream>
#include "csrc/core/allocator/CachingAllocator.h"

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

static DefaultNPUAllocator defaultNPUAllocator;
std::atomic<NPUAllocator*> npu_allocator = &defaultNPUAllocator;

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &defaultNPUAllocator);

void init(CachingAllocator* delegate) {
  defaultNPUAllocator.init(delegate);
}
} // namespace c10::npu::NPUCachingAllocator
