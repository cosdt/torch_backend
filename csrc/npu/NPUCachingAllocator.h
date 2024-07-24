#pragma once

#include "csrc/core/CachingAllocator.h"
#include "npu/core/NPUMacros.h"

namespace c10_npu::NPUCachingAllocator {

using namespace c10_backend::CachingAllocator;

std::mutex* getFreeMutex();

class NPUAllocator : public c10::Allocator {
 public:
  virtual void* raw_alloc_with_stream(size_t nbytes, void* stream) = 0;
  virtual void raw_delete(void* ptr) = 0;
  virtual void setMemoryFraction(double fraction, int device) = 0;
  virtual void emptyCache(bool check_error) = 0;
  virtual void emptyDeviceCache(int device) = 0;
  virtual void recordStream(const c10::DataPtr& ptr, c10::Stream stream) = 0;
  virtual DeviceStats getDeviceStats(int device) = 0;
  virtual void resetAccumulatedStats(int device) = 0;
  virtual void resetPeakStats(int device) = 0;
  virtual SnapshotInfo snapshot() = 0;
  virtual std::string name() = 0;
  virtual bool isHistoryEnabled() = 0;
  virtual void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) = 0;
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;
};

extern std::atomic<NPUAllocator*> npu_allocator;

inline NPUAllocator* get() {
  return npu_allocator.load();
}

void init(CachingAllocator* delegate);

// Called directly by clients.

inline void* raw_alloc_with_stream(size_t nbytes, void* stream) {
  return get()->raw_alloc_with_stream(nbytes, stream);
}

inline void raw_delete(void* ptr) {
  return get()->raw_delete(ptr);
}

inline void setMemoryFraction(double fraction, int device) {
  return get()->setMemoryFraction(fraction, device);
}

inline void emptyCache(bool check_error = true) {
  return get()->emptyCache(check_error);
}

inline void recordStream(const c10::DataPtr& ptr, c10::Stream stream) {
  return get()->recordStream(ptr, stream);
}

inline DeviceStats getDeviceStats(int device) {
  return get()->getDeviceStats(device);
}

inline void resetAccumulatedStats(int device) {
  return get()->resetAccumulatedStats(device);
}

inline void resetPeakStats(int device) {
  return get()->resetPeakStats(device);
}

inline SnapshotInfo snapshot() {
  return get()->snapshot();
}

inline void emptyDeviceCache(int device) {
  return get()->emptyDeviceCache(device);
}

inline std::string name() {
  return get()->name();
}

inline void recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when) {
  return get()->recordHistory(
      enabled, context_recorder, alloc_trace_max_entries, when);
}

inline bool isHistoryEnabled() {
  return get()->isHistoryEnabled();
}

inline void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
  return get()->attachOutOfMemoryObserver(observer);
}
} // namespace c10_npu::NPUCachingAllocator