#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>
#include <c10/util/Registry.h>
#include <c10/util/SmallVector.h>
#include <atomic>

namespace c10::backend::CachingAllocator {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback(){};
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeNPUMemoryCallbacksRegistry, FreeMemoryCallback);

#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeNPUMemoryCallbacksRegistry, name, __VA_ARGS__);

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;
// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from npuMalloc().
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via npuFree)
  StatArray inactive_split;

  // SUM: bytes requested by client code
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // COUNT: total number of failed calls to NPU malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to NPU after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

typedef std::shared_ptr<c10::GatheredContext> (*CreateContextFn)(void);

// Struct containing info of an allocation block (i.e. a fractional part of a
// cudaMalloc)..
struct BlockInfo {
  int64_t size = 0;
  int64_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  std::shared_ptr<c10::GatheredContext> context_when_allocated;
};

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  void* stream = 0;
  int64_t total_size = 0;
  int64_t requested_size = 0;
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  bool is_large = false;
  bool is_expandable = false;
  std::vector<BlockInfo> blocks;
  std::shared_ptr<c10::GatheredContext> context_when_allocated;
};

struct TraceEntry {
  enum Action {
    ALLOC, // API made to the caching allocator for new memory
    FREE_REQUESTED, // API call made to the caching allocator to free memory
    FREE_COMPLETED, // The allocator might have to delay a free because
                    // it is still in use on another stream via
                    // record_stream This event is generated when a free
                    // actually completes.
    SEGMENT_ALLOC, // a call to malloc to get more memory from the OS
    SEGMENT_FREE, // a call to free to return memory to the OS (e.g. to
                  // defragment or empty_caches)
    SEGMENT_MAP, // a call to memMap (used with expandable_segments)
    SEGMENT_UNMAP, // unmap part of a segment (used with expandable
                   // segments)
    SNAPSHOT, // a call to snapshot, used to correlate memory snapshots to
              // trace events
    OOM // the allocator threw an OutOfMemoryError (addr_ is the amount of
        // free bytes reported by device)
  };
  TraceEntry(
      Action action,
      c10::DeviceIndex device,
      int64_t addr,
      size_t size,
      void* stream,
      std::shared_ptr<c10::GatheredContext> context = nullptr)
      : action_(action),
        device_(device),
        addr_(addr),
        context_(std::move(context)),
        stream_(stream),
        size_(size) {}
  Action action_;
  c10::DeviceIndex device_;
  int64_t addr_; // for OOM, this is the amount of free bytes reported by device
  std::shared_ptr<c10::GatheredContext> context_;
  void* stream_;
  int64_t size_;
};

struct SnapshotInfo {
  std::vector<SegmentInfo> segments;
  std::vector<std::vector<TraceEntry>> device_traces;
};

enum struct RecordContext {
  NEVER = 0,
  STATE = 1, // only keep stacks for active allocations
  ALLOC = 2, // additionally keep stacks for allocations in the trace history
  ALL = 3, // additionally record stacks for when something is freed
};

using OutOfMemoryObserver = std::function<void(
    int64_t device,
    int64_t allocated,
    int64_t device_total,
    int64_t device_free)>;

class CachingAllocator : public c10::Allocator {
 public:
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void* raw_alloc_with_stream(size_t nbytes, void* stream) = 0;
  virtual void raw_delete(void* ptr) = 0;
  virtual void init(int device_count) = 0;
  virtual bool initialized() = 0;
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) = 0;
  virtual void emptyCache(bool check_error) = 0;
  virtual void emptyDeviceCache(c10::DeviceIndex device) = 0;
  virtual void cacheInfo(
      c10::DeviceIndex device,
      size_t* cachedAndFree,
      size_t* largestBlock) = 0;
  virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
  virtual void recordStream(const c10::DataPtr& ptr, c10::Stream stream) = 0;
  virtual void eraseStream(const c10::DataPtr& ptr, c10::Stream stream) = 0;
  virtual DeviceStats getDeviceStats(c10::DeviceIndex device) = 0;
  virtual void resetAccumulatedStats(c10::DeviceIndex device) = 0;
  virtual void resetPeakStats(c10::DeviceIndex device) = 0;
  virtual SnapshotInfo snapshot() = 0;
  virtual std::string name() = 0;
  virtual bool isHistoryEnabled() {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) = 0;
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;
};

// Allocator object, statically initialized
// See BackendInitializer in CachingAllocator.cpp.
// Atomic loads on x86 are just normal loads,
// (atomic stores are different), so reading this value
// is no different than loading a pointer.
extern std::atomic<CachingAllocator*> caching_allocator;

inline CachingAllocator* get() {
  return caching_allocator.load();
}

// Called directly by clients.
inline void init(int device_count) {
  return get()->init(device_count);
}

inline void* raw_alloc(size_t nbytes) {
  return get()->raw_alloc(nbytes);
}

inline void* raw_alloc_with_stream(size_t nbytes, void* stream) {
  return get()->raw_alloc_with_stream(nbytes, stream);
}

inline void raw_delete(void* ptr) {
  return get()->raw_delete(ptr);
}

inline void setMemoryFraction(double fraction, c10::DeviceIndex device) {
  return get()->setMemoryFraction(fraction, device);
}

inline void emptyCache(bool check_error = true) {
  return get()->emptyCache(check_error);
}

inline void cacheInfo(
    c10::DeviceIndex device,
    size_t* cachedAndFree,
    size_t* largestBlock) {
  return get()->cacheInfo(device, cachedAndFree, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size) {
  return get()->getBaseAllocation(ptr, size);
}

inline void recordStream(const c10::DataPtr& ptr, c10::Stream stream) {
  return get()->recordStream(ptr, stream);
}

inline void eraseStream(const c10::DataPtr& ptr, c10::Stream stream) {
  return get()->eraseStream(ptr, stream);
}

inline DeviceStats getDeviceStats(c10::DeviceIndex device) {
  return get()->getDeviceStats(device);
}

inline void resetAccumulatedStats(c10::DeviceIndex device) {
  return get()->resetAccumulatedStats(device);
}

inline void resetPeakStats(c10::DeviceIndex device) {
  return get()->resetPeakStats(device);
}

inline SnapshotInfo snapshot() {
  return get()->snapshot();
}

inline void emptyDeviceCache(c10::DeviceIndex device) {
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

} // namespace c10::backend::CachingAllocator
