#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <mutex>

namespace c10_backend {

// Note: create device events when concurrently invoked from multiple threads
// can be very expensive (at least on certain device/driver combinations). Thus,
// we a) serialize event creation at a per-device level, and b) pool the events
// to avoid constantly calling `create event`/`destroy event`. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
template <typename T>
class EventPool {
 public:
  using Event = std::unique_ptr<T, std::function<void(T*)>>;
  EventPool(int device_cout, std::function<std::unique_ptr<T>()> create_event)
      : pools_(device_cout), create_event_(create_event) {}

  Event get(c10::DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(
        device < static_cast<c10::DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](T* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<T>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(create_event_().release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<T>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
  std::function<std::unique_ptr<T>()> create_event_;
};
} // namespace c10_backend