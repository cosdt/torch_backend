#pragma once

#include <ATen/Context.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <c10/core/GeneratorImpl.h>
#include <limits>
#include "csrc/core/Macros.h"
#include "csrc/core/generator/GeneratorImpl.h"

namespace c10::backend {
/**
 * Note [Device Graph-safe RNG states]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Strategy:
 * ~~~~~~~~~
 * A device graph containing multiple RNG ops behaves like a
 * single giant kernel from the perspective of ops external
 * to the graph.  During graph capture, logic below records
 * the total of all offset increments that occur in the graphed
 * region, and records the final total as the offset for the
 * entire graph.
 *
 * When the graph reruns, the logic that reruns it
 * increments this device generator's offset
 * by that total.
 *
 * Meanwhile, within the graph, at capture time, instead of
 * populating PhiloxStates with the uint64_t offset pulled
 * directly from the global state, PhiloState instead
 * holds a pointer to one-element stream-local int64_t device tensor
 * holding an initial offset value, and a uint64_t holding an
 * intra-graph offset. (The intra-graph offset starts from zero
 * when capture begins.)  In each consumer kernel,
 * at::device::philox::unpack computes the offset to use for this kernel
 * as intra-graph offset + *initial offset.
 *
 * When the graph reruns, the logic that reruns it first
 * fill_s the initial offset tensor with this device generator's current offset.
 *
 * The control flow above ensures graphed execution is bitwise
 * identical to eager execution as long as RNG ops are enqueued
 * from a single thread, even if RNG ops and graphs containing
 * RNG ops are enqueued and run simultaneously on multiple streams.
 *
 * Usage:
 * ~~~~~~
 * PhiloxState in this file, and unpack() in
 * GraphsUtils.cuh allow non-divergent use of
 * DeviceGeneratorImpl whether graph capture is underway or not.
 *
 * Each PhiloxState instance should be used for one and only one
 * consumer kernel.
 *
 * Example (see e.g. native/device/Dropout.cu):
 *
 * #include <ATen/GeneratorImpl.h>
 * #include <ATen/device/GraphsUtils.cuh>
 *
 * __global__ void kernel(..., PhiloxState philox_args) {
 *   auto seeds = at::device::philox::unpack(philox_args);
 *   IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
 *   curandStatePhilox4_32_10_t state;
 *   curand_init(std::get<0>(seeds), // seed
 *               idx,                // per-thread subsequence
 *               std::get<1>(seeds), // offset in subsequence
 *               &state);
 *   ...
 * }
 *
 * host_caller(...) {
 *   PhiloxState rng_engine_inputs;
 *   {
 *     // See Note [Acquire lock when using random generators]
 *     std::lock_guard<std::mutex> lock(gen->mutex_);
 *
 *     // gen could be HostState or DevState here! No divergent code needed!
 *     rng_engine_inputs = gen->philox_state(offset_increment);
 *   }
 *   kernel<<<...>>>(..., rng_engine_inputs);
 * }
 *
 */

// Stores state values. Passed as a kernel argument. See "Usage:" above.
struct PhiloxState {
  PhiloxState() = default;
  PhiloxState(const PhiloxState&) = default;
  // Called if graph capture is not underway
  PhiloxState(uint64_t seed, uint64_t offset) {
    seed_ = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxState(
      uint64_t seed,
      int64_t* offset_extragraph,
      uint32_t offset_intragraph) {
    seed_ = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  // Public members, directly accessible by at::device::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  uint64_t seed_;
  Payload offset_;
  uint32_t offset_intragraph_{0};
  bool captured_ = false;
};

struct TORCH_BACKEND_API DeviceGeneratorImpl
    : public c10::backend::Generator::GeneratorImpl {
  // Constructors
  DeviceGeneratorImpl(c10::DeviceIndex device_index = -1);
  ~DeviceGeneratorImpl() = default;

  std::shared_ptr<DeviceGeneratorImpl> clone() const;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread() const;
  void capture_prologue(int64_t* offset_extragraph);
  uint64_t capture_epilogue();
  PhiloxState philox_state(uint64_t increment);

  // Temporarily accommodates call sites that use philox_engine_inputs.
  // Allows incremental refactor of call sites to use philox_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  static c10::DeviceType device_type();

 private:
  DeviceGeneratorImpl* clone_impl() const override;
  uint64_t seed_ = c10::default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
  int64_t* offset_extragraph_ = nullptr;
  uint32_t offset_intragraph_ = 0;
  bool graph_expects_this_gen_ = false;
};

namespace detail {
TORCH_BACKEND_API const at::Generator& getDefaultGenerator(
    c10::DeviceIndex device_index = -1);
TORCH_BACKEND_API at::Generator createGenerator(
    c10::DeviceIndex device_index = -1);

} // namespace detail
} // namespace c10::backend
