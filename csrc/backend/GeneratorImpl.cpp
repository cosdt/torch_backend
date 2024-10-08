#include "GeneratorImpl.h"
#include <ATen/Utils.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <c10/core/StreamGuard.h>
#include "csrc/aten/generated/NPUNativeFunctions.h"
#include "csrc/backend/Functions.h"

namespace c10::backend {
namespace detail {

namespace {

// Ensures we only call GetDeviceCount only once.
static std::once_flag num_device_init_flag;

// Total number of devices in the system.
static int64_t num_devices;

// Ensures default_gens_device is initialized once.
static std::deque<std::once_flag> device_gens_init_flag;

// Default, global device generators, one per device.
static std::vector<at::Generator> default_gens_device;

/*
 * Populates the global variables related to device generators
 * Warning: this function must only be called once!
 */
static void initGenVector() {
  num_devices = c10::backend::device_count();
  device_gens_init_flag.resize(num_devices);
  default_gens_device.resize(num_devices);
}

} // anonymous namespace

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultGenerator gets the default generator for a particular
 * device.
 */
const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
  std::call_once(num_device_init_flag, initGenVector);
  c10::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::backend::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_devices, PTA_ERROR(ErrCode::VALUE));
  }
  std::call_once(device_gens_init_flag[idx], [&] {
    default_gens_device[idx] = at::make_generator<DeviceGeneratorImpl>(idx);
    default_gens_device[idx].seed();
  });
  return default_gens_device[idx];
}

/**
 * Utility to create a DeviceGeneratorImpl. Returns a shared_ptr
 */
at::Generator createGenerator(c10::DeviceIndex device_index) {
  std::call_once(num_device_init_flag, initGenVector);
  c10::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::backend::current_device();
  }
  TORCH_CHECK(
      idx >= 0 && idx < num_devices,
      "The device_index is invalid.",
      PTA_ERROR(ErrCode::VALUE));
  auto gen = at::make_generator<DeviceGeneratorImpl>(idx);
  auto device_gen = at::check_generator<DeviceGeneratorImpl>(gen);
  device_gen->set_current_seed(c10::default_rng_seed_val);
  device_gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace detail

/**
 * DeviceGeneratorImpl class implementation
 */
DeviceGeneratorImpl::DeviceGeneratorImpl(c10::DeviceIndex device_index)
    : GeneratorImpl(device_index) {}

#define CAPTURE_DEFAULT_GENS_MSG                                                    \
  "In regions captured by device graphs, you may only use the default device RNG "  \
  "generator on the device that's current when capture begins. "                    \
  "If you need a non-default (user-supplied) generator, or a generator on another " \
  "device, please file an issue."

/**
 * Gets the current internal state of DeviceGeneratorImpl. The internal
 * state is returned as a CPU byte tensor.
 */
c10::intrusive_ptr<c10::TensorImpl> DeviceGeneratorImpl::get_state() const {
  // The RNG state comprises the seed, and an offset used for Philox.
  // The following line is just here for BC reason. sizeof curandStateMtgp32 is
  // 4120. It used to be static const size_t states_size = MAX_NUM_BLOCKS *
  // sizeof(curandStateMtgp32); MAX_NUM_BLOCKS was 200 and
  // sizeof(curandStateMtgp32) is 4120. Hardcoding these numbers here because
  // this is just host side code and we don't want to worry about linking with
  // device
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu(
      {(int64_t)total_size},
      at::ScalarType::Byte,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt);
  auto rng_state = state_tensor.data_ptr<uint8_t>();
  // since curandStateMTGP is not used anymore, fill gen_states of THCGenerator
  // with deterministic garbage value of -1 gen_states in THCGenerator struct
  // was an array of curandStateMtgp32s.
  auto current_seed = this->current_seed();
  auto offset = static_cast<int64_t>(
      this->philox_offset_per_thread()); // Note that old THCGeneratorState had
                                         // offset as std::atomic<int64_t>
  memcpy(rng_state, &current_seed, seed_size);
  memcpy(rng_state + seed_size, &offset, offset_size);

  return state_tensor.getIntrusivePtr();
}

/**
 * Sets the internal state of DeviceGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and have appropriate size. See
 * comments of DeviceGeneratorImpl::state for information about the layout
 * and size of the internal state.
 */
void DeviceGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  at::detail::check_rng_state(new_state);

  bool no_philox_seed = false;
  auto new_state_size = new_state.numel();
  if (new_state_size == total_size - offset_size) {
    no_philox_seed = true;
  } else {
    TORCH_CHECK(
        new_state_size == total_size,
        "RNG state is wrong size",
        PTA_ERROR(ErrCode::PARAM));
  }

  uint64_t input_seed;
  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&input_seed, new_rng_state, seed_size);
  this->set_current_seed(input_seed);
  int64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
  }
  this->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

/**
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void DeviceGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_CHECK(
      offset % 4 == 0,
      "offset must be a multiple of 4",
      PTA_ERROR(ErrCode::VALUE));
  philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of DeviceGeneratorImpl.
 */
uint64_t DeviceGeneratorImpl::philox_offset_per_thread() const {
  return philox_offset_per_thread_;
}

/**
 * Called by DeviceGraph to prepare this instance for a graph capture region.
 * offset_extragraph is the initial offset at the start of the graphed region.
 * offset_intragraph tracks the offset in the graphed region.
 */
void DeviceGeneratorImpl::capture_prologue(int64_t* offset_extragraph) {
  offset_extragraph_ = offset_extragraph;
  offset_intragraph_ = 0;
  graph_expects_this_gen_ = true;
}

/**
 * Called by DeviceGraph to finalize a graph capture region for this instance.
 */
uint64_t DeviceGeneratorImpl::capture_epilogue() {
  graph_expects_this_gen_ = false;
  return offset_intragraph_;
}

/**
 * Gets the seed and philox offset value to be used in
 * curandStatePhilox4_32_10, in an opaque PhiloxState that's safe
 * and can be used non-divergently in callers whether device graph
 * capture is underway or not.  See
 * Note [Device Graph-safe RNG states]
 *
 * Each kernel using philox has to sensibly increment offset
 * for future users of philox. So it gets the "old" value for
 * itself (before add), and tells subsequent users which offset
 * they should use, since only the kernel knows how many randoms
 * it intends to generate.
 *
 * Increment should be at least the number of curand() random numbers used in
 * each thread. It is the user's responsibility to make sure the increment
 * for philox is never smaller than the number of curand() calls. Increment
 * value > the number of curand() calls won't harm but anything less would mean
 * that you would be reusing random values from previous calls.
 *
 * See Note [Acquire lock when using random generators]
 */
PhiloxState DeviceGeneratorImpl::philox_state(uint64_t increment) {
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  /*
  if (at::device::currentStreamCaptureStatus() !=
  at::device::CaptureStatus::None) { TORCH_CHECK(graph_expects_this_gen_,
                "philox_state for an unexpected device generator used during
  capture. " CAPTURE_DEFAULT_GENS_MSG);
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(this->offset_intragraph_ % 4 == 0);
    uint32_t offset = this->offset_intragraph_;
    TORCH_INTERNAL_ASSERT(this->offset_intragraph_ <=
                          std::numeric_limits<uint32_t>::max() - increment);
    this->offset_intragraph_ += increment;
    return PhiloxState(this->seed_,
                           this->offset_extragraph_,
                           offset);
  } else {
    TORCH_CHECK(!graph_expects_this_gen_,
                "Device generator expects graph capture to be underway, "
                "but the current stream is not capturing.");
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
    uint64_t offset = this->philox_offset_per_thread_;
    this->philox_offset_per_thread_ += increment;
    return PhiloxState(this->seed_, offset);
  } */

  return PhiloxState(this->seed_, 0);
}

/**
 * Temporarily accommodates call sites that use philox_engine_inputs.
 * Allows incremental refactor of call sites to use philox_state.
 */
std::pair<uint64_t, uint64_t> DeviceGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_INTERNAL_ASSERT(
      this->philox_offset_per_thread_ % 4 == 0, PTA_ERROR(ErrCode::INTERNAL));
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

/*
 * Gets the DeviceType of DeviceGeneratorImpl.
 * Used for type checking during run time.
 */
c10::DeviceType DeviceGeneratorImpl::device_type() {
  return c10::DeviceType::PrivateUse1;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<DeviceGeneratorImpl> DeviceGeneratorImpl::clone() const {
  return std::shared_ptr<DeviceGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
DeviceGeneratorImpl* DeviceGeneratorImpl::clone_impl() const {
  auto gen = new DeviceGeneratorImpl(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

// this is used to register generator
at::Generator make_device_generator(c10::DeviceIndex device_index) {
  return at::make_generator<DeviceGeneratorImpl>(device_index);
}

REGISTER_GENERATOR_PRIVATEUSE1(make_device_generator)

} // namespace c10::backend
