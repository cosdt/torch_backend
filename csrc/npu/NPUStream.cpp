#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "csrc/npu/NPUFunctions.h"
#include "csrc/npu/NPUStream.h"
#include "npu/acl/include/acl/acl_rt.h"
#include "npu/adapter/acl_device_adapter.h"
#include "npu/core/NPUException.h"
#include "npu/core/NPUGuard.h"
#include "npu/core/interface/AsyncTaskQueueInterface.h"
#include "npu/core/register/OptionsManager.h"

#define C10_COMPILE_TIME_MAX_NPUS 16

namespace c10_npu {
namespace {

// Global stream state and constants
static c10::once_flag init_flag;
static c10::DeviceIndex num_npus = -1;
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
static constexpr int kStreamTypeBits = 4;

static std::once_flag device_flags[C10_COMPILE_TIME_MAX_NPUS];

// Non-default streams
// Note: the number of NPU devices is determined at run time,
// and the low and high priority pools are lazily initialized
// when the first stream is requested for a device.
// The device flags track the initialization of each device, while
// the low and high priority counters track, for each device, the next stream
// in the pool to be returned when a stream is requested (round-robin fashion
// , see the note in NPUStream.h).
// The streams are "leaked": they are created but never destroyed because the
// destruction of global variables could happen after the NPU runtime has
// already been destroyed and thus invoking NPUStreamDestroy could lead to a
// crash. It's likely an issue in NPU, but to be safe - let's just "forget"
// the destruction.
static std::array<
    std::array<std::atomic<uint32_t>, C10_COMPILE_TIME_MAX_NPUS>,
    max_compile_time_stream_priorities>
    priority_counters;

static std::array<
    std::array<
        std::array<aclrtStream, kStreamsPerPool>,
        C10_COMPILE_TIME_MAX_NPUS>,
    max_compile_time_stream_priorities>
    streams;

thread_local std::unique_ptr<c10::StreamId[]> current_streams = nullptr;

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 54 bits --  -- 5 bits -----  -- 4 bits --     --1 bit --
// zeros          stream id index  StreamIdType     Ext/native stream
//                ignored for ext   ignored for ext
// for external stream, StreamID is a aclStream_t pointer
// this means that last bit will always be 0
// so when constructing StreamId for a native stream we set last bit to 1
// to distinguish between native and external streams
//
//
// We are obligated to treat the stream ID 0 as the default stream, per the
// invariant specified in c10::Stream, so this is one exception to
// "last bit = 1 for native streams". However, all other numbers are entirely
// an internal implementation detail, we reserve the right to renumber streams
// however we like.
//
// Note that it is really important that the MSB is zero; StreamId is a
// *signed* integer, and unsigned to signed conversion outside of the
// bounds of signed integer representation is undefined behavior.  You
// could work around this with something like
// https://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
// but it seems a bit overkill for this.
//
// Also, external managed stream pointers (aclStream_t) can be directly stored
// in the Id field so in this case, we need to check the stream alignment.

class StreamIdType {
  // StreamIdType encodes whether this stream is DEFAULT, EXTernal or
  // for all other native streams, the stream priority (higher value is higher
  // priority)
 private:
  uint8_t stream_type;

 public:
  static const uint8_t DEFAULT = 0x0;
  static const uint8_t EXT = 0xF;

 public:
  StreamIdType(const uint8_t _stream_type) : stream_type(_stream_type) {}

  bool isExt() const {
    return EXT == stream_type;
  }

  bool isDefault() const {
    return DEFAULT == stream_type;
  }

  uint8_t getStreamType() const {
    return stream_type;
  }
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  if (s.isDefault()) {
    stream << "DEFAULT";
  } else if (s.isExt()) {
    stream << "EXT";
  } else {
    stream << "PRIORITY " << int(s.getStreamType());
  }
  return stream;
}

// StreamId is 64-bit, so we can just rely on regular promotion rules.
// We rely on streamIdIndex and streamIdType being non-negative;
// see Note [Hazard when concatenating signed integers]

static inline StreamIdType streamIdType(c10::StreamId s) {
  // Externally allocated streams have their id being the NPUStream_ptr
  // so the last bit will be 0
  if ((!(s & 1)) && s) {
    return StreamIdType(StreamIdType::EXT);
  }
  // last bit is external/internal stream, the mask should start from second
  // rightmost bit
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto val = (s >> 1) & mask_for_type;
  TORCH_INTERNAL_ASSERT(val || !(s & 1), "invalid StreamId", s);
  return StreamIdType(val);
}

static inline size_t streamIdIndex(c10::StreamId s) {
  return static_cast<size_t>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

c10::StreamId makeStreamId(StreamIdType st, size_t si) {
  if (st.isDefault()) {
    return static_cast<c10::StreamId>(0);
  }
  return (static_cast<c10::StreamId>(si) << (kStreamTypeBits + 1)) |
      static_cast<c10::StreamId>(st.getStreamType() << 1) | 1;
}

static void initGlobalStreamState() {
  num_npus = c10_npu::device_count();
  // Check if the number of NPUs matches the expected compile-time max number
  // of NPUs.
  AT_ASSERTM(
      num_npus <= C10_COMPILE_TIME_MAX_NPUS,
      "Number of NPU devices on the machine is larger than the compiled "
      "max number of npus expected (",
      C10_COMPILE_TIME_MAX_NPUS,
      "). Increase that and recompile.",
      PTA_ERROR(ErrCode::VALUE));
}

// Init a single NPU stream
static void initSingleStream(int p, c10::DeviceIndex device_index, int i) {
  auto& stream = streams[p][device_index][i];
  auto pri = -p; // lower number is higher priority

  NPU_CHECK_SUPPORTED_OR_ERROR(acl::AclrtCreateStreamWithConfig(
        &stream,
        0,
        (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)));
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_creation(
        c10::kPrivateUse1, reinterpret_cast<uintptr_t>(stream));
    priority_counters[p][device_index] = 0;
  }
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
static void initDeviceStreamState(c10::DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  NPUGuard device_guard{device_index};
  for (const auto i : c10::irange(kStreamsPerPool)) {
    for (const auto p : c10::irange(max_compile_time_stream_priorities)) {
      initSingleStream(p, device_index, i);
    }
  }
}

static void initNPUStreamsOnce() {
  // Inits default and secondary streams (once, globally)
  c10::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  // NOLINTNEXTLINE(*-arrays)
  current_streams = std::make_unique<c10::StreamId[]>(num_npus);
  for (const auto i : c10::irange(num_npus)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

static inline void check_npu(c10::DeviceIndex device_index) {
  AT_ASSERT(
      device_index >= 0 && device_index < num_npus,
      "Invalid device_index : ",
      device_index,
      ", valid device_index range is [0, ",
      num_npus,
      ")",
      PTA_ERROR(ErrCode::VALUE));
}

static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

NPUStream NPUStreamForId(c10::DeviceIndex device_index, c10::StreamId stream_id) {
  return NPUStream(
      NPUStream::UNCHECKED,
      c10::Stream(
          c10::Stream::UNSAFE,
          c10::Device(c10::DeviceType::PrivateUse1, device_index),
          stream_id));
}

aclError SynchronizeUsedDevices() {
  c10::DeviceIndex cur_device = 0;
  NPU_CHECK_ERROR(GetDevice(&cur_device));
  // Synchronize all used devices
  std::vector<c10::DeviceIndex> device_idx_vec = acl_adapter::GetUsedDevices();
  for (const auto deviceId : device_idx_vec) {
    NPU_CHECK_ERROR(SetDevice(deviceId));
    aclError acl_ret = aclrtSynchronizeDevice();
    if (acl_ret != ACL_ERROR_NONE) {
      return acl_ret;
    }
  }
  NPU_CHECK_ERROR(SetDevice(cur_device));
  return ACL_ERROR_NONE;
}
} // namespace

aclrtStream NPUStream::stream() const {
  c10::DeviceIndex device_index = stream_.device_index();
  c10::StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  if (st.isDefault()) {
    TORCH_CHECK(
        si == 0,
        "Unrecognized stream ",
        stream_,
        " (I think this should be the default stream, but I got a non-zero index ",
        si,
        ").",
        " Did you manufacture the StreamId yourself?  Don't do that; use the",
        " official API like c10::NPU::getStreamFromPool() to get a new stream.");
    return nullptr;
  } else if (st.isExt()) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<aclrtStream>(stream_id);
  } else {
    auto streamType = st.getStreamType();
    TORCH_CHECK(
        streamType >= 1 && streamType <= max_compile_time_stream_priorities,
        "Unrecognized stream ",
        stream_,
        " (I didn't recognize the stream type, ",
        st,
        " with the value ",
        streamType,
        ")");
    return streams[st.getStreamType() - 1][device_index][si];
  }
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
NPUStream getStreamFromPool(const int priority, c10::DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }

  check_npu(device_index);

  // Initializes the stream pools (once)
  std::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);

  auto pri_idx = -priority;
  pri_idx =
      std::min(pri_idx, max_compile_time_stream_priorities - 1); // pri_idx is zero-based
  const auto idx = get_idx(priority_counters[pri_idx][device_index]);
  StreamIdType id_type = StreamIdType(pri_idx + 1);
  return NPUStreamForId(device_index, makeStreamId(id_type, idx));
}

NPUStream getStreamFromPool(const bool isHighPriority, c10::DeviceIndex device) {
  initNPUStreamsOnce();
  int priority = isHighPriority ? -max_compile_time_stream_priorities + 1 : 0;
  return getStreamFromPool(priority, device);
}

NPUStream getStreamFromExternal(
    aclrtStream ext_stream,
    c10::DeviceIndex device_index) {
  // The stream pointer will be the actual id
  return NPUStreamForId(device_index, reinterpret_cast<int64_t>(ext_stream));
}

NPUStream getDefaultNPUStream(c10::DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_npu(device_index);
  return NPUStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

NPUStream getCurrentNPUStream(c10::DeviceIndex device_index) {
  initNPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_npu(device_index);
  return NPUStreamForId(device_index, current_streams[device_index]);
}

void setCurrentNPUStream(NPUStream stream) {
  initNPUStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const NPUStream& s) {
  return stream << s.unwrap();
}

C10_BACKEND_API bool npuSynchronizeDevice(bool check_error) {
  auto acl_ret = aclrtSynchronizeDevice();
  if (check_error) {
    NPU_CHECK_ERROR(acl_ret, "aclrtSynchronizeDevice");
  } else {
    NPU_CHECK_WARN(acl_ret);
  }
  return acl_ret == ACL_ERROR_NONE;
}

C10_BACKEND_API bool npuSynchronizeUsedDevices(bool check_error) {
  auto acl_ret = SynchronizeUsedDevices();
  if (check_error) {
    NPU_CHECK_ERROR(acl_ret);
  } else {
    NPU_CHECK_WARN(acl_ret);
  }
  return acl_ret == ACL_ERROR_NONE;
}

aclError DestroyUsedStreams() {
  c10::DeviceIndex cur_device = 0;
  NPU_CHECK_ERROR(GetDevice(&cur_device));
  // Synchronize all used devices
  std::vector<c10::DeviceIndex> device_idx_vec = acl_adapter::GetUsedDevices();
  for (const auto deviceId : device_idx_vec) {
    NPU_CHECK_ERROR(SetDevice(deviceId));
    NPUStream stream = getCurrentNPUStream(deviceId);
    aclError acl_ret = acl::AclrtDestroyStreamForce(stream);
    if (acl_ret != ACL_ERROR_NONE) {
      return acl_ret;
    }
  }
  NPU_CHECK_ERROR(SetDevice(cur_device));
  return ACL_ERROR_NONE;
}
} // namespace c10_npu
