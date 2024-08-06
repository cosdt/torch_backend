#include "torch_npu/csrc/npu/Memory.h"
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/profiler/python/combined_traceback.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include "csrc/npu/NPUCachingAllocator.h"

PyObject* THNPModule_setMemoryFraction(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* fraction_o = nullptr;
  PyObject* device_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "set_memory_fraction",
        1,
        "(double fraction, int device);");
    return nullptr;
  }
  double fraction = PyFloat_AsDouble(fraction_o);
  int64_t device = PyLong_AsLongLong(device_o);

  c10::npu::NPUCachingAllocator::setMemoryFraction(fraction, device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THNPModule_resetAccumulatedMemoryStats(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "invalid argument to reset_accumulated_memory_stats");

  const int device = (int)THPUtils_unpackLong(arg);
  c10::npu::NPUCachingAllocator::resetAccumulatedStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THNPModule_resetPeakMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");

  const int device = (int)THPUtils_unpackLong(arg);
  c10::npu::NPUCachingAllocator::resetPeakStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

torch::CapturedTraceback* getFromContext(
    const std::shared_ptr<c10::GatheredContext>& x) {
  if (torch::CapturedTraceback* sc =
          dynamic_cast<torch::CapturedTraceback*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(
      false,
      "attempting to gather stack context from the wrong StackContext type.");
}

PyObject* THNPModule_memorySnapshot(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS

  using c10::backend::CachingAllocator::BlockInfo;
  using c10::backend::CachingAllocator::SegmentInfo;

  py::str device_s = "device";
  py::str address_s = "address";
  py::str total_size_s = "total_size";
  py::str allocated_size_s = "allocated_size";
  py::str active_size_s = "active_size";
  py::str requested_size_s = "requested_size";
  py::str stream_s = "stream";
  py::str segment_type_s = "segment_type";
  py::str large_s = "large";
  py::str small_s = "small";
  py::str size_s = "size";
  py::str state_s = "state";
  py::str active_allocated_s = "active_allocated";
  py::str active_pending_free_s = "active_pending_free";
  py::str inactive_s = "inactive";
  py::str addr_s = "addr";
  py::str cpp_frames_s = "cpp_frames";
  py::str blocks_s = "blocks";
  py::str is_expandable_s = "is_expandable";
  py::str frames_s = "frames";

  py::list empty_frames;
  std::vector<torch::CapturedTraceback*> to_gather_frames;
  std::vector<py::dict> to_gather_dest;

  auto add_frame_key = [&](const py::dict& d,
                           const std::shared_ptr<c10::GatheredContext>& ctx) {
    if (ctx) {
      auto sc = getFromContext(ctx);
      to_gather_frames.emplace_back(sc);
      to_gather_dest.emplace_back(d);
    } else {
      d[frames_s] = empty_frames;
    }
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict[device_s] = segmentInfo.device;
    segmentDict[address_s] = segmentInfo.address;
    segmentDict[total_size_s] = segmentInfo.total_size;
    segmentDict[allocated_size_s] = segmentInfo.allocated_size;
    segmentDict[active_size_s] = segmentInfo.active_size;
    segmentDict[requested_size_s] = segmentInfo.requested_size;
    // we want the python objects to pickle easily so use an int to
    // represent the stream rather than a torch.cuda.stream object
    segmentDict[stream_s] = int64_t(segmentInfo.stream);
    segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);
    segmentDict[is_expandable_s] = segmentInfo.is_expandable;
    add_frame_key(segmentDict, segmentInfo.context_when_allocated);

    auto address = segmentInfo.address;
    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict[address_s] = address;
      blockDict[size_s] = blockInfo.size;
      blockDict[requested_size_s] = blockInfo.requested_size;
      blockDict[state_s] =
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s));
      add_frame_key(blockDict, blockInfo.context_when_allocated);
      blocks.append(blockDict);
      address += blockInfo.size;
    }
    segmentDict[blocks_s] = blocks;

    return segmentDict;
  };

  auto snapshot = c10::npu::NPUCachingAllocator::snapshot();
  py::list segments;

  for (const auto& segmentInfo : snapshot.segments) {
    segments.append(segmentInfoToDict(segmentInfo));
  }

  py::list traces;
  py::str action_s = "action";
  py::str alloc_s = "alloc";
  py::str free_requested_s = "free_requested";
  py::str free_completed_s = "free_completed";
  py::str segment_alloc_s = "segment_alloc";
  py::str segment_free_s = "segment_free";
  py::str segment_map_s = "segment_map";
  py::str segment_unmap_s = "segment_unmap";

  py::str snapshot_s = "snapshot";
  py::str oom_s = "oom";
  py::str device_free_s = "device_free";

  using namespace c10::backend::CachingAllocator;

  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
      case TraceEntry::SEGMENT_UNMAP:
        return segment_unmap_s;
      case TraceEntry::SEGMENT_MAP:
        return segment_map_s;
      default:
        AT_ERROR("invalid TraceEntry action");
    }
    throw std::runtime_error("unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    py::list trace;
    for (const auto& te : traceInfo) {
      py::dict trace_entry;
      if (te.context_) {
        // without further compression frames can get really large on dump
        auto sc = getFromContext(te.context_);
        to_gather_frames.emplace_back(sc);
        to_gather_dest.emplace_back(trace_entry);
      }
      trace_entry[action_s] = action_to_str(te.action_);
      trace_entry[TraceEntry::OOM == te.action_ ? device_free_s : addr_s] =
          te.addr_;
      trace_entry[size_s] = te.size_;
      trace_entry[stream_s] = int64_t(te.stream_);
      trace.append(trace_entry);
    }
    traces.append(trace);
  }

  py::dict result;
  result["segments"] = segments;
  result["device_traces"] = traces;

  auto frames = torch::py_symbolize(to_gather_frames);
  for (auto i : c10::irange(frames.size())) {
    to_gather_dest.at(i)[frames_s] = frames.at(i);
  }
  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_attachOutOfMemoryObserver(
    PyObject* _unused,
    PyObject* observer) {
  HANDLE_TH_ERRORS
  Py_XINCREF(observer);
  auto obs = [observer](
                 int64_t device,
                 int64_t alloc,
                 int64_t device_allocated,
                 int64_t device_free) {
    py::gil_scoped_acquire g;
    PyObject* result = PyObject_CallFunction(
        observer, "LLLL", device, alloc, device_allocated, device_free);
    if (!result) {
      throw py::error_already_set();
    }
    Py_XDECREF(result);
  };
  torch::utils::device_lazy_init(at::kPrivateUse1);
  c10::npu::NPUCachingAllocator::attachOutOfMemoryObserver(std::move(obs));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npuCachingAllocator_raw_alloc(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* size_o = nullptr;
  PyObject* stream_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "caching_allocator_alloc",
        1,
        "(ssize_t size, intptr_t stream);");
    return nullptr;
  }
  ssize_t size = PyLong_AsSsize_t(size_o);
  void* stream = static_cast<void*>(PyLong_AsVoidPtr(stream_o));
  void* mem =
      c10::npu::NPUCachingAllocator::raw_alloc_with_stream(size, stream);
  return PyLong_FromVoidPtr(mem);
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npuCachingAllocator_raw_delete(
    PyObject* _unused,
    PyObject* obj) {
  HANDLE_TH_ERRORS
  void* mem_ptr = PyLong_AsVoidPtr(obj);
  c10::npu::NPUCachingAllocator::raw_delete(mem_ptr);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getAllocatorBackend(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(c10::npu::NPUCachingAllocator::name());
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_memoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_allocated");

  const int device = (int)THPUtils_unpackLong(arg);

  using c10::backend::CachingAllocator::DeviceStats;
  using c10::backend::CachingAllocator::Stat;
  using c10::backend::CachingAllocator::StatArray;
  using c10::backend::CachingAllocator::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (size_t i = 0; i < statTypeNames.size(); ++i) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats =
      c10::npu::NPUCachingAllocator::getDeviceStats(device);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::npu::NPUCachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

static struct PyMethodDef THNPModule_methods[] = {
    {"_npu_setMemoryFraction",
     (PyCFunction)THNPModule_setMemoryFraction,
     METH_VARARGS,
     nullptr},
    {"_npu_emptyCache",
     (PyCFunction)THNPModule_emptyCache,
     METH_NOARGS,
     nullptr},
    {"_npu_memoryStats", (PyCFunction)THNPModule_memoryStats, METH_O, nullptr},
    {"_npu_resetAccumulatedMemoryStats",
     (PyCFunction)THNPModule_resetAccumulatedMemoryStats,
     METH_O,
     nullptr},
    {"_npu_resetPeakMemoryStats",
     (PyCFunction)THNPModule_resetPeakMemoryStats,
     METH_O,
     nullptr},
    {"_npu_memorySnapshot",
     (PyCFunction)THNPModule_memorySnapshot,
     METH_NOARGS,
     nullptr},
    {"_npu_attach_out_of_memory_observer",
     THNPModule_attachOutOfMemoryObserver,
     METH_O,
     nullptr},
    {"_npu_npuCachingAllocator_raw_alloc",
     (PyCFunction)THNPModule_npuCachingAllocator_raw_alloc,
     METH_VARARGS,
     nullptr},
    {"_npu_npuCachingAllocator_raw_delete",
     (PyCFunction)THNPModule_npuCachingAllocator_raw_delete,
     METH_O,
     nullptr},
    {"_npu_getAllocatorBackend",
     (PyCFunction)THNPModule_getAllocatorBackend,
     METH_NOARGS,
     nullptr},
    {nullptr}};

namespace torch::backend::memory {

PyMethodDef* python_functions() {
  return THNPModule_methods;
}

} // namespace torch::backend::memory
