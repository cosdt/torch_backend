#include <ATen/Parallel.h>
#include <Python.h>
#include <torch/csrc/profiler/python/combined_traceback.h>

#include "csrc/npu/NPUCachingAllocator.h"
#include "csrc/npu/NPUCachingHostAllocator.h"
#include "npu/core/npu_log.h"
#include "npu/core/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/AutocastMode.h"
#include "torch_npu/csrc/core/TensorType.h"
#include "torch_npu/csrc/npu/Device.h"
#include "torch_npu/csrc/npu/Memory.h"
#include "torch_npu/csrc/npu/Module.h"

PyObject* module;

void AddPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods) {
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

PyObject* THPModule_npu_shutdown(PyObject* /* unused */) {
  // cudaFree is blocking and will synchronize across all kernels executing
  // on the current device, while aclrtFree Free device memory immediately.
  // aclrtSynchronizeDevice should be called before aclrtFree to ensure that
  // all of op tasks completed before device memory free.
  ASCEND_LOGI("NPU shutdown begin.");
  if (!c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
    Py_RETURN_NONE;
  }

  // Return aclrtSynchronizeDevice result. If sync device fails, release host
  // resources forcibly, only record WARN logs when acl interface of stream
  // or event fails.
  bool success = true;
  try {
    ASCEND_LOGI("NPU shutdown synchronize device.");
    c10_npu::synchronize_all_device();
  } catch (std::exception& e) {
    ASCEND_LOGE("SynchronizeDevice failed err=:%s", e.what());
    success = false;
  }
  if (!success) {
    ASCEND_LOGE("NPU shutdown synchronize device failed.");
  }

  NPUCachingHostAllocator_emptyCache();
  try {
    ASCEND_LOGI("NPU shutdown NPUCachingAllocator emptyCache.");
    c10_npu::NPUCachingAllocator::emptyCache(success);
  } catch (std::exception& e) {
    ASCEND_LOGE("NPUCachingAllocator::emptyCache failed err=:%s", e.what());
  }

  ASCEND_LOGI("NPU shutdown NpuSysCtrl Finalize.");
  if (!c10_npu::NpuSysCtrl::IsFinalizeSuccess()) {
    ASCEND_LOGE("NPU shutdown failed.");
  } else {
    ASCEND_LOGI("NPU shutdown success.");
  }

  Py_RETURN_NONE;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
static PyMethodDef TorchNpuMethods[] = {
    {"_npu_shutdown",
     (PyCFunction)THPModule_npu_shutdown,
     METH_NOARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

void THNPStream_init(PyObject* module);
void THNPEvent_init(PyObject* module);

PyMethodDef* THNPModule_get_methods();
PyMethodDef* THNPModule_device_methods();

static std::vector<PyMethodDef> methods;

extern "C" C10_EXPORT PyObject* initModule();
PyObject* initModule() {
  at::internal::lazy_init_num_threads();

  AddPyMethodDefs(methods, TorchNpuMethods);
  AddPyMethodDefs(methods, THNPModule_device_methods());
  AddPyMethodDefs(methods, THNPModule_memory_methods());
  AddPyMethodDefs(methods, THNPModule_get_methods());
  AddPyMethodDefs(methods, torch_npu::utils::npu_extension_functions());
  AddPyMethodDefs(methods, torch_npu::autocast::autocast_mode_functions());
  static struct PyModuleDef torchnpu_module = {
      PyModuleDef_HEAD_INIT, "torch_npu._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&torchnpu_module);

  // This will only initialize base classes and attach them to library namespace
  // They won't be ready for real usage until importing npu module, that will
  // complete the process (but it defines Python classes before calling back
  // into C, so these lines have to execute first)..
  THNPStream_init(module);
  THNPEvent_init(module);

  RegisterNPUDeviceProperties(module);
  BindGetDeviceProperties(module);
  torch::installCapturedTracebackPython();
  return module;
}
