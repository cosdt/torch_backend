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

void THNPStream_init(PyObject* module);
void THNPEvent_init(PyObject* module);

PyMethodDef* THNPModule_get_methods();
PyMethodDef* THNPModule_device_methods();

static std::vector<PyMethodDef> methods;

extern "C" C10_EXPORT PyObject* initModule();
PyObject* initModule() {
  at::internal::lazy_init_num_threads();

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
