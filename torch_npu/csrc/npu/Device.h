#pragma once

#include "npu/core/NPUMacros.h"

struct NPUDeviceProp {
  std::string name;
  size_t totalGlobalMem = 0;
};

struct NPUDeviceMem {
  size_t totalGlobalMem = 0;
  size_t freeMem = 0;
};

TORCH_NPU_API void RegisterNPUDeviceProperties(PyObject* module);
TORCH_NPU_API void BindGetDeviceProperties(PyObject* module);
TORCH_NPU_API void RegisterNPUDeviceMemories(PyObject* module);

PyObject* THNPModule_getDevice_wrap(PyObject* self);
PyObject* THNPModule_setDevice_wrap(PyObject* self, PyObject* arg);
PyObject* THNPModule_getDeviceName_wrap(PyObject* self, PyObject* arg);
