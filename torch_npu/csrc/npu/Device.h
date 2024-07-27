#pragma once

#include "csrc/core/Macros.h"

struct NPUDeviceProp {
  std::string name;
  size_t totalGlobalMem = 0;
};

TORCH_BACKEND_API void RegisterNPUDeviceProperties(PyObject* module);
TORCH_BACKEND_API void BindGetDeviceProperties(PyObject* module);

PyObject* THNPModule_getDevice_wrap(PyObject* self);
PyObject* THNPModule_setDevice_wrap(PyObject* self, PyObject* arg);
