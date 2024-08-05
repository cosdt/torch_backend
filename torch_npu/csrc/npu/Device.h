#pragma once

#include <Python.h>
#include "csrc/core/Macros.h"

TORCH_BACKEND_API PyMethodDef* THNPModule_device_methods();
TORCH_BACKEND_API void RegisterNPUDeviceProperties(PyObject* module);
TORCH_BACKEND_API void BindGetDeviceProperties(PyObject* module);
