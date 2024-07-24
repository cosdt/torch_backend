#ifndef THNP_NPU_MODULE_INC
#define THNP_NPU_MODULE_INC
#include "npu/core/NPUMacros.h"

TORCH_NPU_API void RegisterNpuPluggableAllocator(PyObject* module);
PyObject* THNPModule_getDriverVersion(PyObject* self);
PyObject* THNPModule_isDriverSufficient(PyObject* self);
PyObject* THNPModule_getCurrentBlasHandle_wrap(PyObject* self);

#define CHANGE_UNIT_SIZE 1024.0
#endif
