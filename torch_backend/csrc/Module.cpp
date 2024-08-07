#include <ATen/Parallel.h>
#include <Python.h>
#include <torch/csrc/profiler/python/combined_traceback.h>
#include <torch/csrc/utils.h>

#include "torch_backend/csrc/core/python_tensor.h"
#include "torch_backend/csrc/backend/Device.h"
#include "torch_backend/csrc/backend/Event.h"
#include "torch_backend/csrc/backend/Init.h"
#include "torch_backend/csrc/backend/Lock.h"
#include "torch_backend/csrc/backend/Memory.h"
#include "torch_backend/csrc/backend/Stream.h"

#define BACKEND_MODULE_NAME "torch_backend._C"

PyObject* module;
static std::vector<PyMethodDef> methods;

extern "C" C10_EXPORT PyObject* initModule();
PyObject* initModule() {
  at::internal::lazy_init_num_threads();

  THPUtils_addPyMethodDefs(methods, torch::backend::init::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::backend::lock::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::backend::device::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::backend::memory::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::backend::stream::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::backend::tensor::python_functions());

  static struct PyModuleDef torch_backend_module = {
      PyModuleDef_HEAD_INIT, BACKEND_MODULE_NAME, nullptr, -1, methods.data()};
  module = PyModule_Create(&torch_backend_module);

  torch::backend::stream::init(module);
  torch::backend::event::init(module);
  torch::backend::device::init(module);

  torch::installCapturedTracebackPython();
  return module;
}
