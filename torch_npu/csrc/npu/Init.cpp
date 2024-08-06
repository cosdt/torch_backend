#include "torch_npu/csrc/npu/Init.h"
#include <ATen/Context.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include "csrc/aten/generated/python_functions.h"
#include "csrc/npu/NPUFunctions.h"
#include "csrc/npu/NPUGeneratorImpl.h"

namespace torch::backend::init {

static PyObject* THNPModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    at::globalContext().lazyInitPrivateUse1();
  }
  auto m = THPObjectPtr(PyImport_ImportModule("torch.npu"));
  if (!m) {
    throw python_error();
  }

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };
  c10::DeviceIndex num_npus = c10::npu::device_count();
  auto default_npu_generators = PyTuple_New(static_cast<Py_ssize_t>(num_npus));
  for (c10::DeviceIndex i = 0; i < num_npus; i++) {
    auto gen = at_npu::detail::getDefaultNPUGenerator(i);
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_npu_generators, i, (PyObject*)cast_gen);
  }
  at_npu::autograd::generated::initialize_autogenerated_functions(m);
  set_module_attr("default_generators", default_npu_generators);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef THNPModule_methods[] = {
    {"_npu_init", (PyCFunction)THNPModule_initExtension, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return THNPModule_methods;
}

} // namespace torch::backend::init
