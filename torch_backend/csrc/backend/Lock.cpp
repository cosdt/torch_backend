#include "torch_backend/csrc/backend/Lock.h"
#include <pybind11/pybind11.h>
#include "csrc/backend/NPUFunctions.h"

namespace torch::backend::lock {

// We need to ensure that as long as a thread will NEVER loose the GIL as long
// as it holds the NPU mutex. Otherwise another thread might be scheduled and
// try to e.g. allocate a new tensor which will cause a deadlock. It's enough to
// have a single global, because it can be only set once (npuMutex is not
// recursive) by the thread that owns the mutex (obviously there can be only one
// such thread).
static PyGILState_STATE npuMutexGILState;

PyObject* THNPModule_npuLockMutex(PyObject* module, PyObject* noargs) {
  auto mutex = c10::backend::getFreeMutex();
  // This has to be a busy loop because we **absolutely need to** hold the GIL
  // or it's a recipe for a deadlock otherwise (if we let other Python threads
  // run while we have the cudaMutex, but not the GIL, they might try to e.g.
  // free a CUDA tensor and acquire the cudaMutex without giving up the GIL,
  // because it happens deep within THC).
  while (true) {
    if (mutex->try_lock()) {
      break;
    }
    {
      pybind11::gil_scoped_release no_gil;
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  npuMutexGILState = PyGILState_Ensure();
  Py_RETURN_NONE;
}

PyObject* THNPModule_npuUnlockMutex(PyObject* module, PyObject* noargs) {
  auto mutex = c10::backend::getFreeMutex();
  PyGILState_Release(npuMutexGILState);
  mutex->unlock();
  Py_RETURN_NONE;
}

static PyMethodDef THNPModule_methods[] = {
    {"_npu_lock_mutex",
     (PyCFunction)THNPModule_npuLockMutex,
     METH_NOARGS,
     nullptr},
    {"_npu_unlock_mutex",
     (PyCFunction)THNPModule_npuUnlockMutex,
     METH_NOARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return THNPModule_methods;
}

} // namespace torch::backend::lock
