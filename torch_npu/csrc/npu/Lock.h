#pragma once

#include <Python.h>
#include "csrc/core/Macros.h"

TORCH_BACKEND_API PyMethodDef* THNPModule_lock_methods();
