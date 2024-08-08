#pragma once

#include <npu/acl/include/acl/acl.h>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "c10/core/Device.h"
#include "c10/macros/Export.h"
#include "csrc/core/Macros.h"
#include "npu/core/npu_log.h"

namespace c10::npu {

void TryInitDevice();

} // namespace c10::npu
