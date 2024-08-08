#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>
#include "acl/include/acl/acl.h"
#include "c10/core/Device.h"
#include "c10/macros/Export.h"
#include "core/npu_log.h"
#include "csrc/core/Macros.h"

namespace c10::npu {

void TryInitDevice();

} // namespace c10::npu
