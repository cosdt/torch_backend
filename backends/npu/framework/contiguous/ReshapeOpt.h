#ifndef __PLUGIN_NATIVE_NPU_CONTIGUOUS_RESHAPE__
#define __PLUGIN_NATIVE_NPU_CONTIGUOUS_RESHAPE__

#include "aten/common/InnerNpuNativeFunction.h"
#include "csrc/backend/NPUCachingHostAllocator.h"
#include "framework/contiguous/ContiguousOpt.h"

namespace at_npu {
namespace native {

bool can_use_memecpy_for_NZ_format(const ContiguousTensorDesc&);
bool can_use_memcpy_for_other_format(const ContiguousTensorDesc&);
bool check_reshape_match_flex(
    const ContiguousTensorDesc&,
    const ContiguousTensorDesc&);
bool check_reshape_match(
    const ContiguousTensorDesc&,
    const ContiguousTensorDesc&);
bool check_reshape_match_flex(const ContiguousTensorDesc&);
bool check_reshape_match(const ContiguousTensorDesc&);
bool CanUseMemcpyForOtherFormat(const at::Tensor&);

} // namespace native
} // namespace at_npu

#endif
