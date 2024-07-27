#include <c10/core/Allocator.h>
#include <c10/util/SmallVector.h>

#include <npu/acl/include/acl/acl.h>
#include "csrc/npu/NPUStream.h"
#include "npu/core/NPUException.h"
#include "npu/core/NPUMacros.h"

c10::Allocator* getNPUCachingHostAllocator(void);

aclError NPUCachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10_npu::NPUStream stream);

bool NPUCachingHostAllocator_isPinndPtr(const void* ptr);
// Releases cached pinned memory allocations via npuHostFree
TORCH_NPU_API void NPUCachingHostAllocator_emptyCache(void);

c10::Allocator* getNPUPinnedMemoryAllocator(void);
