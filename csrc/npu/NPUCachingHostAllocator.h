#pragma once

#include <c10/core/Allocator.h>

#include "csrc/core/Macros.h"
#include "csrc/npu/NPUStream.h"
#include "npu/core/NPUException.h"

c10::Allocator* getNPUCachingHostAllocator(void);

bool NPUCachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::npu::NPUStream stream);

void NPUCachingHostAllocator_emptyCache(void);

// TODO(FFFrog): Remove
bool NPUCachingHostAllocator_isPinndPtr(const void* ptr);
