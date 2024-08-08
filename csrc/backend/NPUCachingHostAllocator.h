#pragma once

#include <c10/core/Allocator.h>

#include "csrc/backend/NPUStream.h"
#include "csrc/core/Macros.h"

// TODO(FFFrog):
// Remove later
#include "core/NPUException.h"

c10::Allocator* getNPUCachingHostAllocator(void);

bool NPUCachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::backend::NPUStream stream);

void NPUCachingHostAllocator_emptyCache(void);

// TODO(FFFrog): Remove
bool NPUCachingHostAllocator_isPinndPtr(const void* ptr);
