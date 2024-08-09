#pragma once

#include <c10/core/Allocator.h>

#include "csrc/backend/NPUStream.h"
#include "csrc/core/Macros.h"

// TODO(FFFrog):
// Remove later
#include "core/NPUException.h"

namespace c10::backend::CachingHostAllocator {

c10::Allocator* getAllocator(void);

bool recordEvent(void* ptr, void* ctx, c10::backend::NPUStream stream);

void emptyCache(void);

// TODO(FFFrog): Remove
bool isPinndPtr(const void* ptr);

} // namespace c10::backend::CachingHostAllocator