#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <string>

namespace torch_npu {

TORCH_BACKEND_API void _record_memory_history(
    c10::optional<std::string> enabled = "all",
    c10::optional<std::string> context = "all",
    std::string stacks = "all",
    size_t max_entries = UINT64_MAX);

TORCH_BACKEND_API std::string _memory_snapshot_pickled();

} // namespace torch_npu
