#ifndef __PULGIN_NATIVE_CONTIGUOUS_CONTIGUOUS_UTILS__
#define __PULGIN_NATIVE_CONTIGUOUS_CONTIGUOUS_UTILS__

#include <c10/util/SmallVector.h>

#include "framework/utils/NPUDefinition.h"
#include "framework/utils/NpuUtils.h"
#include "acl/include/acl/acl_base.h"

namespace at_npu {
namespace native {
// Max size of discontiguous cases vector
constexpr int MAX_CASES = 8;
// Max size of shape size
constexpr int MAX_DIM = 5;

// Define the discontiguous cases vector to be optimized
using OptimizationCases = c10::SmallVector<std::string, MAX_CASES>;

struct ContiguousTensorDesc {
  bool is_contiguous_;
  c10::SmallVector<int64_t, MAX_DIM> sizes_;
  c10::SmallVector<int64_t, MAX_DIM> strides_;
  int64_t offset_;
  c10::SmallVector<int64_t, MAX_DIM> base_sizes_;
  c10::SmallVector<int64_t, MAX_DIM> base_strides_;
  c10::SmallVector<int64_t, MAX_DIM> storage_sizes_;
  int64_t base_offset_;
  aclFormat npu_format_;
  OptimizationCases opt_cases_;
  void refresh_contiguous_using_size_and_stride();
  void reset_optimization_cases(const OptimizationCases& opt_cases);
  void add_optimization_case(const std::string& opt_case);
  void find_match_optimization_cases();
  size_t hash_src_desc;
  bool cached_contiguous;
};

} // namespace native
} // namespace at_npu

#endif
