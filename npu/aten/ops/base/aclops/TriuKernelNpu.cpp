// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "npu/aten/AclOpsInterface.h"
#include "npu/aten/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& triu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, int64_t k) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Triu")
      .Input(self)
      .Output(result)
      .Attr("diagonal", k)
      .Run();
  return result;
}
} // namespace

at::Tensor& triu_out(const at::Tensor& self, int64_t k, at::Tensor& result) {
  npu_preparation::CheckOut(
      {self},
      result,
      self);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    triu_out_npu_nocheck(contiguous_result, self, k);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    triu_out_npu_nocheck(result, self, k);
  }
  return result;
}

at::Tensor triu(const at::Tensor& self, int64_t k) {
  at::Tensor result = npu_preparation::apply_tensor(self);
  triu_out_npu_nocheck(result, self, k);
  return result;
}

at::Tensor& triu_(at::Tensor& self, int64_t k) {
  return acl_op::triu_out(self, k, self);
}
} // namespace acl_op