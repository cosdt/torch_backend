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
#include "npu/aten/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
at::Tensor& mean_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor& result) {
  return mean_out_common_nocheck(self, dim.value(), keepdim, dtype, result);
}

at::Tensor mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  return mean_common_nocheck(self, dim.value(), keepdim, dtype);
}
} // namespace acl_op