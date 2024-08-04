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
at::Tensor gelu_backward(const at::Tensor& grad, const at::Tensor& self, c10::string_view approximate) {
  return gelu_backward_common_nocheck(grad, self);
}
}  // namespace acl_op
