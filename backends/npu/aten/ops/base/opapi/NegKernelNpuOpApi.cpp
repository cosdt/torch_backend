// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include "aten/AclOpsInterface.h"
#include "aten/OpApiInterface.h"
#include "aten/utils/op_api_common.h"

namespace op_api {

at::Tensor& neg_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnNeg, acl_op::neg_out(self, result));
  at_npu::native::OpPreparation::check_tensor({self}, result, self.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnNeg, self, result);
  return result;
}

at::Tensor neg(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnNeg, acl_op::neg(self));
  // construct the output tensor of the NPU
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self.sizes(), self.options());

  EXEC_NPU_CMD(aclnnNeg, self, result);
  return result;
}

at::Tensor& neg_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceNeg, acl_op::neg_(self));
  at_npu::native::OpPreparation::check_memory({self}, {self});

  EXEC_NPU_CMD(aclnnInplaceNeg, self);
  return self;
}
}  // namespace op_api
