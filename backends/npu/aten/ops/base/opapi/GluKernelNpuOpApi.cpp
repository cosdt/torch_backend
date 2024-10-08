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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &glu_out(const at::Tensor &self, int64_t dim, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnGlu, acl_op::glu_out(self, dim, result));
  auto output_size = op_infer::glu_npu_output_size(self, dim);

  // the dtype that does not check self must be equal to the result dtype
  npu_preparation::check_tensor({self}, result, output_size);

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnGlu, self, dim, result);
  return result;
}

at::Tensor glu(const at::Tensor &self, int64_t dim) {
  DO_COMPATIBILITY(aclnnGlu, acl_op::glu(self, dim));
  auto output_size = op_infer::glu_npu_output_size(self, dim);
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnGlu, self, dim, result);
  return result;
}
} // namespace op_api
