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
#include "npu/aten/OpApiInterface.h"
#include "npu/aten/utils/op_api_common.h"

namespace op_api {
at::Tensor& isin_out(const at::Scalar& element, const at::Tensor &test_element,
                     bool assume_unique, bool invert, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnIsInScalarTensor, acl_op::isin_out(element, test_element, assume_unique, invert, result));
  c10::SmallVector<int64_t, SIZE> shape_small_vec;
  at_npu::native::OpPreparation::check_tensor({test_element}, result, at::ScalarType::Bool, shape_small_vec);
  EXEC_NPU_CMD(aclnnIsInScalarTensor, element, test_element, assume_unique, invert, result);
  return result;
}
} // namespace op_api
