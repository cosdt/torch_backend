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
using npu_preparation = at_npu::native::OpPreparation;
using small_vector = c10::SmallVector<int64_t, op_infer::SIZE>;

static small_vector aclnn_lerp_broadcast_size(const at::Tensor &self, const at::Tensor &end, const at::Tensor &weight)
{
    auto expanded_size = op_infer::broadcast_ops_npu_output_size(self, end);
    auto output_size = op_infer::broadcast_ops_npu_output_size(expanded_size, weight.sizes());
    return output_size;
}

at::Tensor &lerp_out(const at::Tensor &self, const at::Tensor &end, const at::Tensor &weight, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnLerp, acl_op::lerp_out(self, end, weight, result));
    auto output_size = aclnn_lerp_broadcast_size(self, end, weight);
    npu_preparation::check_tensor({self, end, weight}, result, result.scalar_type(), output_size);
    EXEC_NPU_CMD(aclnnLerp, self, end, weight, result);
    return result;
}

at::Tensor &lerp_out(const at::Tensor &self, const at::Tensor &end, const at::Scalar &weight, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnLerps, acl_op::lerp_out(self, end, weight, result));
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, end);
    npu_preparation::check_tensor({self, end}, result, result.scalar_type(), output_size);
    EXEC_NPU_CMD(aclnnLerps, self, end, weight, result);
    return result;
}

at::Tensor lerp(const at::Tensor &self, const at::Tensor &end, const at::Tensor &weight)
{
    DO_COMPATIBILITY(aclnnLerp, acl_op::lerp(self, end, weight));
    auto output_size = aclnn_lerp_broadcast_size(self, end, weight);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
    EXEC_NPU_CMD(aclnnLerp, self, end, weight, result);
    return result;
}

at::Tensor lerp(const at::Tensor &self, const at::Tensor &end, const at::Scalar &weight)
{
    DO_COMPATIBILITY(aclnnLerps, acl_op::lerp(self, end, weight));
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, end);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
    EXEC_NPU_CMD(aclnnLerps, self, end, weight, result);
    return result;
}

at::Tensor &lerp_(at::Tensor &self, const at::Tensor &end, const at::Tensor &weight)
{
    DO_COMPATIBILITY(aclnnInplaceLerp, acl_op::lerp_(self, end, weight));
    EXEC_NPU_CMD(aclnnInplaceLerp, self, end, weight);
    return self;
}

at::Tensor &lerp_(at::Tensor &self, const at::Tensor &end, const at::Scalar &weight)
{
    DO_COMPATIBILITY(aclnnInplaceLerps, acl_op::lerp_(self, end, weight));
    EXEC_NPU_CMD(aclnnInplaceLerps, self, end, weight);
    return self;
}
} // namespace op_api