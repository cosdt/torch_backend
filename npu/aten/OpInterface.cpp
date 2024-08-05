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

#include "npu/aten/AclOpsInterface.h"
#include "npu/aten/OpApiInterface.h"
#include "npu/aten/OpInterface.h"
#include "npu/aten/SparseOpsInterface.h"
#include "npu/framework/interface/EnvVariables.h"
#include "npu/framework/FormatHelper.h"

namespace op_plugin {
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _linalg_svd_out(const at::Tensor & A, bool full_matrices, bool compute_uv, c10::optional<c10::string_view> driver, at::Tensor & U, at::Tensor & S, at::Tensor & Vh){
    return acl_op::_linalg_svd_out(A, full_matrices, compute_uv, driver, U, S, Vh);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_out(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias) && at_npu::native::FormatHelper::IsOpInputBaseFormat(running_mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(running_var) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out) && at_npu::native::FormatHelper::IsOpInputBaseFormat(save_mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(save_invstd)) {
        return op_api::native_batch_norm_out(input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
    } else {
        return acl_op::native_batch_norm_out(input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
    }
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_apply_adam_out(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, c10::optional<bool> use_locking, c10::optional<bool> use_nesterov, at::Tensor & var, at::Tensor & m, at::Tensor & v){
    return acl_op::npu_apply_adam_out(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, var, m, v);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_apply_adam_w_out(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & weight_decay, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const c10::optional<at::Tensor> & max_grad_norm, c10::optional<bool> amsgrad, c10::optional<bool> maximize, at::Tensor & var, at::Tensor & m, at::Tensor & v){
    return acl_op::npu_apply_adam_w_out(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize, var, m, v);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_bert_apply_adam_out(const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const at::Scalar & max_grad_norm, const at::Scalar & global_grad_norm, const at::Scalar & weight_decay, const c10::optional<at::Scalar> & step_size, int64_t adam_mode, at::Tensor & var, at::Tensor & m, at::Tensor & v){
    return acl_op::npu_bert_apply_adam_out(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode, var, m, v);
}
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool2d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::adaptive_max_pool2d_out(self, output_size, out, indices);
    } else {
        return acl_op::adaptive_max_pool2d_out(self, output_size, out, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> aminmax_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & min, at::Tensor & max){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max)) {
        return op_api::aminmax_out(self, dim, keepdim, min, max);
    } else {
        return acl_op::aminmax_out(self, dim, keepdim, min, max);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out(const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::kthvalue_out(self, k, dim, keepdim, values, indices);
    } else {
        return acl_op::kthvalue_out(self, k, dim, keepdim, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out(const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::kthvalue_out(self, k, dim, keepdim, values, indices);
    } else {
        return acl_op::kthvalue_out(self, k, dim, keepdim, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_qr_out(const at::Tensor & self, c10::string_view mode, at::Tensor & Q, at::Tensor & R){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(Q) && at_npu::native::FormatHelper::IsOpInputBaseFormat(R)) {
        return op_api::linalg_qr_out(self, mode, Q, R);
    } else {
        return acl_op::linalg_qr_out(self, mode, Q, R);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> log_sigmoid_forward_out(const at::Tensor & self, at::Tensor & output, at::Tensor & buffer){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(buffer)) {
        return op_api::log_sigmoid_forward_out(self, output, buffer);
    } else {
        return acl_op::log_sigmoid_forward_out(self, output, buffer);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> max_out(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & max, at::Tensor & max_values){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max_values)) {
        return op_api::max_out(self, dim, keepdim, max, max_values);
    } else {
        return acl_op::max_out(self, dim, keepdim, max, max_values);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> max_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max_values)) {
        return op_api::max_out(self, dim, keepdim, max, max_values);
    } else {
        return acl_op::max_out(self, dim, keepdim, max, max_values);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool2d_with_indices_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::max_pool2d_with_indices_out(self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
    } else {
        return acl_op::max_pool2d_with_indices_out(self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool3d_with_indices_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices){
    return acl_op::max_pool3d_with_indices_out(self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> median_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::median_out(self, dim, keepdim, values, indices);
    } else {
        return acl_op::median_out(self, dim, keepdim, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> min_out(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min_indices)) {
        return op_api::min_out(self, dim, keepdim, min, min_indices);
    } else {
        return acl_op::min_out(self, dim, keepdim, min, min_indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> min_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min_indices)) {
        return op_api::min_out(self, dim, keepdim, min, min_indices);
    } else {
        return acl_op::min_out(self, dim, keepdim, min, min_indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> multilabel_margin_loss_forward_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & output, at::Tensor & is_target){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(is_target)) {
        return op_api::multilabel_margin_loss_forward_out(self, target, reduction, output, is_target);
    } else {
        return acl_op::multilabel_margin_loss_forward_out(self, target, reduction, output, is_target);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss2d_forward_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight)) {
        return op_api::nll_loss2d_forward_out(self, target, weight, reduction, ignore_index, output, total_weight);
    } else {
        return acl_op::nll_loss2d_forward_out(self, target, weight, reduction, ignore_index, output, total_weight);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight)) {
        return op_api::nll_loss_forward_out(self, target, weight, reduction, ignore_index, output, total_weight);
    } else {
        return acl_op::nll_loss_forward_out(self, target, weight, reduction, ignore_index, output, total_weight);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out(const at::Tensor & self, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::sort_out(self, dim, descending, values, indices);
    } else {
        return acl_op::sort_out(self, dim, descending, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices){
    return op_api::sort_out(self, stable, dim, descending, values, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::sort_out(self, dim, descending, values, indices);
    } else {
        return acl_op::sort_out(self, dim, descending, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> topk_out(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::topk_out(self, k, dim, largest, sorted, values, indices);
    } else {
        return acl_op::topk_out(self, k, dim, largest, sorted, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> triangular_solve_out(const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular, at::Tensor & X, at::Tensor & M){
    return acl_op::triangular_solve_out(self, A, upper, transpose, unitriangular, X, M);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention_backward(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & out_proj_weight, const c10::optional<at::Tensor> & query_bias, const c10::optional<at::Tensor> & key_bias, const c10::optional<at::Tensor> & value_bias, const c10::optional<at::Tensor> & out_proj_bias, const at::Tensor & query_res, const at::Tensor & key_res, const at::Tensor & value_res, const at::Tensor & attn_scores, const at::Tensor & attn_res, const at::Tensor & context, const at::Tensor & y_grad, const at::Tensor & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float){
    return acl_op::npu_multi_head_attention_backward(query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, query_res, key_res, value_res, attn_scores, attn_res, context, y_grad, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction){
    return acl_op::npu_lstm(input, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_cell(const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh){
    return acl_op::npu_lstm_cell(input, w_ih, w_hh, h, c, b_ih, b_hh);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_data(const at::Tensor & input, const at::Tensor & batch_sizes, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction){
    return acl_op::npu_lstm_data(input, batch_sizes, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & attn_mask, const at::Tensor & out_proj_weight, const c10::optional<at::Tensor> & query_bias, const c10::optional<at::Tensor> & key_bias, const c10::optional<at::Tensor> & value_bias, const c10::optional<at::Tensor> & out_proj_bias, const c10::optional<at::Tensor> & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float){
    return acl_op::npu_multi_head_attention(query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_cell_backward(const c10::optional<at::Tensor> & grady, const c10::optional<at::Tensor> & gradh, const c10::optional<at::Tensor> & gradc, const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & c_output, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc){
    return acl_op::npu_lstm_cell_backward(grady, gradh, gradc, input, w_ih, w_hh, h, c, y_output, h_output, c_output, i, j, f, o, tanhc);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_gru(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first){
    return acl_op::npu_gru(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_gru_backward(const c10::optional<at::Tensor> & grady, const c10::optional<at::Tensor> & gradh, const at::Tensor & input, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, const at::Tensor & hx, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & output_updata, const at::Tensor & output_reset, const at::Tensor & output_new, const at::Tensor & hidden_new){
    return acl_op::npu_gru_backward(grady, gradh, input, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, hx, y_output, h_output, output_updata, output_reset, output_new, hidden_new);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_backward(const c10::optional<at::Tensor> & grady, const c10::optional<at::Tensor> & gradh, const c10::optional<at::Tensor> & gradc, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & hx, const at::Tensor & cx, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & c_output, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc){
    return acl_op::npu_lstm_backward(grady, gradh, gradc, input, weight, bias, hx, cx, y_output, h_output, c_output, i, j, f, o, tanhc);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_data_backward(const c10::optional<at::Tensor> & grady_opt, const c10::optional<at::Tensor> & gradh_opt, const c10::optional<at::Tensor> & gradc_opt, const at::Tensor & input, const at::Tensor & batch_sizes, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & init_h, const at::Tensor & init_c, const at::Tensor & y, const at::Tensor & h, const at::Tensor & c, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc, bool flag_direction){
    return acl_op::npu_lstm_data_backward(grady_opt, gradh_opt, gradc_opt, input, batch_sizes, weight, bias, init_h, init_c, y, h, c, i, j, f, o, tanhc, flag_direction);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_fusion_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t head_num, c10::string_view input_layout, const c10::optional<at::Tensor> & pse, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync){
    return op_api::npu_fusion_attention(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, scale, keep_prob, pre_tockens, next_tockens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t> _batch_norm_impl_index(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled){
    return acl_op::_batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices) && at_npu::native::FormatHelper::IsOpInputBaseFormat(offsets) && at_npu::native::FormatHelper::IsOpInputBaseFormat(per_sample_weights)) {
        return op_api::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
    } else {
        return acl_op::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag_forward_only(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx){
    return acl_op::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> batch_norm_backward_reduce(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(invstd) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
    } else {
        return acl_op::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_layer_norm(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const at::Tensor & beta, double epsilon, bool additional_output){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(x1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(x2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma) && at_npu::native::FormatHelper::IsOpInputBaseFormat(beta)) {
        return op_api::npu_add_layer_norm(x1, x2, gamma, beta, epsilon, additional_output);
    } else {
        return acl_op::npu_add_layer_norm(x1, x2, gamma, beta, epsilon, additional_output);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_layer_norm_backward(const c10::optional<at::Tensor> & dy_opt, const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & rstd, const at::Tensor & mean, const at::Tensor & gamma, const c10::optional<at::Tensor> & dsum_opt){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(dy_opt) && at_npu::native::FormatHelper::IsOpInputBaseFormat(x1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(x2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma) && at_npu::native::FormatHelper::IsOpInputBaseFormat(dsum_opt)) {
        return op_api::npu_add_layer_norm_backward(dy_opt, x1, x2, rstd, mean, gamma, dsum_opt);
    } else {
        return acl_op::npu_add_layer_norm_backward(dy_opt, x1, x2, rstd, mean, gamma, dsum_opt);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_batch_nms(const at::Tensor & self, const at::Tensor & scores, double score_threshold, double iou_threshold, int64_t max_size_per_class, int64_t max_total_size, bool change_coordinate_frame, bool transpose_box){
    return acl_op::npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame, transpose_box);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_deep_norm_backward(const at::Tensor & dy, const at::Tensor & x, const at::Tensor & gx, const at::Tensor & gamma, const at::Tensor & mean, const at::Tensor & rstd, double alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(dy) && at_npu::native::FormatHelper::IsOpInputBaseFormat(x) && at_npu::native::FormatHelper::IsOpInputBaseFormat(gx) && at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd)) {
        return op_api::npu_deep_norm_backward(dy, x, gx, gamma, mean, rstd, alpha);
    } else {
        return acl_op::npu_deep_norm_backward(dy, x, gx, gamma, mean, rstd, alpha);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_deformable_conv2dbk(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & offset_out, const at::Tensor & weight, const at::Tensor & offset, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated){
    return acl_op::npu_deformable_conv2dbk(input, grad_output, offset_out, weight, offset, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_fusion_attention_grad(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & dy, int64_t head_num, c10::string_view input_layout, const c10::optional<at::Tensor> & pse, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & softmax_max, const c10::optional<at::Tensor> & softmax_sum, const c10::optional<at::Tensor> & softmax_in, const c10::optional<at::Tensor> & attention_in, double scale_value, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, int64_t seed, int64_t offset, int64_t numels, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync){
    return op_api::npu_fusion_attention_grad(query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, scale_value, keep_prob, pre_tockens, next_tockens, inner_precise, seed, offset, numels, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _batch_norm_impl_index_backward(int64_t impl_index, const at::Tensor & input, const at::Tensor & grad_output, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var_transform, bool train, double eps, ::std::array<bool,3> output_mask, const at::Tensor & reservedSpace){
    return acl_op::_batch_norm_impl_index_backward(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _linalg_svd(const at::Tensor & A, bool full_matrices, bool compute_uv, c10::optional<c10::string_view> driver){
    return acl_op::_linalg_svd(A, full_matrices, compute_uv, driver);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, bool training, double momentum, double eps){
    return acl_op::_native_batch_norm_legit(input, weight, bias, running_mean, running_var, training, momentum, eps);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _slow_conv2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, ::std::array<bool,3> output_mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::_slow_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_mask);
    } else {
        return acl_op::_slow_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _unique2(const at::Tensor & self, bool sorted, bool return_inverse, bool return_counts){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::_unique2(self, sorted, return_inverse, return_counts);
    } else {
        return acl_op::_unique2(self, sorted, return_inverse, return_counts);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> conv_tbc_backward(const at::Tensor & self, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, int64_t pad){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::conv_tbc_backward(self, input, weight, bias, pad);
    } else {
        return acl_op::conv_tbc_backward(self, input, weight, bias, pad);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalIntArrayRef bias_sizes, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask);
    } else {
        return acl_op::convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
    } else {
        return acl_op::convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm(const at::Tensor & data, const at::Tensor & batch_sizes, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional){
    return acl_op::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm(const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first){
    return acl_op::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> matmul_double_backward(const c10::optional<at::Tensor> & grad_self, const c10::optional<at::Tensor> & grad_other, const at::Tensor & grad_out, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,3> mask){
    return op_api::matmul_double_backward(grad_self, grad_other, grad_out, self, other, mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias) && at_npu::native::FormatHelper::IsOpInputBaseFormat(running_mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(running_var)) {
        return op_api::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
    } else {
        return acl_op::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(running_mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(running_var) && at_npu::native::FormatHelper::IsOpInputBaseFormat(save_mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(save_invstd)) {
        return op_api::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
    } else {
        return acl_op::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::native_group_norm(input, weight, bias, N, C, HxW, group, eps);
    } else {
        return acl_op::native_group_norm(input, weight, bias, N, C, HxW, group, eps);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, ::std::array<bool,3> output_mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
    } else {
        return acl_op::native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::native_layer_norm(input, normalized_shape, weight, bias, eps);
    } else {
        return acl_op::native_layer_norm(input, normalized_shape, weight, bias, eps);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
    } else {
        return acl_op::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_add_rms_norm(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, double epsilon){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(x1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(x2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma)) {
        return op_api::npu_add_rms_norm(x1, x2, gamma, epsilon);
    } else {
        return acl_op::npu_add_rms_norm(x1, x2, gamma, epsilon);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_apply_adam(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, c10::optional<bool> use_locking, c10::optional<bool> use_nesterov){
    return acl_op::npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_apply_adam_w(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & weight_decay, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const c10::optional<at::Tensor> & max_grad_norm, c10::optional<bool> amsgrad, c10::optional<bool> maximize){
    return acl_op::npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_bert_apply_adam(const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const at::Scalar & max_grad_norm, const at::Scalar & global_grad_norm, const at::Scalar & weight_decay, const c10::optional<at::Scalar> & step_size, int64_t adam_mode){
    return acl_op::npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv2d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask){
    return acl_op::npu_conv2d_backward(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv3d_backward(const at::Tensor & input, const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask){
    return acl_op::npu_conv3d_backward(input, grad, weight, stride, padding, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv_transpose2d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask){
    return acl_op::npu_conv_transpose2d_backward(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv_transpose3d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask){
    return acl_op::npu_conv_transpose3d_backward(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_convolution_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask){
    return acl_op::npu_convolution_backward(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_convolution_transpose_backward(const at::Tensor & input, const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> grad_input_mask){
    return acl_op::npu_convolution_transpose_backward(input, grad, weight, padding, output_padding, stride, dilation, groups, grad_input_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_deep_norm(const at::Tensor & x, const at::Tensor & gx, const at::Tensor & beta, const at::Tensor & gamma, double alpha, double epsilon){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(x) && at_npu::native::FormatHelper::IsOpInputBaseFormat(gx) && at_npu::native::FormatHelper::IsOpInputBaseFormat(beta) && at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma)) {
        return op_api::npu_deep_norm(x, gx, beta, gamma, alpha, epsilon);
    } else {
        return acl_op::npu_deep_norm(x, gx, beta, gamma, alpha, epsilon);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_dropout_with_add_softmax(const at::Tensor & self, const at::Tensor & x1, const at::Scalar & alpha, double prob, int64_t dim){
    return acl_op::npu_dropout_with_add_softmax(self, x1, alpha, prob, dim);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_backward(const at::Tensor & grad_output, const at::Tensor & softmax_output, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool value_transpose, bool dx_transpose){
    return acl_op::npu_fused_attention_score_backward(grad_output, softmax_output, query_layer, key_layer, value_layer, mask, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_fwd(const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose){
    return acl_op::npu_fused_attention_score_fwd(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_grad(const at::Tensor & grad_output, const at::Tensor & softmax_output, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool value_transpose, bool dx_transpose){
    return acl_op::npu_fused_attention_score_grad(grad_output, softmax_output, query_layer, key_layer, value_layer, mask, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_silu(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t group, double eps){
    return op_api::npu_group_norm_silu(input, weight, bias, group, eps);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_layernorm_grad(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias){
    return acl_op::npu_layernorm_grad(grad_out, input, normalized_shape, mean, rstd, weight, bias);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_gating_top_k_softmax(const at::Tensor & x, const c10::optional<at::Tensor> & finished, int64_t k){
    return op_api::npu_moe_gating_top_k_softmax(x, finished, k);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_init_routing(const at::Tensor & x, const at::Tensor & row_idx, const at::Tensor & expert_idx, int64_t active_num){
    return op_api::npu_moe_init_routing(x, row_idx, expert_idx, active_num);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention_v2_grad(const at::Tensor & attention_score_grad, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & softmax_log_max_sum, const at::Tensor & attention_score, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & alibi_mask, double scale, int64_t head_num, c10::string_view input_layout, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t seed, int64_t offset, int64_t numels, bool gen_mask_parallel, bool sync){
    return op_api::npu_multi_head_attention_v2_grad(attention_score_grad, query, key, value, softmax_log_max_sum, attention_score, atten_mask, alibi_mask, scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens, seed, offset, numels, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nms_with_mask(const at::Tensor & input, const at::Scalar & iou_threshold){
    return acl_op::npu_nms_with_mask(input, iou_threshold);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_rotary_mul_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2){
    return acl_op::npu_rotary_mul_backward(grad, self, r1, r2);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_dilated2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::slow_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
    } else {
        return acl_op::slow_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_transpose2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::slow_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, output_mask);
    } else {
        return acl_op::slow_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_consecutive(const at::Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::unique_consecutive(self, return_inverse, return_counts, dim);
    } else {
        return acl_op::unique_consecutive(self, return_inverse, return_counts, dim);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim(const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::unique_dim(self, dim, sorted, return_inverse, return_counts);
    } else {
        return acl_op::unique_dim(self, dim, sorted, return_inverse, return_counts);
    }
}
::std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_multi_head_attention_v2(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & alibi_mask, double scale, int64_t head_num, c10::string_view input_layout, double keep_prob, int64_t pre_tokens, int64_t next_tokens, bool gen_mask_parallel, bool sync){
    return op_api::npu_multi_head_attention_v2(query, key, value, atten_mask, alibi_mask, scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor> _aminmax(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::_aminmax(self);
    } else {
        return acl_op::_aminmax(self);
    }
}
::std::tuple<at::Tensor,at::Tensor> _aminmax(const at::Tensor & self, int64_t dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::_aminmax(self, dim, keepdim);
    } else {
        return acl_op::_aminmax(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> _conv_depthwise2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,2> output_mask){
    return acl_op::_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
::std::tuple<at::Tensor,at::Tensor> _ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool zero_infinity){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(log_probs) && at_npu::native::FormatHelper::IsOpInputBaseFormat(targets)) {
        return op_api::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
    } else {
        return acl_op::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
    }
}
::std::tuple<at::Tensor,at::Tensor> _dropout_with_byte_mask(const at::Tensor & self, double p){
    return acl_op::_dropout_with_byte_mask(self, p);
}
::std::tuple<at::Tensor,at::Tensor> _npu_ciou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag){
    return acl_op::_npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}
::std::tuple<at::Tensor,at::Tensor> _npu_dropout(const at::Tensor & self, double p){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::_npu_dropout(self, p);
    } else {
        return acl_op::_npu_dropout(self, p);
    }
}
::std::tuple<at::Tensor,at::Tensor> _pack_padded_sequence(const at::Tensor & input, const at::Tensor & lengths, bool batch_first){
    return acl_op::_pack_padded_sequence(input, lengths, batch_first);
}
::std::tuple<at::Tensor,at::Tensor> _pad_packed_sequence(const at::Tensor & data, const at::Tensor & batch_sizes, bool batch_first, const at::Scalar & padding_value, int64_t total_length){
    return acl_op::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
}
::std::tuple<at::Tensor,at::Tensor> _prelu_kernel_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::_prelu_kernel_backward(grad_output, self, weight);
    } else {
        return acl_op::_prelu_kernel_backward(grad_output, self, weight);
    }
}
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool2d(const at::Tensor & self, at::IntArrayRef output_size){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::adaptive_max_pool2d(self, output_size);
    } else {
        return acl_op::adaptive_max_pool2d(self, output_size);
    }
}
::std::tuple<at::Tensor,at::Tensor> aminmax(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim){
    return op_api::aminmax(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_update(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts){
    return acl_op::batch_norm_gather_stats_update(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_with_counts(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(invstd) && at_npu::native::FormatHelper::IsOpInputBaseFormat(running_mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(running_var) && at_npu::native::FormatHelper::IsOpInputBaseFormat(counts)) {
        return op_api::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
    } else {
        return acl_op::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
    }
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_reduce(const at::Tensor & input, double eps){
    return acl_op::batch_norm_reduce(input, eps);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_stats(const at::Tensor & input, double eps){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input)) {
        return op_api::batch_norm_stats(input, eps);
    } else {
        return acl_op::batch_norm_stats(input, eps);
    }
}
::std::tuple<at::Tensor,at::Tensor> grid_sampler_2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grid)) {
        return op_api::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
    } else {
        return acl_op::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor> grid_sampler_3d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grid)) {
        return op_api::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
    } else {
        return acl_op::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor> gru(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first){
    return acl_op::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
::std::tuple<at::Tensor,at::Tensor> kthvalue(const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::kthvalue(self, k, dim, keepdim);
    } else {
        return acl_op::kthvalue(self, k, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> kthvalue(const at::Tensor & self, int64_t k, int64_t dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::kthvalue(self, k, dim, keepdim);
    } else {
        return acl_op::kthvalue(self, k, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> linalg_qr(const at::Tensor & self, c10::string_view mode){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::linalg_qr(self, mode);
    } else {
        return acl_op::linalg_qr(self, mode);
    }
}
::std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::log_sigmoid_forward(self);
    } else {
        return acl_op::log_sigmoid_forward(self);
    }
}
::std::tuple<at::Tensor,at::Tensor> lstm_cell(const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh){
    return acl_op::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
}
::std::tuple<at::Tensor,at::Tensor> matmul_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,2> mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::matmul_backward(grad, self, other, mask);
    } else {
        return acl_op::matmul_backward(grad, self, other, mask);
    }
}
::std::tuple<at::Tensor,at::Tensor> max(const at::Tensor & self, at::Dimname dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::max(self, dim, keepdim);
    } else {
        return acl_op::max(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> max(const at::Tensor & self, int64_t dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::max(self, dim, keepdim);
    } else {
        return acl_op::max(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> max_pool2d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    } else {
        return acl_op::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    }
}
::std::tuple<at::Tensor,at::Tensor> max_pool3d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode){
    return acl_op::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}
::std::tuple<at::Tensor,at::Tensor> median(const at::Tensor & self, int64_t dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::median(self, dim, keepdim);
    } else {
        return acl_op::median(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> min(const at::Tensor & self, at::Dimname dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::min(self, dim, keepdim);
    } else {
        return acl_op::min(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> min(const at::Tensor & self, int64_t dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::min(self, dim, keepdim);
    } else {
        return acl_op::min(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> multilabel_margin_loss_forward(const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::multilabel_margin_loss_forward(self, target, reduction);
    } else {
        return acl_op::multilabel_margin_loss_forward(self, target, reduction);
    }
}
::std::tuple<at::Tensor,at::Tensor> nanmedian(const at::Tensor & self, int64_t dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::nanmedian(self, dim, keepdim);
    } else {
        return acl_op::nanmedian(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> native_dropout(const at::Tensor & input, double p, c10::optional<bool> train){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input)) {
        return op_api::native_dropout(input, p, train);
    } else {
        return acl_op::native_dropout(input, p, train);
    }
}
::std::tuple<at::Tensor,at::Tensor> nll_loss2d_forward(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::nll_loss2d_forward(self, target, weight, reduction, ignore_index);
    } else {
        return acl_op::nll_loss2d_forward(self, target, weight, reduction, ignore_index);
    }
}
::std::tuple<at::Tensor,at::Tensor> nll_loss_forward(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::nll_loss_forward(self, target, weight, reduction, ignore_index);
    } else {
        return acl_op::nll_loss_forward(self, target, weight, reduction, ignore_index);
    }
}
::std::tuple<at::Tensor,at::Tensor> npu_all_gather_base_mm(const at::Tensor & self, const at::Tensor & x2, c10::string_view hcom, int64_t world_size, const c10::optional<at::Tensor> & bias, int64_t gather_index, bool gather_output, int64_t comm_turn){
    return op_api::npu_all_gather_base_mm(self, x2, hcom, world_size, bias, gather_index, gather_output, comm_turn);
}
::std::tuple<at::Tensor,at::Tensor> npu_apply_rotary_pos_emb(const at::Tensor & query, const at::Tensor & key, const at::Tensor & cos, const at::Tensor & sin, c10::string_view layout){
    return op_api::npu_apply_rotary_pos_emb(query, key, cos, sin, layout);
}
::std::tuple<at::Tensor,at::Tensor> npu_ciou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, const c10::optional<at::Tensor> & atan_sub, bool trans, bool is_cross, int64_t mode){
    return acl_op::npu_ciou_backward(grad, bboxes, gtboxes, atan_sub, trans, is_cross, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_deformable_conv2d(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & offset, const c10::optional<at::Tensor> & bias, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated){
    return acl_op::npu_deformable_conv2d(input, weight, offset, bias, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
}
::std::tuple<at::Tensor,at::Tensor> npu_diou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode){
    return acl_op::npu_diou_backward(grad, bboxes, gtboxes, trans, is_cross, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_dropout_do_mask(const at::Tensor & self, const at::Tensor & mask, double p){
    return acl_op::npu_dropout_do_mask(self, mask, p);
}
::std::tuple<at::Tensor,at::Tensor> npu_dropout_with_add_softmax_backward(const at::Tensor & grad, const at::Tensor & mask, const at::Tensor & softmax_out, const at::Scalar & alpha, double prob, int64_t dim){
    return acl_op::npu_dropout_with_add_softmax_backward(grad, mask, softmax_out, alpha, prob, dim);
}
::std::tuple<at::Tensor,at::Tensor> npu_geglu(const at::Tensor & self, int64_t dim, int64_t approximate, bool activate_left){
    return op_api::npu_geglu(self, dim, approximate, activate_left);
}
::std::tuple<at::Tensor,at::Tensor> npu_giou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode){
    return acl_op::npu_giou_backward(grad, bboxes, gtboxes, trans, is_cross, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_ifmr(const at::Tensor & data, const at::Tensor & data_min, const at::Tensor & data_max, const at::Tensor & cumsum, double min_percentile, double max_percentile, double search_start, double search_end, double search_step, bool with_offset){
    return acl_op::npu_ifmr(data, data_min, data_max, cumsum, min_percentile, max_percentile, search_start, search_end, search_step, with_offset);
}
::std::tuple<at::Tensor,at::Tensor> npu_linear_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::npu_linear_backward(grad, input, weight);
    } else {
        return acl_op::npu_linear_backward(grad, input, weight);
    }
}
::std::tuple<at::Tensor,at::Tensor> npu_max(const at::Tensor & self, at::Dimname dim, bool keepdim){
    return acl_op::npu_max(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_max(const at::Tensor & self, int64_t dim, bool keepdim){
    return acl_op::npu_max(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_min(const at::Tensor & self, at::Dimname dim, bool keepdim){
    return acl_op::npu_min(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_min(const at::Tensor & self, int64_t dim, bool keepdim){
    return acl_op::npu_min(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_nms_rotated(const at::Tensor & self, const at::Tensor & scores, double iou_threshold, double scores_threshold, int64_t max_output_size, int64_t mode){
    return acl_op::npu_nms_rotated(self, scores, iou_threshold, scores_threshold, max_output_size, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_nms_v4(const at::Tensor & self, const at::Tensor & scores, const at::Scalar & max_output_size, const at::Tensor & iou_threshold, const at::Tensor & scores_threshold, bool pad_to_max_output_size){
    return acl_op::npu_nms_v4(self, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size);
}
::std::tuple<at::Tensor,at::Tensor> npu_random_choice_with_mask(const at::Tensor & x, int64_t count, int64_t seed, int64_t seed2){
    return acl_op::npu_random_choice_with_mask(x, count, seed, seed2);
}
::std::tuple<at::Tensor,at::Tensor> npu_rms_norm(const at::Tensor & self, const at::Tensor & gamma, double epsilon){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma)) {
        return op_api::npu_rms_norm(self, gamma, epsilon);
    } else {
        return acl_op::npu_rms_norm(self, gamma, epsilon);
    }
}
::std::tuple<at::Tensor,at::Tensor> npu_rms_norm_backward(const at::Tensor & dy, const at::Tensor & self, const at::Tensor & gamma, const at::Tensor & rstd){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(dy) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma) && at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd)) {
        return op_api::npu_rms_norm_backward(dy, self, gamma, rstd);
    } else {
        return acl_op::npu_rms_norm_backward(dy, self, gamma, rstd);
    }
}
::std::tuple<at::Tensor,at::Tensor> slogdet(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::slogdet(self);
    } else {
        return acl_op::slogdet(self);
    }
}
::std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, at::Dimname dim, bool descending){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sort(self, dim, descending);
    } else {
        return acl_op::sort(self, dim, descending);
    }
}
::std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending){
    return op_api::sort(self, stable, dim, descending);
}
::std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, int64_t dim, bool descending){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sort(self, dim, descending);
    } else {
        return acl_op::sort(self, dim, descending);
    }
}
::std::tuple<at::Tensor,at::Tensor> std_mean(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::std_mean(self, dim, correction, keepdim);
    } else {
        return acl_op::std_mean(self, dim, correction, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> topk(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::topk(self, k, dim, largest, sorted);
    } else {
        return acl_op::topk(self, k, dim, largest, sorted);
    }
}
::std::tuple<at::Tensor,at::Tensor> var_mean(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::var_mean(self, dim, correction, keepdim);
    } else {
        return acl_op::var_mean(self, dim, correction, keepdim);
    }
}
::std::vector<at::Tensor> _foreach_add(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_add(tensors, scalars);
}
::std::vector<at::Tensor> _foreach_add(at::TensorList tensors, const at::Scalar & scalar){
    return op_api::_foreach_add(tensors, scalar);
}
::std::vector<at::Tensor> _foreach_add(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha){
    return op_api::_foreach_add(tensors1, tensors2, alpha);
}
::std::vector<at::Tensor> _foreach_addcdiv(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_addcdiv(input, tensor1, tensor2, scalars);
}
::std::vector<at::Tensor> _foreach_addcdiv(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value){
    return op_api::_foreach_addcdiv(input, tensor1, tensor2, value);
}
::std::vector<at::Tensor> _foreach_addcdiv(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars){
    return op_api::_foreach_addcdiv(self, tensor1, tensor2, scalars);
}
::std::vector<at::Tensor> _foreach_addcmul(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_addcmul(input, tensor1, tensor2, scalars);
}
::std::vector<at::Tensor> _foreach_addcmul(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value){
    return op_api::_foreach_addcmul(input, tensor1, tensor2, value);
}
::std::vector<at::Tensor> _foreach_addcmul(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars){
    return op_api::_foreach_addcmul(self, tensor1, tensor2, scalars);
}
::std::vector<at::Tensor> _foreach_ceil(at::TensorList self){
    return op_api::_foreach_ceil(self);
}
::std::vector<at::Tensor> _foreach_cos(at::TensorList tensors){
    return op_api::_foreach_cos(tensors);
}
::std::vector<at::Tensor> _foreach_div(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_div(self, scalar);
}
::std::vector<at::Tensor> _foreach_div(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_div(tensors, scalars);
}
::std::vector<at::Tensor> _foreach_div(at::TensorList tensors1, at::TensorList tensors2){
    return op_api::_foreach_div(tensors1, tensors2);
}
::std::vector<at::Tensor> _foreach_exp(at::TensorList tensors){
    return op_api::_foreach_exp(tensors);
}
::std::vector<at::Tensor> _foreach_floor(at::TensorList self){
    return op_api::_foreach_floor(self);
}
::std::vector<at::Tensor> _foreach_frac(at::TensorList self){
    return op_api::_foreach_frac(self);
}
::std::vector<at::Tensor> _foreach_maximum(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_maximum(self, scalars);
}
::std::vector<at::Tensor> _foreach_maximum(at::TensorList self, at::TensorList other){
    return op_api::_foreach_maximum(self, other);
}
::std::vector<at::Tensor> _foreach_maximum(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_maximum(self, scalar);
}
::std::vector<at::Tensor> _foreach_minimum(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_minimum(self, scalars);
}
::std::vector<at::Tensor> _foreach_minimum(at::TensorList self, at::TensorList other){
    return op_api::_foreach_minimum(self, other);
}
::std::vector<at::Tensor> _foreach_minimum(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_minimum(self, scalar);
}
::std::vector<at::Tensor> _foreach_mul(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_mul(tensors, scalars);
}
::std::vector<at::Tensor> _foreach_mul(at::TensorList tensors, const at::Scalar & scalar){
    return op_api::_foreach_mul(tensors, scalar);
}
::std::vector<at::Tensor> _foreach_mul(at::TensorList tensors1, at::TensorList tensors2){
    return op_api::_foreach_mul(tensors1, tensors2);
}
::std::vector<at::Tensor> _foreach_neg(at::TensorList tensors){
    return op_api::_foreach_neg(tensors);
}
::std::vector<at::Tensor> _foreach_pow(at::TensorList self, at::ArrayRef<at::Scalar> exponent){
    return op_api::_foreach_pow(self, exponent);
}
::std::vector<at::Tensor> _foreach_pow(at::TensorList self, at::TensorList exponent){
    return op_api::_foreach_pow(self, exponent);
}
::std::vector<at::Tensor> _foreach_pow(at::TensorList tensors, const at::Scalar & scalar){
    return op_api::_foreach_pow(tensors, scalar);
}
::std::vector<at::Tensor> _foreach_round(at::TensorList self){
    return op_api::_foreach_round(self);
}
::std::vector<at::Tensor> _foreach_sigmoid(at::TensorList tensors){
    return op_api::_foreach_sigmoid(tensors);
}
::std::vector<at::Tensor> _foreach_sqrt(at::TensorList tensors){
    return op_api::_foreach_sqrt(tensors);
}
::std::vector<at::Tensor> _foreach_sub(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_sub(self, scalar);
}
::std::vector<at::Tensor> _foreach_sub(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_sub(tensors, scalars);
}
::std::vector<at::Tensor> _foreach_sub(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha){
    return op_api::_foreach_sub(tensors1, tensors2, alpha);
}
::std::vector<at::Tensor> _foreach_trunc(at::TensorList self){
    return op_api::_foreach_trunc(self);
}
::std::vector<at::Tensor> npu_fused_attention_layernorm_qkv_fwd(const at::Tensor & x, const at::Tensor & kernel_query, const at::Tensor & kernel_key, const at::Tensor & kernel_value, const at::Tensor & gamma, const at::Tensor & beta, const c10::optional<at::Tensor> & bias_query, const c10::optional<at::Tensor> & bias_key, const c10::optional<at::Tensor> & bias_value, int64_t seq_len, int64_t num_heads, double eps){
    return acl_op::npu_fused_attention_layernorm_qkv_fwd(x, kernel_query, kernel_key, kernel_value, gamma, beta, bias_query, bias_key, bias_value, seq_len, num_heads, eps);
}
::std::vector<at::Tensor> npu_fused_attention_qkv_grad(const at::Tensor & grad_output_query, const at::Tensor & grad_output_key, const at::Tensor & grad_output_value, const at::Tensor & query_kernel, const at::Tensor & key_kernel, const at::Tensor & value_kernel, const at::Tensor & hidden_states, const at::Tensor & grad_output_ln){
    return acl_op::npu_fused_attention_qkv_grad(grad_output_query, grad_output_key, grad_output_value, query_kernel, key_kernel, value_kernel, hidden_states, grad_output_ln);
}
::std::vector<at::Tensor> npu_grouped_matmul(at::TensorList x, at::TensorList weight, c10::optional<at::TensorList> bias, c10::optional<at::TensorList> scale, c10::optional<at::TensorList> offset, c10::optional<at::TensorList> antiquant_scale, c10::optional<at::TensorList> antiquant_offset, at::OptionalIntArrayRef group_list, c10::optional<int64_t> split_item, c10::optional<at::ScalarType> output_dtype){
    return op_api::npu_grouped_matmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list, split_item, output_dtype);
}
::std::vector<at::Tensor> npu_scatter_list(at::TensorList self, const at::Tensor & indices, const at::Tensor & updates, const c10::optional<at::Tensor> & mask, c10::string_view reduce, int64_t axis){
    return op_api::npu_scatter_list(self, indices, updates, mask, reduce, axis);
}
::std::vector<at::Tensor> where(const at::Tensor & condition){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(condition)) {
        return op_api::where(condition);
    } else {
        return acl_op::where(condition);
    }
}
at::Tensor & __ilshift__(at::Tensor & self, const at::Scalar & other){
    return acl_op::__ilshift__(self, other);
}
at::Tensor & __ilshift__(at::Tensor & self, const at::Tensor & other){
    return acl_op::__ilshift__(self, other);
}
at::Tensor & __ior__(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::__ior__(self, other);
    } else {
        return acl_op::__ior__(self, other);
    }
}
at::Tensor & __ior__(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::__ior__(self, other);
    } else {
        return acl_op::__ior__(self, other);
    }
}
at::Tensor & __irshift__(at::Tensor & self, const at::Scalar & other){
    return acl_op::__irshift__(self, other);
}
at::Tensor & __irshift__(at::Tensor & self, const at::Tensor & other){
    return acl_op::__irshift__(self, other);
}
at::Tensor & _add_relu_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    return acl_op::_add_relu_(self, other, alpha);
}
at::Tensor & _add_relu_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out){
    return acl_op::_add_relu_out(self, other, alpha, out);
}
at::Tensor & _index_put_impl_(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values)) {
        return op_api::_index_put_impl_(self, indices, values, accumulate, unsafe);
    } else {
        return acl_op::_index_put_impl_(self, indices, values, accumulate, unsafe);
    }
}
at::Tensor & _log_softmax_backward_data_out(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::_log_softmax_backward_data_out(grad_output, output, dim, input_dtype, out);
    } else {
        return acl_op::_log_softmax_backward_data_out(grad_output, output, dim, input_dtype, out);
    }
}
at::Tensor & _log_softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out){
    return op_api::_log_softmax_out(self, dim, half_to_float, out);
}
at::Tensor & _slow_conv2d_forward_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output)) {
        return op_api::_slow_conv2d_forward_out(self, weight, kernel_size, bias, stride, padding, output);
    } else {
        return acl_op::_slow_conv2d_forward_out(self, weight, kernel_size, bias, stride, padding, output);
    }
}
at::Tensor & _softmax_backward_data_out(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::_softmax_backward_data_out(grad_output, output, dim, input_dtype, grad_input);
    } else {
        return acl_op::_softmax_backward_data_out(grad_output, output, dim, input_dtype, grad_input);
    }
}
at::Tensor & _softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::_softmax_out(self, dim, half_to_float, out);
    } else {
        return acl_op::_softmax_out(self, dim, half_to_float, out);
    }
}
at::Tensor & abs_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::abs_(self);
    } else {
        return acl_op::abs_(self);
    }
}
at::Tensor & abs_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::abs_out(self, out);
    } else {
        return acl_op::abs_out(self, out);
    }
}
at::Tensor & acos_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::acos_(self);
    } else {
        return acl_op::acos_(self);
    }
}
at::Tensor & acos_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::acos_out(self, out);
    } else {
        return acl_op::acos_out(self, out);
    }
}
at::Tensor & acosh_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::acosh_(self);
    } else {
        return acl_op::acosh_(self);
    }
}
at::Tensor & acosh_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::acosh_out(self, out);
    } else {
        return acl_op::acosh_out(self, out);
    }
}
at::Tensor & adaptive_avg_pool2d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::adaptive_avg_pool2d_out(self, output_size, out);
    } else {
        return acl_op::adaptive_avg_pool2d_out(self, output_size, out);
    }
}
at::Tensor & adaptive_avg_pool3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::adaptive_avg_pool3d_backward_out(grad_output, self, grad_input);
    } else {
        return acl_op::adaptive_avg_pool3d_backward_out(grad_output, self, grad_input);
    }
}
at::Tensor & adaptive_avg_pool3d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out){
    return acl_op::adaptive_avg_pool3d_out(self, output_size, out);
}
at::Tensor & adaptive_max_pool2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input){
    return acl_op::adaptive_max_pool2d_backward_out(grad_output, self, indices, grad_input);
}
at::Tensor & add_(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::add_(self, other, alpha);
    } else {
        return acl_op::add_(self, other, alpha);
    }
}
at::Tensor & add_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::add_(self, other, alpha);
    } else {
        return acl_op::add_(self, other, alpha);
    }
}
at::Tensor & add_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::add_out(self, other, alpha, out);
    } else {
        return acl_op::add_out(self, other, alpha, out);
    }
}
at::Tensor & add_out_sparse(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out){
    return sparse::add_out_sparse(self, other, alpha, out);
}
at::Tensor & addbmm_(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2)) {
        return op_api::addbmm_(self, batch1, batch2, beta, alpha);
    } else {
        return acl_op::addbmm_(self, batch1, batch2, beta, alpha);
    }
}
at::Tensor & addbmm_out(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::addbmm_out(self, batch1, batch2, beta, alpha, out);
    } else {
        return acl_op::addbmm_out(self, batch1, batch2, beta, alpha, out);
    }
}
at::Tensor & addcdiv_(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2)) {
        return op_api::addcdiv_(self, tensor1, tensor2, value);
    } else {
        return acl_op::addcdiv_(self, tensor1, tensor2, value);
    }
}
at::Tensor & addcdiv_out(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::addcdiv_out(self, tensor1, tensor2, value, out);
    } else {
        return acl_op::addcdiv_out(self, tensor1, tensor2, value, out);
    }
}
at::Tensor & addcmul_(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2)) {
        return op_api::addcmul_(self, tensor1, tensor2, value);
    } else {
        return acl_op::addcmul_(self, tensor1, tensor2, value);
    }
}
at::Tensor & addcmul_out(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::addcmul_out(self, tensor1, tensor2, value, out);
    } else {
        return acl_op::addcmul_out(self, tensor1, tensor2, value, out);
    }
}
at::Tensor & addmm_(at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2)) {
        return op_api::addmm_(self, mat1, mat2, beta, alpha);
    } else {
        return acl_op::addmm_(self, mat1, mat2, beta, alpha);
    }
}
at::Tensor & addmm_out(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::addmm_out(self, mat1, mat2, beta, alpha, out);
    } else {
        return acl_op::addmm_out(self, mat1, mat2, beta, alpha, out);
    }
}
at::Tensor & addmv_(at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec)) {
        return op_api::addmv_(self, mat, vec, beta, alpha);
    } else {
        return acl_op::addmv_(self, mat, vec, beta, alpha);
    }
}
at::Tensor & addmv_out(const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::addmv_out(self, mat, vec, beta, alpha, out);
    } else {
        return acl_op::addmv_out(self, mat, vec, beta, alpha, out);
    }
}
at::Tensor & addr_(at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec2)) {
        return op_api::addr_(self, vec1, vec2, beta, alpha);
    } else {
        return acl_op::addr_(self, vec1, vec2, beta, alpha);
    }
}
at::Tensor & addr_out(const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::addr_out(self, vec1, vec2, beta, alpha, out);
    } else {
        return acl_op::addr_out(self, vec1, vec2, beta, alpha, out);
    }
}
at::Tensor & all_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::all_out(self, out);
    } else {
        return acl_op::all_out(self, out);
    }
}
at::Tensor & all_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::all_out(self, dim, keepdim, out);
    } else {
        return acl_op::all_out(self, dim, keepdim, out);
    }
}
at::Tensor & amax_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::amax_out(self, dim, keepdim, out);
    } else {
        return acl_op::amax_out(self, dim, keepdim, out);
    }
}
at::Tensor & amin_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::amin_out(self, dim, keepdim, out);
    } else {
        return acl_op::amin_out(self, dim, keepdim, out);
    }
}
at::Tensor & angle_out(const at::Tensor & self, at::Tensor & out){
    return op_api::angle_out(self, out);
}
at::Tensor & any_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::any_out(self, out);
    } else {
        return acl_op::any_out(self, out);
    }
}
at::Tensor & any_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::any_out(self, dim, keepdim, out);
    } else {
        return acl_op::any_out(self, dim, keepdim, out);
    }
}
at::Tensor & arange_out(const at::Scalar & end, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::arange_out(end, out);
    } else {
        return acl_op::arange_out(end, out);
    }
}
at::Tensor & arange_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::arange_out(start, end, step, out);
    } else {
        return acl_op::arange_out(start, end, step, out);
    }
}
at::Tensor & argmax_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::argmax_out(self, dim, keepdim, out);
    } else {
        return acl_op::argmax_out(self, dim, keepdim, out);
    }
}
at::Tensor & argmin_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::argmin_out(self, dim, keepdim, out);
    } else {
        return acl_op::argmin_out(self, dim, keepdim, out);
    }
}
at::Tensor & asin_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::asin_(self);
    } else {
        return acl_op::asin_(self);
    }
}
at::Tensor & asin_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::asin_out(self, out);
    } else {
        return acl_op::asin_out(self, out);
    }
}
at::Tensor & asinh_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::asinh_(self);
    } else {
        return acl_op::asinh_(self);
    }
}
at::Tensor & asinh_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::asinh_out(self, out);
    } else {
        return acl_op::asinh_out(self, out);
    }
}
at::Tensor & atan2_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::atan2_(self, other);
    } else {
        return acl_op::atan2_(self, other);
    }
}
at::Tensor & atan2_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::atan2_out(self, other, out);
    } else {
        return acl_op::atan2_out(self, other, out);
    }
}
at::Tensor & atan_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::atan_(self);
    } else {
        return acl_op::atan_(self);
    }
}
at::Tensor & atan_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::atan_out(self, out);
    } else {
        return acl_op::atan_out(self, out);
    }
}
at::Tensor & atanh_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::atanh_(self);
    } else {
        return acl_op::atanh_(self);
    }
}
at::Tensor & atanh_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::atanh_out(self, out);
    } else {
        return acl_op::atanh_out(self, out);
    }
}
at::Tensor & avg_pool2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::avg_pool2d_backward_out(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
    } else {
        return acl_op::avg_pool2d_backward_out(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
    }
}
at::Tensor & avg_pool2d_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::avg_pool2d_out(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
    } else {
        return acl_op::avg_pool2d_out(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
    }
}
at::Tensor & avg_pool3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input){
    return acl_op::avg_pool3d_backward_out(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
}
at::Tensor & avg_pool3d_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out){
    return acl_op::avg_pool3d_out(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
}
at::Tensor & baddbmm_(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2)) {
        return op_api::baddbmm_(self, batch1, batch2, beta, alpha);
    } else {
        return acl_op::baddbmm_(self, batch1, batch2, beta, alpha);
    }
}
at::Tensor & baddbmm_out(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::baddbmm_out(self, batch1, batch2, beta, alpha, out);
    } else {
        return acl_op::baddbmm_out(self, batch1, batch2, beta, alpha, out);
    }
}
at::Tensor & batch_norm_elemt_out(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(invstd) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::batch_norm_elemt_out(input, weight, bias, mean, invstd, eps, out);
    } else {
        return acl_op::batch_norm_elemt_out(input, weight, bias, mean, invstd, eps, out);
    }
}
at::Tensor & bernoulli_(at::Tensor & self, const at::Tensor & p, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(p)) {
        return op_api::bernoulli_(self, p, generator);
    } else {
        return acl_op::bernoulli_(self, p, generator);
    }
}
at::Tensor & bernoulli_(at::Tensor & self, double p, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::bernoulli_(self, p, generator);
    } else {
        return acl_op::bernoulli_(self, p, generator);
    }
}
at::Tensor & bernoulli_out(const at::Tensor & self, c10::optional<at::Generator> generator, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::bernoulli_out(self, generator, out);
    } else {
        return acl_op::bernoulli_out(self, generator, out);
    }
}
at::Tensor & binary_cross_entropy_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::binary_cross_entropy_backward_out(grad_output, self, target, weight, reduction, grad_input);
    } else {
        return acl_op::binary_cross_entropy_backward_out(grad_output, self, target, weight, reduction, grad_input);
    }
}
at::Tensor & binary_cross_entropy_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::binary_cross_entropy_out(self, target, weight, reduction, out);
    } else {
        return acl_op::binary_cross_entropy_out(self, target, weight, reduction, out);
    }
}
at::Tensor & bitwise_and_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::bitwise_and_(self, other);
    } else {
        return acl_op::bitwise_and_(self, other);
    }
}
at::Tensor & bitwise_and_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::bitwise_and_(self, other);
    } else {
        return acl_op::bitwise_and_(self, other);
    }
}
at::Tensor & bitwise_and_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::bitwise_and_out(self, other, out);
    } else {
        return acl_op::bitwise_and_out(self, other, out);
    }
}
at::Tensor & bitwise_and_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::bitwise_and_out(self, other, out);
    } else {
        return acl_op::bitwise_and_out(self, other, out);
    }
}
at::Tensor & bitwise_not_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::bitwise_not_(self);
    } else {
        return acl_op::bitwise_not_(self);
    }
}
at::Tensor & bitwise_not_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::bitwise_not_out(self, out);
    } else {
        return acl_op::bitwise_not_out(self, out);
    }
}
at::Tensor & bitwise_or_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::bitwise_or_out(self, other, out);
    } else {
        return acl_op::bitwise_or_out(self, other, out);
    }
}
at::Tensor & bitwise_or_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::bitwise_or_out(self, other, out);
    } else {
        return acl_op::bitwise_or_out(self, other, out);
    }
}
at::Tensor & bitwise_xor_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::bitwise_xor_(self, other);
    } else {
        return acl_op::bitwise_xor_(self, other);
    }
}
at::Tensor & bitwise_xor_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::bitwise_xor_(self, other);
    } else {
        return acl_op::bitwise_xor_(self, other);
    }
}
at::Tensor & bitwise_xor_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::bitwise_xor_out(self, other, out);
    } else {
        return acl_op::bitwise_xor_out(self, other, out);
    }
}
at::Tensor & bitwise_xor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::bitwise_xor_out(self, other, out);
    } else {
        return acl_op::bitwise_xor_out(self, other, out);
    }
}
at::Tensor & bmm_out(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::bmm_out(self, mat2, out);
    } else {
        return acl_op::bmm_out(self, mat2, out);
    }
}
at::Tensor & bucketize_out(const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out){
    return op_api::bucketize_out(self, boundaries, out_int32, right, out);
}
at::Tensor & cat_out(at::TensorList tensors, at::Dimname dim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::cat_out(tensors, dim, out);
    } else {
        return acl_op::cat_out(tensors, dim, out);
    }
}
at::Tensor & cat_out(const at::ITensorListRef & tensors, int64_t dim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::cat_out(tensors, dim, out);
    } else {
        return acl_op::cat_out(tensors, dim, out);
    }
}
at::Tensor & ceil_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::ceil_(self);
    } else {
        return acl_op::ceil_(self);
    }
}
at::Tensor & ceil_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::ceil_out(self, out);
    } else {
        return acl_op::ceil_out(self, out);
    }
}
at::Tensor & celu_(at::Tensor & self, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::celu_(self, alpha);
    } else {
        return acl_op::celu_(self, alpha);
    }
}
at::Tensor & clamp_(at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::clamp_(self, min, max);
    } else {
        return acl_op::clamp_(self, min, max);
    }
}
at::Tensor & clamp_(at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max)) {
        return op_api::clamp_(self, min, max);
    } else {
        return acl_op::clamp_(self, min, max);
    }
}
at::Tensor & clamp_max_(at::Tensor & self, const at::Scalar & max){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::clamp_max_(self, max);
    } else {
        return acl_op::clamp_max_(self, max);
    }
}
at::Tensor & clamp_max_(at::Tensor & self, const at::Tensor & max){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max)) {
        return op_api::clamp_max_(self, max);
    } else {
        return acl_op::clamp_max_(self, max);
    }
}
at::Tensor & clamp_max_out(const at::Tensor & self, const at::Scalar & max, at::Tensor & out){
    return acl_op::clamp_max_out(self, max, out);
}
at::Tensor & clamp_max_out(const at::Tensor & self, const at::Tensor & max, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::clamp_max_out(self, max, out);
    } else {
        return acl_op::clamp_max_out(self, max, out);
    }
}
at::Tensor & clamp_min_(at::Tensor & self, const at::Scalar & min){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::clamp_min_(self, min);
    } else {
        return acl_op::clamp_min_(self, min);
    }
}
at::Tensor & clamp_min_(at::Tensor & self, const at::Tensor & min){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min)) {
        return op_api::clamp_min_(self, min);
    } else {
        return acl_op::clamp_min_(self, min);
    }
}
at::Tensor & clamp_min_out(const at::Tensor & self, const at::Scalar & min, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::clamp_min_out(self, min, out);
    } else {
        return acl_op::clamp_min_out(self, min, out);
    }
}
at::Tensor & clamp_min_out(const at::Tensor & self, const at::Tensor & min, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::clamp_min_out(self, min, out);
    } else {
        return acl_op::clamp_min_out(self, min, out);
    }
}
at::Tensor & clamp_out(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::clamp_out(self, min, max, out);
    } else {
        return acl_op::clamp_out(self, min, max, out);
    }
}
at::Tensor & clamp_out(const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::clamp_out(self, min, max, out);
    } else {
        return acl_op::clamp_out(self, min, max, out);
    }
}
at::Tensor & col2im_out(const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::col2im_out(self, output_size, kernel_size, dilation, padding, stride, out);
    } else {
        return acl_op::col2im_out(self, output_size, kernel_size, dilation, padding, stride, out);
    }
}
at::Tensor & complex_out(const at::Tensor & real, const at::Tensor & imag, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(real) && at_npu::native::FormatHelper::IsOpInputBaseFormat(imag) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::complex_out(real, imag, out);
    } else {
        return acl_op::complex_out(real, imag, out);
    }
}
at::Tensor & cos_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::cos_(self);
    } else {
        return acl_op::cos_(self);
    }
}
at::Tensor & cos_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::cos_out(self, out);
    } else {
        return acl_op::cos_out(self, out);
    }
}
at::Tensor & cosh_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::cosh_(self);
    } else {
        return acl_op::cosh_(self);
    }
}
at::Tensor & cosh_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::cosh_out(self, out);
    } else {
        return acl_op::cosh_out(self, out);
    }
}
at::Tensor & cumprod_(at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype){
    return acl_op::cumprod_(self, dim, dtype);
}
at::Tensor & cumprod_(at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype){
    return acl_op::cumprod_(self, dim, dtype);
}
at::Tensor & cumprod_out(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    return acl_op::cumprod_out(self, dim, dtype, out);
}
at::Tensor & cumprod_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    return acl_op::cumprod_out(self, dim, dtype, out);
}
at::Tensor & cumsum_out(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::cumsum_out(self, dim, dtype, out);
    } else {
        return acl_op::cumsum_out(self, dim, dtype, out);
    }
}
at::Tensor & cumsum_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::cumsum_out(self, dim, dtype, out);
    } else {
        return acl_op::cumsum_out(self, dim, dtype, out);
    }
}
at::Tensor & diag_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::diag_out(self, diagonal, out);
    } else {
        return acl_op::diag_out(self, diagonal, out);
    }
}
at::Tensor & div_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::div_(self, other);
    } else {
        return acl_op::div_(self, other);
    }
}
at::Tensor & div_(at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::div_(self, other, rounding_mode);
    } else {
        return acl_op::div_(self, other, rounding_mode);
    }
}
at::Tensor & div_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::div_(self, other);
    } else {
        return acl_op::div_(self, other);
    }
}
at::Tensor & div_(at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::div_(self, other, rounding_mode);
    } else {
        return acl_op::div_(self, other, rounding_mode);
    }
}
at::Tensor & div_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::div_out(self, other, out);
    } else {
        return acl_op::div_out(self, other, out);
    }
}
at::Tensor & div_out(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::div_out(self, other, rounding_mode, out);
    } else {
        return acl_op::div_out(self, other, rounding_mode, out);
    }
}
at::Tensor & dot_out(const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::dot_out(self, tensor, out);
    } else {
        return acl_op::dot_out(self, tensor, out);
    }
}
at::Tensor & elu_(at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::elu_(self, alpha, scale, input_scale);
    } else {
        return acl_op::elu_(self, alpha, scale, input_scale);
    }
}
at::Tensor & elu_backward_out(const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self_or_result) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::elu_backward_out(grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
    } else {
        return acl_op::elu_backward_out(grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
    }
}
at::Tensor & elu_out(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::elu_out(self, alpha, scale, input_scale, out);
    } else {
        return acl_op::elu_out(self, alpha, scale, input_scale, out);
    }
}
at::Tensor & embedding_renorm_(at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::embedding_renorm_(self, indices, max_norm, norm_type);
    } else {
        return acl_op::embedding_renorm_(self, indices, max_norm, norm_type);
    }
}
at::Tensor & eq_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::eq_(self, other);
    } else {
        return acl_op::eq_(self, other);
    }
}
at::Tensor & eq_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::eq_(self, other);
    } else {
        return acl_op::eq_(self, other);
    }
}
at::Tensor & eq_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::eq_out(self, other, out);
    } else {
        return acl_op::eq_out(self, other, out);
    }
}
at::Tensor & eq_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::eq_out(self, other, out);
    } else {
        return acl_op::eq_out(self, other, out);
    }
}
at::Tensor & erf_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::erf_(self);
    } else {
        return acl_op::erf_(self);
    }
}
at::Tensor & erf_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::erf_out(self, out);
    } else {
        return acl_op::erf_out(self, out);
    }
}
at::Tensor & erfc_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::erfc_(self);
    } else {
        return acl_op::erfc_(self);
    }
}
at::Tensor & erfc_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::erfc_out(self, out);
    } else {
        return acl_op::erfc_out(self, out);
    }
}
at::Tensor & erfinv_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::erfinv_(self);
    } else {
        return acl_op::erfinv_(self);
    }
}
at::Tensor & erfinv_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::erfinv_out(self, out);
    } else {
        return acl_op::erfinv_out(self, out);
    }
}
at::Tensor & exp2_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::exp2_(self);
    } else {
        return acl_op::exp2_(self);
    }
}
at::Tensor & exp2_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::exp2_out(self, out);
    } else {
        return acl_op::exp2_out(self, out);
    }
}
at::Tensor & exp_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::exp_(self);
    } else {
        return acl_op::exp_(self);
    }
}
at::Tensor & exp_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::exp_out(self, out);
    } else {
        return acl_op::exp_out(self, out);
    }
}
at::Tensor & expm1_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::expm1_(self);
    } else {
        return acl_op::expm1_(self);
    }
}
at::Tensor & expm1_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::expm1_out(self, out);
    } else {
        return acl_op::expm1_out(self, out);
    }
}
at::Tensor & exponential_(at::Tensor & self, double lambd, c10::optional<at::Generator> generator){
    return op_api::exponential_(self, lambd, generator);
}
at::Tensor & eye_out(int64_t n, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::eye_out(n, out);
    } else {
        return acl_op::eye_out(n, out);
    }
}
at::Tensor & eye_out(int64_t n, int64_t m, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::eye_out(n, m, out);
    } else {
        return acl_op::eye_out(n, m, out);
    }
}
at::Tensor & fill_(at::Tensor & self, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::fill_(self, value);
    } else {
        return acl_op::fill_(self, value);
    }
}
at::Tensor & fill_(at::Tensor & self, const at::Tensor & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(value)) {
        return op_api::fill_(self, value);
    } else {
        return acl_op::fill_(self, value);
    }
}
at::Tensor & fill_diagonal_(at::Tensor & self, const at::Scalar & fill_value, bool wrap){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::fill_diagonal_(self, fill_value, wrap);
    } else {
        return acl_op::fill_diagonal_(self, fill_value, wrap);
    }
}
at::Tensor & floor_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::floor_(self);
    } else {
        return acl_op::floor_(self);
    }
}
at::Tensor & floor_divide_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::floor_divide_(self, other);
    } else {
        return acl_op::floor_divide_(self, other);
    }
}
at::Tensor & floor_divide_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::floor_divide_(self, other);
    } else {
        return acl_op::floor_divide_(self, other);
    }
}
at::Tensor & floor_divide_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::floor_divide_out(self, other, out);
    } else {
        return acl_op::floor_divide_out(self, other, out);
    }
}
at::Tensor & floor_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::floor_out(self, out);
    } else {
        return acl_op::floor_out(self, out);
    }
}
at::Tensor & fmod_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::fmod_(self, other);
    } else {
        return acl_op::fmod_(self, other);
    }
}
at::Tensor & fmod_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::fmod_(self, other);
    } else {
        return acl_op::fmod_(self, other);
    }
}
at::Tensor & fmod_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::fmod_out(self, other, out);
    } else {
        return acl_op::fmod_out(self, other, out);
    }
}
at::Tensor & fmod_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::fmod_out(self, other, out);
    } else {
        return acl_op::fmod_out(self, other, out);
    }
}
at::Tensor & frac_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::frac_(self);
    } else {
        return acl_op::frac_(self);
    }
}
at::Tensor & frac_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::frac_out(self, out);
    } else {
        return acl_op::frac_out(self, out);
    }
}
at::Tensor & gather_out(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::gather_out(self, dim, index, sparse_grad, out);
    } else {
        return acl_op::gather_out(self, dim, index, sparse_grad, out);
    }
}
at::Tensor & gather_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::gather_out(self, dim, index, sparse_grad, out);
    } else {
        return acl_op::gather_out(self, dim, index, sparse_grad, out);
    }
}
at::Tensor & gcd_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::gcd_out(self, other, out);
    } else {
        return acl_op::gcd_out(self, other, out);
    }
}
at::Tensor & ge_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::ge_(self, other);
    } else {
        return acl_op::ge_(self, other);
    }
}
at::Tensor & ge_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::ge_(self, other);
    } else {
        return acl_op::ge_(self, other);
    }
}
at::Tensor & ge_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::ge_out(self, other, out);
    } else {
        return acl_op::ge_out(self, other, out);
    }
}
at::Tensor & ge_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::ge_out(self, other, out);
    } else {
        return acl_op::ge_out(self, other, out);
    }
}
at::Tensor & glu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::glu_backward_out(grad_output, self, dim, grad_input);
    } else {
        return acl_op::glu_backward_out(grad_output, self, dim, grad_input);
    }
}
at::Tensor & glu_out(const at::Tensor & self, int64_t dim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::glu_out(self, dim, out);
    } else {
        return acl_op::glu_out(self, dim, out);
    }
}
at::Tensor & gt_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::gt_(self, other);
    } else {
        return acl_op::gt_(self, other);
    }
}
at::Tensor & gt_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::gt_(self, other);
    } else {
        return acl_op::gt_(self, other);
    }
}
at::Tensor & gt_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::gt_out(self, other, out);
    } else {
        return acl_op::gt_out(self, other, out);
    }
}
at::Tensor & gt_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::gt_out(self, other, out);
    } else {
        return acl_op::gt_out(self, other, out);
    }
}
at::Tensor & hardshrink_backward_out(const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::hardshrink_backward_out(grad_out, self, lambd, grad_input);
    } else {
        return acl_op::hardshrink_backward_out(grad_out, self, lambd, grad_input);
    }
}
at::Tensor & hardshrink_out(const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::hardshrink_out(self, lambd, out);
    } else {
        return acl_op::hardshrink_out(self, lambd, out);
    }
}
at::Tensor & hardsigmoid_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardsigmoid_(self);
    } else {
        return acl_op::hardsigmoid_(self);
    }
}
at::Tensor & hardsigmoid_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::hardsigmoid_out(self, out);
    } else {
        return acl_op::hardsigmoid_out(self, out);
    }
}
at::Tensor & hardswish_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardswish_(self);
    } else {
        return acl_op::hardswish_(self);
    }
}
at::Tensor & hardswish_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::hardswish_out(self, out);
    } else {
        return acl_op::hardswish_out(self, out);
    }
}
at::Tensor & hardtanh_(at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardtanh_(self, min_val, max_val);
    } else {
        return acl_op::hardtanh_(self, min_val, max_val);
    }
}
at::Tensor & hardtanh_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input){
    return acl_op::hardtanh_backward_out(grad_output, self, min_val, max_val, grad_input);
}
at::Tensor & hardtanh_out(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out){
    return acl_op::hardtanh_out(self, min_val, max_val, out);
}
at::Tensor & histc_out(const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::histc_out(self, bins, min, max, out);
    } else {
        return acl_op::histc_out(self, bins, min, max, out);
    }
}
at::Tensor & im2col_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::im2col_out(self, kernel_size, dilation, padding, stride, out);
    } else {
        return acl_op::im2col_out(self, kernel_size, dilation, padding, stride, out);
    }
}
at::Tensor & index_add_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(source) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::index_add_out(self, dim, index, source, alpha, out);
    } else {
        return acl_op::index_add_out(self, dim, index, source, alpha, out);
    }
}
at::Tensor & index_copy_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(source)) {
        return op_api::index_copy_(self, dim, index, source);
    } else {
        return acl_op::index_copy_(self, dim, index, source);
    }
}
at::Tensor & index_fill_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index)) {
        return op_api::index_fill_(self, dim, index, value);
    } else {
        return acl_op::index_fill_(self, dim, index, value);
    }
}
at::Tensor & index_fill_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(value)) {
        return op_api::index_fill_(self, dim, index, value);
    } else {
        return acl_op::index_fill_(self, dim, index, value);
    }
}
at::Tensor & index_put_(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values)) {
        return op_api::index_put_(self, indices, values, accumulate);
    } else {
        return acl_op::index_put_(self, indices, values, accumulate);
    }
}
at::Tensor & index_select_out(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::index_select_out(self, dim, index, out);
    } else {
        return acl_op::index_select_out(self, dim, index, out);
    }
}
at::Tensor & index_select_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::index_select_out(self, dim, index, out);
    } else {
        return acl_op::index_select_out(self, dim, index, out);
    }
}
at::Tensor & inverse_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::inverse_out(self, out);
    } else {
        return acl_op::inverse_out(self, out);
    }
}
at::Tensor & isin_out(const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(test_elements) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::isin_out(element, test_elements, assume_unique, invert, out);
    } else {
        return acl_op::isin_out(element, test_elements, assume_unique, invert, out);
    }
}
at::Tensor & isin_out(const at::Tensor & element, const at::Scalar & test_elements, bool assume_unique, bool invert, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(element) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::isin_out(element, test_elements, assume_unique, invert, out);
    } else {
        return acl_op::isin_out(element, test_elements, assume_unique, invert, out);
    }
}
at::Tensor & isneginf_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::isneginf_out(self, out);
    } else {
        return acl_op::isneginf_out(self, out);
    }
}
at::Tensor & isposinf_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::isposinf_out(self, out);
    } else {
        return acl_op::isposinf_out(self, out);
    }
}
at::Tensor & le_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::le_(self, other);
    } else {
        return acl_op::le_(self, other);
    }
}
at::Tensor & le_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::le_(self, other);
    } else {
        return acl_op::le_(self, other);
    }
}
at::Tensor & le_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::le_out(self, other, out);
    } else {
        return acl_op::le_out(self, other, out);
    }
}
at::Tensor & le_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::le_out(self, other, out);
    } else {
        return acl_op::le_out(self, other, out);
    }
}
at::Tensor & leaky_relu_(at::Tensor & self, const at::Scalar & negative_slope){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::leaky_relu_(self, negative_slope);
    } else {
        return acl_op::leaky_relu_(self, negative_slope);
    }
}
at::Tensor & leaky_relu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result, at::Tensor & grad_input){
    return op_api::leaky_relu_backward_out(grad_output, self, negative_slope, self_is_result, grad_input);
}
at::Tensor & leaky_relu_out(const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::leaky_relu_out(self, negative_slope, out);
    } else {
        return acl_op::leaky_relu_out(self, negative_slope, out);
    }
}
at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Scalar & weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(end)) {
        return op_api::lerp_(self, end, weight);
    } else {
        return acl_op::lerp_(self, end, weight);
    }
}
at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Tensor & weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(end) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::lerp_(self, end, weight);
    } else {
        return acl_op::lerp_(self, end, weight);
    }
}
at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(end) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::lerp_out(self, end, weight, out);
    } else {
        return acl_op::lerp_out(self, end, weight, out);
    }
}
at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(end) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::lerp_out(self, end, weight, out);
    } else {
        return acl_op::lerp_out(self, end, weight, out);
    }
}
at::Tensor & linalg_cross_out(const at::Tensor & self, const at::Tensor & other, int64_t dim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::linalg_cross_out(self, other, dim, out);
    } else {
        return acl_op::linalg_cross_out(self, other, dim, out);
    }
}
at::Tensor & linalg_matrix_norm_out(const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    return acl_op::linalg_matrix_norm_out(self, ord, dim, keepdim, dtype, out);
}
at::Tensor & linalg_matrix_norm_out(const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    return acl_op::linalg_matrix_norm_out(self, ord, dim, keepdim, dtype, out);
}
at::Tensor & linalg_norm_out(const at::Tensor & self, c10::string_view ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    return acl_op::linalg_norm_out(self, ord, dim, keepdim, dtype, out);
}
at::Tensor & linalg_norm_out(const at::Tensor & self, const c10::optional<at::Scalar> & ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    return acl_op::linalg_norm_out(self, ord, dim, keepdim, dtype, out);
}
at::Tensor & linalg_solve_triangular_out(const at::Tensor & self, const at::Tensor & B, bool upper, bool left, bool unitriangular, at::Tensor & out){
    return op_api::linalg_solve_triangular_out(self, B, upper, left, unitriangular, out);
}
at::Tensor & linalg_svdvals_out(const at::Tensor & A, c10::optional<c10::string_view> driver, at::Tensor & out){
    return acl_op::linalg_svdvals_out(A, driver, out);
}
at::Tensor & linalg_vector_norm_out(const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::linalg_vector_norm_out(self, ord, dim, keepdim, dtype, out);
    } else {
        return acl_op::linalg_vector_norm_out(self, ord, dim, keepdim, dtype, out);
    }
}
at::Tensor & linspace_out(const at::Scalar & start, const at::Scalar & end, int64_t steps, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::linspace_out(start, end, steps, out);
    } else {
        return acl_op::linspace_out(start, end, steps, out);
    }
}
at::Tensor & log10_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::log10_(self);
    } else {
        return acl_op::log10_(self);
    }
}
at::Tensor & log10_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::log10_out(self, out);
    } else {
        return acl_op::log10_out(self, out);
    }
}
at::Tensor & log1p_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::log1p_(self);
    } else {
        return acl_op::log1p_(self);
    }
}
at::Tensor & log1p_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::log1p_out(self, out);
    } else {
        return acl_op::log1p_out(self, out);
    }
}
at::Tensor & log2_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::log2_(self);
    } else {
        return acl_op::log2_(self);
    }
}
at::Tensor & log2_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::log2_out(self, out);
    } else {
        return acl_op::log2_out(self, out);
    }
}
at::Tensor & log_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::log_(self);
    } else {
        return acl_op::log_(self);
    }
}
at::Tensor & log_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::log_out(self, out);
    } else {
        return acl_op::log_out(self, out);
    }
}
at::Tensor & log_sigmoid_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(buffer) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::log_sigmoid_backward_out(grad_output, self, buffer, grad_input);
    } else {
        return acl_op::log_sigmoid_backward_out(grad_output, self, buffer, grad_input);
    }
}
at::Tensor & log_sigmoid_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::log_sigmoid_out(self, out);
    } else {
        return acl_op::log_sigmoid_out(self, out);
    }
}
at::Tensor & logaddexp2_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::logaddexp2_out(self, other, out);
    } else {
        return acl_op::logaddexp2_out(self, other, out);
    }
}
at::Tensor & logaddexp_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::logaddexp_out(self, other, out);
    } else {
        return acl_op::logaddexp_out(self, other, out);
    }
}
at::Tensor & logical_and_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::logical_and_(self, other);
    } else {
        return acl_op::logical_and_(self, other);
    }
}
at::Tensor & logical_and_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::logical_and_out(self, other, out);
    } else {
        return acl_op::logical_and_out(self, other, out);
    }
}
at::Tensor & logical_not_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::logical_not_(self);
    } else {
        return acl_op::logical_not_(self);
    }
}
at::Tensor & logical_not_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::logical_not_out(self, out);
    } else {
        return acl_op::logical_not_out(self, out);
    }
}
at::Tensor & logical_or_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::logical_or_(self, other);
    } else {
        return acl_op::logical_or_(self, other);
    }
}
at::Tensor & logical_or_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::logical_or_out(self, other, out);
    } else {
        return acl_op::logical_or_out(self, other, out);
    }
}
at::Tensor & logical_xor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::logical_xor_out(self, other, out);
    } else {
        return acl_op::logical_xor_out(self, other, out);
    }
}
at::Tensor & logspace_out(const at::Scalar & start, const at::Scalar & end, int64_t steps, double base, at::Tensor & out){
    return acl_op::logspace_out(start, end, steps, base, out);
}
at::Tensor & logsumexp_out(const at::Tensor & self, at::DimnameList dim, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::logsumexp_out(self, dim, keepdim, out);
    } else {
        return acl_op::logsumexp_out(self, dim, keepdim, out);
    }
}
at::Tensor & logsumexp_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::logsumexp_out(self, dim, keepdim, out);
    } else {
        return acl_op::logsumexp_out(self, dim, keepdim, out);
    }
}
at::Tensor & lt_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::lt_(self, other);
    } else {
        return acl_op::lt_(self, other);
    }
}
at::Tensor & lt_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::lt_(self, other);
    } else {
        return acl_op::lt_(self, other);
    }
}
at::Tensor & lt_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::lt_out(self, other, out);
    } else {
        return acl_op::lt_out(self, other, out);
    }
}
at::Tensor & lt_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::lt_out(self, other, out);
    } else {
        return acl_op::lt_out(self, other, out);
    }
}
at::Tensor & masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mask)) {
        return op_api::masked_fill_(self, mask, value);
    } else {
        return acl_op::masked_fill_(self, mask, value);
    }
}
at::Tensor & masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Tensor & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mask) && at_npu::native::FormatHelper::IsOpInputBaseFormat(value)) {
        return op_api::masked_fill_(self, mask, value);
    } else {
        return acl_op::masked_fill_(self, mask, value);
    }
}
at::Tensor & masked_scatter_(at::Tensor & self, const at::Tensor & mask, const at::Tensor & source){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mask) && at_npu::native::FormatHelper::IsOpInputBaseFormat(source)) {
        return op_api::masked_scatter_(self, mask, source);
    } else {
        return acl_op::masked_scatter_(self, mask, source);
    }
}
at::Tensor & masked_select_out(const at::Tensor & self, const at::Tensor & mask, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mask) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::masked_select_out(self, mask, out);
    } else {
        return acl_op::masked_select_out(self, mask, out);
    }
}
at::Tensor & matmul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::matmul_out(self, other, out);
    } else {
        return acl_op::matmul_out(self, other, out);
    }
}
at::Tensor & max_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::max_out(self, other, out);
    } else {
        return acl_op::max_out(self, other, out);
    }
}
at::Tensor & max_out_sparse(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    return sparse::max_out_sparse(self, other, out);
}
at::Tensor & max_pool2d_with_indices_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::max_pool2d_with_indices_backward_out(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
    } else {
        return acl_op::max_pool2d_with_indices_backward_out(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
    }
}
at::Tensor & max_pool3d_with_indices_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input){
    return acl_op::max_pool3d_with_indices_backward_out(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
}
at::Tensor & max_unpool2d_out(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::max_unpool2d_out(self, indices, output_size, out);
    } else {
        return acl_op::max_unpool2d_out(self, indices, output_size, out);
    }
}
at::Tensor & max_unpool3d_out(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::max_unpool3d_out(self, indices, output_size, stride, padding, out);
    } else {
        return acl_op::max_unpool3d_out(self, indices, output_size, stride, padding, out);
    }
}
at::Tensor & maximum_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::maximum_out(self, other, out);
    } else {
        return acl_op::maximum_out(self, other, out);
    }
}
at::Tensor & mean_out(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::mean_out(self, dim, keepdim, dtype, out);
    } else {
        return acl_op::mean_out(self, dim, keepdim, dtype, out);
    }
}
at::Tensor & mean_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::mean_out(self, dim, keepdim, dtype, out);
    } else {
        return acl_op::mean_out(self, dim, keepdim, dtype, out);
    }
}
at::Tensor & min_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::min_out(self, other, out);
    } else {
        return acl_op::min_out(self, other, out);
    }
}
at::Tensor & minimum_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::minimum_out(self, other, out);
    } else {
        return acl_op::minimum_out(self, other, out);
    }
}
at::Tensor & mish_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::mish_(self);
    } else {
        return acl_op::mish_(self);
    }
}
at::Tensor & mish_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::mish_out(self, out);
    } else {
        return acl_op::mish_out(self, out);
    }
}
at::Tensor & mm_out(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::mm_out(self, mat2, out);
    } else {
        return acl_op::mm_out(self, mat2, out);
    }
}
at::Tensor & mse_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::mse_loss_backward_out(grad_output, self, target, reduction, grad_input);
    } else {
        return acl_op::mse_loss_backward_out(grad_output, self, target, reduction, grad_input);
    }
}
at::Tensor & mse_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::mse_loss_out(self, target, reduction, out);
    } else {
        return acl_op::mse_loss_out(self, target, reduction, out);
    }
}
at::Tensor & mul_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::mul_(self, other);
    } else {
        return acl_op::mul_(self, other);
    }
}
at::Tensor & mul_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::mul_(self, other);
    } else {
        return acl_op::mul_(self, other);
    }
}
at::Tensor & mul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::mul_out(self, other, out);
    } else {
        return acl_op::mul_out(self, other, out);
    }
}
at::Tensor & multilabel_margin_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::multilabel_margin_loss_out(self, target, reduction, out);
    } else {
        return acl_op::multilabel_margin_loss_out(self, target, reduction, out);
    }
}
at::Tensor & multinomial_out(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::multinomial_out(self, num_samples, replacement, generator, out);
    } else {
        return acl_op::multinomial_out(self, num_samples, replacement, generator, out);
    }
}
at::Tensor & mv_out(const at::Tensor & self, const at::Tensor & vec, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::mv_out(self, vec, out);
    } else {
        return acl_op::mv_out(self, vec, out);
    }
}
at::Tensor & nan_to_num_(at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::nan_to_num_(self, nan, posinf, neginf);
    } else {
        return acl_op::nan_to_num_(self, nan, posinf, neginf);
    }
}
at::Tensor & nan_to_num_out(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::nan_to_num_out(self, nan, posinf, neginf, out);
    } else {
        return acl_op::nan_to_num_out(self, nan, posinf, neginf, out);
    }
}
at::Tensor & nansum_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    return op_api::nansum_out(self, dim, keepdim, dtype, out);
}
at::Tensor & ne_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::ne_(self, other);
    } else {
        return acl_op::ne_(self, other);
    }
}
at::Tensor & ne_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::ne_(self, other);
    } else {
        return acl_op::ne_(self, other);
    }
}
at::Tensor & ne_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::ne_out(self, other, out);
    } else {
        return acl_op::ne_out(self, other, out);
    }
}
at::Tensor & ne_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::ne_out(self, other, out);
    } else {
        return acl_op::ne_out(self, other, out);
    }
}
at::Tensor & neg_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::neg_(self);
    } else {
        return acl_op::neg_(self);
    }
}
at::Tensor & neg_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::neg_out(self, out);
    } else {
        return acl_op::neg_out(self, out);
    }
}
at::Tensor & nll_loss2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::nll_loss2d_backward_out(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
    } else {
        return acl_op::nll_loss2d_backward_out(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
    }
}
at::Tensor & nll_loss2d_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out){
    return acl_op::nll_loss2d_out(self, target, weight, reduction, ignore_index, out);
}
at::Tensor & nll_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::nll_loss_backward_out(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
    } else {
        return acl_op::nll_loss_backward_out(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
    }
}
at::Tensor & nll_loss_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out){
    return acl_op::nll_loss_out(self, target, weight, reduction, ignore_index, out);
}
at::Tensor & nonzero_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::nonzero_out(self, out);
    } else {
        return acl_op::nonzero_out(self, out);
    }
}
at::Tensor & norm_out(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::norm_out(self, p, dim, keepdim, dtype, out);
    } else {
        return acl_op::norm_out(self, p, dim, keepdim, dtype, out);
    }
}
at::Tensor & norm_out(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::norm_out(self, p, dim, keepdim, out);
    } else {
        return acl_op::norm_out(self, p, dim, keepdim, out);
    }
}
at::Tensor & normal_(at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::normal_(self, mean, std, generator);
    } else {
        return acl_op::normal_(self, mean, std, generator);
    }
}
at::Tensor & normal_out(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(std) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::normal_out(mean, std, generator, out);
    } else {
        return acl_op::normal_out(mean, std, generator, out);
    }
}
at::Tensor & normal_out(const at::Tensor & mean, double std, c10::optional<at::Generator> generator, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::normal_out(mean, std, generator, out);
    } else {
        return acl_op::normal_out(mean, std, generator, out);
    }
}
at::Tensor & normal_out(double mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(std) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::normal_out(mean, std, generator, out);
    } else {
        return acl_op::normal_out(mean, std, generator, out);
    }
}
at::Tensor & normal_out(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::normal_out(mean, std, size, generator, out);
    } else {
        return acl_op::normal_out(mean, std, size, generator, out);
    }
}
at::Tensor & npu_broadcast_out(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out){
    return acl_op::npu_broadcast_out(self, size, out);
}
at::Tensor & npu_conv2d_out(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor & out){
    return acl_op::npu_conv2d_out(input, weight, bias, stride, padding, dilation, groups, out);
}
at::Tensor & npu_conv3d_out(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor & out){
    return acl_op::npu_conv3d_out(input, weight, bias, stride, padding, dilation, groups, out);
}
at::Tensor & npu_dtype_cast_(at::Tensor & self, const at::Tensor & src){
    return acl_op::npu_dtype_cast_(self, src);
}
at::Tensor & npu_indexing_out(const at::Tensor & self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor & out){
    return acl_op::npu_indexing_out(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, out);
}
at::Tensor & npu_quant_scatter_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, const at::Tensor & quant_scales, const c10::optional<at::Tensor> & quant_zero_points, int64_t axis, int64_t quant_axis, c10::string_view reduce){
    return op_api::npu_quant_scatter_(self, indices, updates, quant_scales, quant_zero_points, axis, quant_axis, reduce);
}
at::Tensor & npu_reshape_out(const at::Tensor & self, at::IntArrayRef shape, bool can_refresh, at::Tensor & out){
    return acl_op::npu_reshape_out(self, shape, can_refresh, out);
}
at::Tensor & npu_scatter_nd_update_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates){
    return op_api::npu_scatter_nd_update_(self, indices, updates);
}
at::Tensor & npu_silu_(at::Tensor & self){
    return acl_op::npu_silu_(self);
}
at::Tensor & npu_slice_out(const at::Tensor & self, at::IntArrayRef offsets, at::IntArrayRef size, at::Tensor & out){
    return acl_op::npu_slice_out(self, offsets, size, out);
}
at::Tensor & npu_sort_v2_out(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & out){
    return acl_op::npu_sort_v2_out(self, dim, descending, out);
}
at::Tensor & npu_stride_copy_out(const at::Tensor & self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar & storage_offset, at::Tensor & out){
    return acl_op::npu_stride_copy_out(self, shape, stride, storage_offset, out);
}
at::Tensor & npu_transpose_out(const at::Tensor & self, at::IntArrayRef perm, bool require_contiguous, at::Tensor & out){
    return acl_op::npu_transpose_out(self, perm, require_contiguous, out);
}
at::Tensor & npu_view_copy(at::Tensor & self, const at::Tensor & other, bool non_blocking){
    return acl_op::npu_view_copy(self, other, non_blocking);
}
at::Tensor & one_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::one_(self);
    } else {
        return acl_op::one_(self);
    }
}
at::Tensor & ones_out(at::IntArrayRef size, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::ones_out(size, out);
    } else {
        return acl_op::ones_out(size, out);
    }
}
at::Tensor & pow_(at::Tensor & self, const at::Scalar & exponent){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::pow_(self, exponent);
    } else {
        return acl_op::pow_(self, exponent);
    }
}
at::Tensor & pow_(at::Tensor & self, const at::Tensor & exponent){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent)) {
        return op_api::pow_(self, exponent);
    } else {
        return acl_op::pow_(self, exponent);
    }
}
at::Tensor & pow_out(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::pow_out(self, exponent, out);
    } else {
        return acl_op::pow_out(self, exponent, out);
    }
}
at::Tensor & pow_out(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::pow_out(self, exponent, out);
    } else {
        return acl_op::pow_out(self, exponent, out);
    }
}
at::Tensor & pow_out(const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::pow_out(self, exponent, out);
    } else {
        return acl_op::pow_out(self, exponent, out);
    }
}
at::Tensor & prod_out(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::prod_out(self, dim, keepdim, dtype, out);
    } else {
        return acl_op::prod_out(self, dim, keepdim, dtype, out);
    }
}
at::Tensor & put_(at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(source)) {
        return op_api::put_(self, index, source, accumulate);
    } else {
        return acl_op::put_(self, index, source, accumulate);
    }
}
at::Tensor & random_(at::Tensor & self, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::random_(self, generator);
    } else {
        return acl_op::random_(self, generator);
    }
}
at::Tensor & random_(at::Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::random_(self, from, to, generator);
    } else {
        return acl_op::random_(self, from, to, generator);
    }
}
at::Tensor & random_(at::Tensor & self, int64_t to, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::random_(self, to, generator);
    } else {
        return acl_op::random_(self, to, generator);
    }
}
at::Tensor & randperm_out(int64_t n, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::randperm_out(n, out);
    } else {
        return acl_op::randperm_out(n, out);
    }
}
at::Tensor & randperm_out(int64_t n, c10::optional<at::Generator> generator, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::randperm_out(n, generator, out);
    } else {
        return acl_op::randperm_out(n, generator, out);
    }
}
at::Tensor & range_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::range_out(start, end, step, out);
    } else {
        return acl_op::range_out(start, end, step, out);
    }
}
at::Tensor & reciprocal_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::reciprocal_(self);
    } else {
        return acl_op::reciprocal_(self);
    }
}
at::Tensor & reciprocal_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::reciprocal_out(self, out);
    } else {
        return acl_op::reciprocal_out(self, out);
    }
}
at::Tensor & reflection_pad1d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::reflection_pad1d_backward_out(grad_output, self, padding, grad_input);
    } else {
        return acl_op::reflection_pad1d_backward_out(grad_output, self, padding, grad_input);
    }
}
at::Tensor & reflection_pad1d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::reflection_pad1d_out(self, padding, out);
    } else {
        return acl_op::reflection_pad1d_out(self, padding, out);
    }
}
at::Tensor & reflection_pad2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::reflection_pad2d_backward_out(grad_output, self, padding, grad_input);
    } else {
        return acl_op::reflection_pad2d_backward_out(grad_output, self, padding, grad_input);
    }
}
at::Tensor & reflection_pad2d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::reflection_pad2d_out(self, padding, out);
    } else {
        return acl_op::reflection_pad2d_out(self, padding, out);
    }
}
at::Tensor & reflection_pad3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    return op_api::reflection_pad3d_backward_out(grad_output, self, padding, grad_input);
}
at::Tensor & reflection_pad3d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::reflection_pad3d_out(self, padding, out);
    } else {
        return acl_op::reflection_pad3d_out(self, padding, out);
    }
}
at::Tensor & relu_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::relu_(self);
    } else {
        return acl_op::relu_(self);
    }
}
at::Tensor & remainder_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::remainder_(self, other);
    } else {
        return acl_op::remainder_(self, other);
    }
}
at::Tensor & remainder_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::remainder_(self, other);
    } else {
        return acl_op::remainder_(self, other);
    }
}
at::Tensor & remainder_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::remainder_out(self, other, out);
    } else {
        return acl_op::remainder_out(self, other, out);
    }
}
at::Tensor & remainder_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::remainder_out(self, other, out);
    } else {
        return acl_op::remainder_out(self, other, out);
    }
}
at::Tensor & renorm_(at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::renorm_(self, p, dim, maxnorm);
    } else {
        return acl_op::renorm_(self, p, dim, maxnorm);
    }
}
at::Tensor & renorm_out(const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::renorm_out(self, p, dim, maxnorm, out);
    } else {
        return acl_op::renorm_out(self, p, dim, maxnorm, out);
    }
}
at::Tensor & replication_pad1d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::replication_pad1d_backward_out(grad_output, self, padding, grad_input);
    } else {
        return acl_op::replication_pad1d_backward_out(grad_output, self, padding, grad_input);
    }
}
at::Tensor & replication_pad1d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::replication_pad1d_out(self, padding, out);
    } else {
        return acl_op::replication_pad1d_out(self, padding, out);
    }
}
at::Tensor & replication_pad2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::replication_pad2d_backward_out(grad_output, self, padding, grad_input);
    } else {
        return acl_op::replication_pad2d_backward_out(grad_output, self, padding, grad_input);
    }
}
at::Tensor & replication_pad2d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::replication_pad2d_out(self, padding, out);
    } else {
        return acl_op::replication_pad2d_out(self, padding, out);
    }
}
at::Tensor & replication_pad3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    return op_api::replication_pad3d_backward_out(grad_output, self, padding, grad_input);
}
at::Tensor & replication_pad3d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::replication_pad3d_out(self, padding, out);
    } else {
        return acl_op::replication_pad3d_out(self, padding, out);
    }
}
at::Tensor & round_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::round_(self);
    } else {
        return acl_op::round_(self);
    }
}
at::Tensor & round_(at::Tensor & self, int64_t decimals){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::round_(self, decimals);
    } else {
        return acl_op::round_(self, decimals);
    }
}
at::Tensor & round_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::round_out(self, out);
    } else {
        return acl_op::round_out(self, out);
    }
}
at::Tensor & round_out(const at::Tensor & self, int64_t decimals, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::round_out(self, decimals, out);
    } else {
        return acl_op::round_out(self, decimals, out);
    }
}
at::Tensor & rrelu_with_noise_(at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(noise)) {
        return op_api::rrelu_with_noise_(self, noise, lower, upper, training, generator);
    } else {
        return acl_op::rrelu_with_noise_(self, noise, lower, upper, training, generator);
    }
}
at::Tensor & rrelu_with_noise_out(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(noise) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::rrelu_with_noise_out(self, noise, lower, upper, training, generator, out);
    } else {
        return acl_op::rrelu_with_noise_out(self, noise, lower, upper, training, generator, out);
    }
}
at::Tensor & rsqrt_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::rsqrt_(self);
    } else {
        return acl_op::rsqrt_(self);
    }
}
at::Tensor & rsqrt_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::rsqrt_out(self, out);
    } else {
        return acl_op::rsqrt_out(self, out);
    }
}
at::Tensor & scatter_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value){
    return acl_op::scatter_(self, dim, index, value);
}
at::Tensor & scatter_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src){
    return acl_op::scatter_(self, dim, index, src);
}
at::Tensor & scatter_add_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(src)) {
        return op_api::scatter_add_(self, dim, index, src);
    } else {
        return acl_op::scatter_add_(self, dim, index, src);
    }
}
at::Tensor & scatter_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::scatter_out(self, dim, index, value, out);
    } else {
        return acl_op::scatter_out(self, dim, index, value, out);
    }
}
at::Tensor & scatter_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(src) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::scatter_out(self, dim, index, src, out);
    } else {
        return acl_op::scatter_out(self, dim, index, src, out);
    }
}
at::Tensor & scatter_update_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t axis){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices) && at_npu::native::FormatHelper::IsOpInputBaseFormat(updates)) {
        return op_api::scatter_update_(self, indices, updates, axis);
    } else {
        return acl_op::scatter_update_(self, indices, updates, axis);
    }
}
at::Tensor & searchsorted_out(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<at::Tensor> & sorter, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(sorted_sequence) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(sorter) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::searchsorted_out(sorted_sequence, self, out_int32, right, side, sorter, out);
    } else {
        return acl_op::searchsorted_out(sorted_sequence, self, out_int32, right, side, sorter, out);
    }
}
at::Tensor & selu_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::selu_(self);
    } else {
        return acl_op::selu_(self);
    }
}
at::Tensor & sgn_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::sgn_out(self, out);
    } else {
        return acl_op::sgn_out(self, out);
    }
}
at::Tensor & sigmoid_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sigmoid_(self);
    } else {
        return acl_op::sigmoid_(self);
    }
}
at::Tensor & sigmoid_backward_out(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::sigmoid_backward_out(grad_output, output, grad_input);
    } else {
        return acl_op::sigmoid_backward_out(grad_output, output, grad_input);
    }
}
at::Tensor & sigmoid_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::sigmoid_out(self, out);
    } else {
        return acl_op::sigmoid_out(self, out);
    }
}
at::Tensor & sign_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sign_(self);
    } else {
        return acl_op::sign_(self);
    }
}
at::Tensor & sign_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::sign_out(self, out);
    } else {
        return acl_op::sign_out(self, out);
    }
}
at::Tensor & signbit_out(const at::Tensor & self, at::Tensor & out){
    return op_api::signbit_out(self, out);
}
at::Tensor & silu_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::silu_(self);
    } else {
        return acl_op::silu_(self);
    }
}
at::Tensor & silu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::silu_backward_out(grad_output, self, grad_input);
    } else {
        return acl_op::silu_backward_out(grad_output, self, grad_input);
    }
}
at::Tensor & silu_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::silu_out(self, out);
    } else {
        return acl_op::silu_out(self, out);
    }
}
at::Tensor & sin_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sin_(self);
    } else {
        return acl_op::sin_(self);
    }
}
at::Tensor & sin_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::sin_out(self, out);
    } else {
        return acl_op::sin_out(self, out);
    }
}
at::Tensor & sinc_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sinc_(self);
    } else {
        return acl_op::sinc_(self);
    }
}
at::Tensor & sinc_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::sinc_out(self, out);
    } else {
        return acl_op::sinc_out(self, out);
    }
}
at::Tensor & sinh_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sinh_(self);
    } else {
        return acl_op::sinh_(self);
    }
}
at::Tensor & sinh_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::sinh_out(self, out);
    } else {
        return acl_op::sinh_out(self, out);
    }
}
at::Tensor & slow_conv3d_forward_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output){
    return acl_op::slow_conv3d_forward_out(self, weight, kernel_size, bias, stride, padding, output);
}
at::Tensor & slow_conv3d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out){
    return acl_op::slow_conv3d_out(self, weight, kernel_size, bias, stride, padding, out);
}
at::Tensor & slow_conv_transpose2d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::slow_conv_transpose2d_out(self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
    } else {
        return acl_op::slow_conv_transpose2d_out(self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
    }
}
at::Tensor & smooth_l1_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::smooth_l1_loss_backward_out(grad_output, self, target, reduction, beta, grad_input);
    } else {
        return acl_op::smooth_l1_loss_backward_out(grad_output, self, target, reduction, beta, grad_input);
    }
}
at::Tensor & smooth_l1_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::smooth_l1_loss_out(self, target, reduction, beta, out);
    } else {
        return acl_op::smooth_l1_loss_out(self, target, reduction, beta, out);
    }
}
at::Tensor & soft_margin_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::soft_margin_loss_backward_out(grad_output, self, target, reduction, grad_input);
    } else {
        return acl_op::soft_margin_loss_backward_out(grad_output, self, target, reduction, grad_input);
    }
}
at::Tensor & soft_margin_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::soft_margin_loss_out(self, target, reduction, out);
    } else {
        return acl_op::soft_margin_loss_out(self, target, reduction, out);
    }
}
at::Tensor & softplus_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::softplus_backward_out(grad_output, self, beta, threshold, grad_input);
    } else {
        return acl_op::softplus_backward_out(grad_output, self, beta, threshold, grad_input);
    }
}
at::Tensor & softplus_out(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::softplus_out(self, beta, threshold, out);
    } else {
        return acl_op::softplus_out(self, beta, threshold, out);
    }
}
at::Tensor & softshrink_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::softshrink_backward_out(grad_output, self, lambd, grad_input);
    } else {
        return acl_op::softshrink_backward_out(grad_output, self, lambd, grad_input);
    }
}
at::Tensor & softshrink_out(const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::softshrink_out(self, lambd, out);
    } else {
        return acl_op::softshrink_out(self, lambd, out);
    }
}
at::Tensor & sqrt_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sqrt_(self);
    } else {
        return acl_op::sqrt_(self);
    }
}
at::Tensor & sqrt_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::sqrt_out(self, out);
    } else {
        return acl_op::sqrt_out(self, out);
    }
}
at::Tensor & stack_out(at::TensorList tensors, int64_t dim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::stack_out(tensors, dim, out);
    } else {
        return acl_op::stack_out(tensors, dim, out);
    }
}
at::Tensor & std_out(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::std_out(self, dim, correction, keepdim, out);
    } else {
        return acl_op::std_out(self, dim, correction, keepdim, out);
    }
}
at::Tensor & sub_(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sub_(self, other, alpha);
    } else {
        return acl_op::sub_(self, other, alpha);
    }
}
at::Tensor & sub_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::sub_(self, other, alpha);
    } else {
        return acl_op::sub_(self, other, alpha);
    }
}
at::Tensor & sub_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::sub_out(self, other, alpha, out);
    } else {
        return acl_op::sub_out(self, other, alpha, out);
    }
}
at::Tensor & sum_out(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::sum_out(self, dim, keepdim, dtype, out);
    } else {
        return acl_op::sum_out(self, dim, keepdim, dtype, out);
    }
}
at::Tensor & sum_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::sum_out(self, dim, keepdim, dtype, out);
    } else {
        return acl_op::sum_out(self, dim, keepdim, dtype, out);
    }
}
at::Tensor & take_out(const at::Tensor & self, const at::Tensor & index, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::take_out(self, index, out);
    } else {
        return acl_op::take_out(self, index, out);
    }
}
at::Tensor & tan_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::tan_(self);
    } else {
        return acl_op::tan_(self);
    }
}
at::Tensor & tan_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::tan_out(self, out);
    } else {
        return acl_op::tan_out(self, out);
    }
}
at::Tensor & tanh_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::tanh_(self);
    } else {
        return acl_op::tanh_(self);
    }
}
at::Tensor & tanh_backward_out(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::tanh_backward_out(grad_output, output, grad_input);
    } else {
        return acl_op::tanh_backward_out(grad_output, output, grad_input);
    }
}
at::Tensor & tanh_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::tanh_out(self, out);
    } else {
        return acl_op::tanh_out(self, out);
    }
}
at::Tensor & threshold_(at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::threshold_(self, threshold, value);
    } else {
        return acl_op::threshold_(self, threshold, value);
    }
}
at::Tensor & threshold_out(const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::threshold_out(self, threshold, value, out);
    } else {
        return acl_op::threshold_out(self, threshold, value, out);
    }
}
at::Tensor & tril_(at::Tensor & self, int64_t diagonal){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::tril_(self, diagonal);
    } else {
        return acl_op::tril_(self, diagonal);
    }
}
at::Tensor & tril_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::tril_out(self, diagonal, out);
    } else {
        return acl_op::tril_out(self, diagonal, out);
    }
}
at::Tensor & triu_(at::Tensor & self, int64_t diagonal){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::triu_(self, diagonal);
    } else {
        return acl_op::triu_(self, diagonal);
    }
}
at::Tensor & triu_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::triu_out(self, diagonal, out);
    } else {
        return acl_op::triu_out(self, diagonal, out);
    }
}
at::Tensor & trunc_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::trunc_(self);
    } else {
        return acl_op::trunc_(self);
    }
}
at::Tensor & trunc_out(const at::Tensor & self, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::trunc_out(self, out);
    } else {
        return acl_op::trunc_out(self, out);
    }
}
at::Tensor & uniform_(at::Tensor & self, double from, double to, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::uniform_(self, from, to, generator);
    } else {
        return acl_op::uniform_(self, from, to, generator);
    }
}
at::Tensor & upsample_bicubic2d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::upsample_bicubic2d_backward_out(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    } else {
        return acl_op::upsample_bicubic2d_backward_out(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    }
}
at::Tensor & upsample_bicubic2d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::upsample_bicubic2d_out(self, output_size, align_corners, scales_h, scales_w, out);
    } else {
        return acl_op::upsample_bicubic2d_out(self, output_size, align_corners, scales_h, scales_w, out);
    }
}
at::Tensor & upsample_bilinear2d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::upsample_bilinear2d_backward_out(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    } else {
        return acl_op::upsample_bilinear2d_backward_out(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    }
}
at::Tensor & upsample_bilinear2d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::upsample_bilinear2d_out(self, output_size, align_corners, scales_h, scales_w, out);
    } else {
        return acl_op::upsample_bilinear2d_out(self, output_size, align_corners, scales_h, scales_w, out);
    }
}
at::Tensor & upsample_linear1d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::upsample_linear1d_out(self, output_size, align_corners, scales, out);
    } else {
        return acl_op::upsample_linear1d_out(self, output_size, align_corners, scales, out);
    }
}
at::Tensor & upsample_nearest1d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::upsample_nearest1d_backward_out(grad_output, output_size, input_size, scales, grad_input);
    } else {
        return acl_op::upsample_nearest1d_backward_out(grad_output, output_size, input_size, scales, grad_input);
    }
}
at::Tensor & upsample_nearest1d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::upsample_nearest1d_out(self, output_size, scales, out);
    } else {
        return acl_op::upsample_nearest1d_out(self, output_size, scales, out);
    }
}
at::Tensor & upsample_nearest2d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::upsample_nearest2d_backward_out(grad_output, output_size, input_size, scales_h, scales_w, grad_input);
    } else {
        return acl_op::upsample_nearest2d_backward_out(grad_output, output_size, input_size, scales_h, scales_w, grad_input);
    }
}
at::Tensor & upsample_nearest2d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::upsample_nearest2d_out(self, output_size, scales_h, scales_w, out);
    } else {
        return acl_op::upsample_nearest2d_out(self, output_size, scales_h, scales_w, out);
    }
}
at::Tensor & upsample_nearest3d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::upsample_nearest3d_backward_out(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
    } else {
        return acl_op::upsample_nearest3d_backward_out(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
    }
}
at::Tensor & upsample_nearest3d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::upsample_nearest3d_out(self, output_size, scales_d, scales_h, scales_w, out);
    } else {
        return acl_op::upsample_nearest3d_out(self, output_size, scales_d, scales_h, scales_w, out);
    }
}
at::Tensor & upsample_trilinear3d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input)) {
        return op_api::upsample_trilinear3d_backward_out(grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input);
    } else {
        return acl_op::upsample_trilinear3d_backward_out(grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input);
    }
}
at::Tensor & upsample_trilinear3d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::upsample_trilinear3d_out(self, output_size, align_corners, scales_d, scales_h, scales_w, out);
    } else {
        return acl_op::upsample_trilinear3d_out(self, output_size, align_corners, scales_d, scales_h, scales_w, out);
    }
}
at::Tensor & var_out(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::var_out(self, dim, correction, keepdim, out);
    } else {
        return acl_op::var_out(self, dim, correction, keepdim, out);
    }
}
at::Tensor & vdot_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::vdot_out(self, other, out);
    } else {
        return acl_op::vdot_out(self, other, out);
    }
}
at::Tensor & where_out(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(condition) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::where_out(condition, self, other, out);
    } else {
        return acl_op::where_out(condition, self, other, out);
    }
}
at::Tensor & xlogy_(at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::xlogy_(self, other);
    } else {
        return acl_op::xlogy_(self, other);
    }
}
at::Tensor & xlogy_(at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::xlogy_(self, other);
    } else {
        return acl_op::xlogy_(self, other);
    }
}
at::Tensor & xlogy_out(const at::Scalar & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::xlogy_out(self, other, out);
    } else {
        return acl_op::xlogy_out(self, other, out);
    }
}
at::Tensor & xlogy_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::xlogy_out(self, other, out);
    } else {
        return acl_op::xlogy_out(self, other, out);
    }
}
at::Tensor & xlogy_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::xlogy_out(self, other, out);
    } else {
        return acl_op::xlogy_out(self, other, out);
    }
}
at::Tensor & zero_(at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::zero_(self);
    } else {
        return acl_op::zero_(self);
    }
}
at::Tensor & zeros_out(at::IntArrayRef size, at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::zeros_out(size, out);
    } else {
        return acl_op::zeros_out(size, out);
    }
}
at::Tensor __lshift__(const at::Tensor & self, const at::Scalar & other){
    return acl_op::__lshift__(self, other);
}
at::Tensor __lshift__(const at::Tensor & self, const at::Tensor & other){
    return acl_op::__lshift__(self, other);
}
at::Tensor __rshift__(const at::Tensor & self, const at::Scalar & other){
    return acl_op::__rshift__(self, other);
}
at::Tensor __rshift__(const at::Tensor & self, const at::Tensor & other){
    return acl_op::__rshift__(self, other);
}
at::Tensor __xor__(const at::Tensor & self, const at::Scalar & other){
    return acl_op::__xor__(self, other);
}
at::Tensor __xor__(const at::Tensor & self, const at::Tensor & other){
    return acl_op::__xor__(self, other);
}
at::Tensor _adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::_adaptive_avg_pool2d(self, output_size);
    } else {
        return acl_op::_adaptive_avg_pool2d(self, output_size);
    }
}
at::Tensor _adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::_adaptive_avg_pool2d_backward(grad_output, self);
    } else {
        return acl_op::_adaptive_avg_pool2d_backward(grad_output, self);
    }
}
at::Tensor _adaptive_avg_pool3d(const at::Tensor & self, at::IntArrayRef output_size){
    return acl_op::_adaptive_avg_pool3d(self, output_size);
}
at::Tensor _adaptive_avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::_adaptive_avg_pool3d_backward(grad_output, self);
    } else {
        return acl_op::_adaptive_avg_pool3d_backward(grad_output, self);
    }
}
at::Tensor _add_relu(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    return acl_op::_add_relu(self, other, alpha);
}
at::Tensor _cdist_backward(const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2, double p, const at::Tensor & cdist){
    return acl_op::_cdist_backward(grad, x1, x2, p, cdist);
}
at::Tensor _cdist_forward(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode){
    return acl_op::_cdist_forward(x1, x2, p, compute_mode);
}
at::Tensor _coalesce_sparse(const at::Tensor & self){
    return sparse::_coalesce_sparse(self);
}
at::Tensor _conj(const at::Tensor & self){
    return acl_op::_conj(self);
}
at::Tensor _conv_depthwise2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
    } else {
        return acl_op::_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
    }
}
at::Tensor _convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
    } else {
        return acl_op::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
    }
}
at::Tensor _ctc_loss_backward(const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad) && at_npu::native::FormatHelper::IsOpInputBaseFormat(log_probs) && at_npu::native::FormatHelper::IsOpInputBaseFormat(targets) && at_npu::native::FormatHelper::IsOpInputBaseFormat(neg_log_likelihood) && at_npu::native::FormatHelper::IsOpInputBaseFormat(log_alpha)) {
        return op_api::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
    } else {
        return acl_op::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
    }
}
at::Tensor _dim_arange(const at::Tensor & like, int64_t dim){
    return acl_op::_dim_arange(like, dim);
}
at::Tensor _dropout_with_byte_mask_backward(const at::Tensor & grad_output, const at::Tensor & mask, double p){
    return acl_op::_dropout_with_byte_mask_backward(grad_output, mask, p);
}
at::Tensor _embedding_bag_backward_symint(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, c10::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx){
    return acl_op::_embedding_bag_backward_symint(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx);
}
at::Tensor _embedding_bag_dense_backward(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx){
    return acl_op::_embedding_bag_dense_backward(grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
}
at::Tensor _embedding_bag_per_sample_weights_backward(const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx){
    return acl_op::_embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, offset2bag, mode, padding_idx);
}
at::Tensor _empty_affine_quantized(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format){
    return acl_op::_empty_affine_quantized(size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
}
at::Tensor _log_softmax(const at::Tensor & self, int64_t dim, bool half_to_float){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::_log_softmax(self, dim, half_to_float);
    } else {
        return acl_op::_log_softmax(self, dim, half_to_float);
    }
}
at::Tensor _log_softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output)) {
        return op_api::_log_softmax_backward_data(grad_output, output, dim, input_dtype);
    } else {
        return acl_op::_log_softmax_backward_data(grad_output, output, dim, input_dtype);
    }
}
at::Tensor _nnpack_spatial_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride){
    return acl_op::_nnpack_spatial_convolution(input, weight, bias, padding, stride);
}
at::Tensor _npu_dropout_gen_mask(const at::Tensor & self, at::IntArrayRef size, double p, int64_t seed, int64_t offset, c10::optional<bool> parallel, c10::optional<bool> sync){
    return acl_op::_npu_dropout_gen_mask(self, size, p, seed, offset, parallel, sync);
}
at::Tensor _npu_silent_check(at::Tensor & input_grad, const at::Tensor & val, at::Tensor & pre_val, at::Tensor & min_val, at::Tensor & max_val, const at::Tensor & val_counter, int64_t c_min_steps, double c_thresh_l1, double c_coeff_l1, double c_thresh_l2, double c_coeff_l2){
    return acl_op::_npu_silent_check(input_grad, val, pre_val, min_val, max_val, val_counter, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2);
}
at::Tensor _pdist_forward(const at::Tensor & self, double p){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::_pdist_forward(self, p);
    } else {
        return acl_op::_pdist_forward(self, p);
    }
}
at::Tensor _prelu_kernel(const at::Tensor & self, const at::Tensor & weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::_prelu_kernel(self, weight);
    } else {
        return acl_op::_prelu_kernel(self, weight);
    }
}
at::Tensor _slow_conv2d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::_slow_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
    } else {
        return acl_op::_slow_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
    }
}
at::Tensor _softmax(const at::Tensor & self, int64_t dim, bool half_to_float){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::_softmax(self, dim, half_to_float);
    } else {
        return acl_op::_softmax(self, dim, half_to_float);
    }
}
at::Tensor _softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output)) {
        return op_api::_softmax_backward_data(grad_output, output, dim, input_dtype);
    } else {
        return acl_op::_softmax_backward_data(grad_output, output, dim, input_dtype);
    }
}
at::Tensor _weight_norm(const at::Tensor & v, const at::Tensor & g, int64_t dim){
    return acl_op::_weight_norm(v, g, dim);
}
at::Tensor abs(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::abs(self);
    } else {
        return acl_op::abs(self);
    }
}
at::Tensor acos(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::acos(self);
    } else {
        return acl_op::acos(self);
    }
}
at::Tensor acosh(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::acosh(self);
    } else {
        return acl_op::acosh(self);
    }
}
at::Tensor adaptive_avg_pool1d(const at::Tensor & self, at::IntArrayRef output_size){
    return acl_op::adaptive_avg_pool1d(self, output_size);
}
at::Tensor adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::adaptive_avg_pool2d(self, output_size);
    } else {
        return acl_op::adaptive_avg_pool2d(self, output_size);
    }
}
at::Tensor adaptive_avg_pool3d(const at::Tensor & self, at::IntArrayRef output_size){
    return acl_op::adaptive_avg_pool3d(self, output_size);
}
at::Tensor adaptive_max_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices){
    return acl_op::adaptive_max_pool2d_backward(grad_output, self, indices);
}
at::Tensor add(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::add(self, other, alpha);
    } else {
        return acl_op::add(self, other, alpha);
    }
}
at::Tensor add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::add(self, other, alpha);
    } else {
        return acl_op::add(self, other, alpha);
    }
}
at::Tensor addbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2)) {
        return op_api::addbmm(self, batch1, batch2, beta, alpha);
    } else {
        return acl_op::addbmm(self, batch1, batch2, beta, alpha);
    }
}
at::Tensor addcdiv(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2)) {
        return op_api::addcdiv(self, tensor1, tensor2, value);
    } else {
        return acl_op::addcdiv(self, tensor1, tensor2, value);
    }
}
at::Tensor addcmul(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2)) {
        return op_api::addcmul(self, tensor1, tensor2, value);
    } else {
        return acl_op::addcmul(self, tensor1, tensor2, value);
    }
}
at::Tensor addmm(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2)) {
        return op_api::addmm(self, mat1, mat2, beta, alpha);
    } else {
        return acl_op::addmm(self, mat1, mat2, beta, alpha);
    }
}
at::Tensor addmv(const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec)) {
        return op_api::addmv(self, mat, vec, beta, alpha);
    } else {
        return acl_op::addmv(self, mat, vec, beta, alpha);
    }
}
at::Tensor addr(const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec2)) {
        return op_api::addr(self, vec1, vec2, beta, alpha);
    } else {
        return acl_op::addr(self, vec1, vec2, beta, alpha);
    }
}
at::Tensor affine_grid_generator(const at::Tensor & theta, at::IntArrayRef size, bool align_corners){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(theta)) {
        return op_api::affine_grid_generator(theta, size, align_corners);
    } else {
        return acl_op::affine_grid_generator(theta, size, align_corners);
    }
}
at::Tensor affine_grid_generator_backward(const at::Tensor & grad, at::IntArrayRef size, bool align_corners){
    return acl_op::affine_grid_generator_backward(grad, size, align_corners);
}
at::Tensor all(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::all(self);
    } else {
        return acl_op::all(self);
    }
}
at::Tensor all(const at::Tensor & self, int64_t dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::all(self, dim, keepdim);
    } else {
        return acl_op::all(self, dim, keepdim);
    }
}
at::Tensor amax(const at::Tensor & self, at::IntArrayRef dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::amax(self, dim, keepdim);
    } else {
        return acl_op::amax(self, dim, keepdim);
    }
}
at::Tensor amin(const at::Tensor & self, at::IntArrayRef dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::amin(self, dim, keepdim);
    } else {
        return acl_op::amin(self, dim, keepdim);
    }
}
at::Tensor angle(const at::Tensor & self){
    return op_api::angle(self);
}
at::Tensor any(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::any(self);
    } else {
        return acl_op::any(self);
    }
}
at::Tensor any(const at::Tensor & self, int64_t dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::any(self, dim, keepdim);
    } else {
        return acl_op::any(self, dim, keepdim);
    }
}
at::Tensor arange(const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::arange(end, dtype, layout, device, pin_memory);
    } else {
        return acl_op::arange(end, dtype, layout, device, pin_memory);
    }
}
at::Tensor arange(const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::arange(start, end, dtype, layout, device, pin_memory);
    } else {
        return acl_op::arange(start, end, dtype, layout, device, pin_memory);
    }
}
at::Tensor arange(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::arange(start, end, step, dtype, layout, device, pin_memory);
    } else {
        return acl_op::arange(start, end, step, dtype, layout, device, pin_memory);
    }
}
at::Tensor argmax(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::argmax(self, dim, keepdim);
    } else {
        return acl_op::argmax(self, dim, keepdim);
    }
}
at::Tensor argmin(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::argmin(self, dim, keepdim);
    } else {
        return acl_op::argmin(self, dim, keepdim);
    }
}
at::Tensor argsort(const at::Tensor & self, at::Dimname dim, bool descending){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::argsort(self, dim, descending);
    } else {
        return acl_op::argsort(self, dim, descending);
    }
}
at::Tensor argsort(const at::Tensor & self, int64_t dim, bool descending){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::argsort(self, dim, descending);
    } else {
        return acl_op::argsort(self, dim, descending);
    }
}
at::Tensor asin(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::asin(self);
    } else {
        return acl_op::asin(self);
    }
}
at::Tensor asinh(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::asinh(self);
    } else {
        return acl_op::asinh(self);
    }
}
at::Tensor atan(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::atan(self);
    } else {
        return acl_op::atan(self);
    }
}
at::Tensor atan2(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::atan2(self, other);
    } else {
        return acl_op::atan2(self, other);
    }
}
at::Tensor atanh(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::atanh(self);
    } else {
        return acl_op::atanh(self);
    }
}
at::Tensor avg_pool2d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    } else {
        return acl_op::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    }
}
at::Tensor avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::avg_pool2d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    } else {
        return acl_op::avg_pool2d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    }
}
at::Tensor avg_pool3d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override){
    return acl_op::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
at::Tensor avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override){
    return acl_op::avg_pool3d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
at::Tensor baddbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1) && at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2)) {
        return op_api::baddbmm(self, batch1, batch2, beta, alpha);
    } else {
        return acl_op::baddbmm(self, batch1, batch2, beta, alpha);
    }
}
at::Tensor batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled){
    return acl_op::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
at::Tensor batch_norm_backward_elemt(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, const at::Tensor & sum_dy, const at::Tensor & sum_dy_xmu, const at::Tensor & count){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(invstd) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(sum_dy) && at_npu::native::FormatHelper::IsOpInputBaseFormat(sum_dy_xmu) && at_npu::native::FormatHelper::IsOpInputBaseFormat(count)) {
        return op_api::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
    } else {
        return acl_op::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
    }
}
at::Tensor batch_norm_elemt(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(invstd)) {
        return op_api::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
    } else {
        return acl_op::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
    }
}
at::Tensor bernoulli(const at::Tensor & self, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::bernoulli(self, generator);
    } else {
        return acl_op::bernoulli(self, generator);
    }
}
at::Tensor bernoulli(const at::Tensor & self, double p, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::bernoulli(self, p, generator);
    } else {
        return acl_op::bernoulli(self, p, generator);
    }
}
at::Tensor binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::binary_cross_entropy(self, target, weight, reduction);
    } else {
        return acl_op::binary_cross_entropy(self, target, weight, reduction);
    }
}
at::Tensor binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
    } else {
        return acl_op::binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
    }
}
at::Tensor binary_cross_entropy_with_logits(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(pos_weight)) {
        return op_api::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
    } else {
        return acl_op::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
    }
}
at::Tensor bincount(const at::Tensor & self, const c10::optional<at::Tensor> & weights, int64_t minlength){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weights)) {
        return op_api::bincount(self, weights, minlength);
    } else {
        return acl_op::bincount(self, weights, minlength);
    }
}
at::Tensor bitwise_and(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::bitwise_and(self, other);
    } else {
        return acl_op::bitwise_and(self, other);
    }
}
at::Tensor bitwise_and(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::bitwise_and(self, other);
    } else {
        return acl_op::bitwise_and(self, other);
    }
}
at::Tensor bitwise_not(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::bitwise_not(self);
    } else {
        return acl_op::bitwise_not(self);
    }
}
at::Tensor bitwise_or(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::bitwise_or(self, other);
    } else {
        return acl_op::bitwise_or(self, other);
    }
}
at::Tensor bitwise_or(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::bitwise_or(self, other);
    } else {
        return acl_op::bitwise_or(self, other);
    }
}
at::Tensor bitwise_xor(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::bitwise_xor(self, other);
    } else {
        return acl_op::bitwise_xor(self, other);
    }
}
at::Tensor bitwise_xor(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::bitwise_xor(self, other);
    } else {
        return acl_op::bitwise_xor(self, other);
    }
}
at::Tensor bmm(const at::Tensor & self, const at::Tensor & mat2){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2)) {
        return op_api::bmm(self, mat2);
    } else {
        return acl_op::bmm(self, mat2);
    }
}
at::Tensor bucketize(const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right){
    return op_api::bucketize(self, boundaries, out_int32, right);
}
at::Tensor bucketize(const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right){
    return op_api::bucketize(self, boundaries, out_int32, right);
}
at::Tensor cat(at::TensorList tensors, at::Dimname dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors)) {
        return op_api::cat(tensors, dim);
    } else {
        return acl_op::cat(tensors, dim);
    }
}
at::Tensor cat(const at::ITensorListRef & tensors, int64_t dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors)) {
        return op_api::cat(tensors, dim);
    } else {
        return acl_op::cat(tensors, dim);
    }
}
at::Tensor cdist(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode){
    return acl_op::cdist(x1, x2, p, compute_mode);
}
at::Tensor ceil(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::ceil(self);
    } else {
        return acl_op::ceil(self);
    }
}
at::Tensor celu(const at::Tensor & self, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::celu(self, alpha);
    } else {
        return acl_op::celu(self, alpha);
    }
}
at::Tensor channel_shuffle(const at::Tensor & self, int64_t groups){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::channel_shuffle(self, groups);
    } else {
        return acl_op::channel_shuffle(self, groups);
    }
}
at::Tensor clamp(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::clamp(self, min, max);
    } else {
        return acl_op::clamp(self, min, max);
    }
}
at::Tensor clamp(const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max)) {
        return op_api::clamp(self, min, max);
    } else {
        return acl_op::clamp(self, min, max);
    }
}
at::Tensor clamp_max(const at::Tensor & self, const at::Scalar & max){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::clamp_max(self, max);
    } else {
        return acl_op::clamp_max(self, max);
    }
}
at::Tensor clamp_max(const at::Tensor & self, const at::Tensor & max){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(max)) {
        return op_api::clamp_max(self, max);
    } else {
        return acl_op::clamp_max(self, max);
    }
}
at::Tensor clamp_min(const at::Tensor & self, const at::Scalar & min){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::clamp_min(self, min);
    } else {
        return acl_op::clamp_min(self, min);
    }
}
at::Tensor clamp_min(const at::Tensor & self, const at::Tensor & min){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(min)) {
        return op_api::clamp_min(self, min);
    } else {
        return acl_op::clamp_min(self, min);
    }
}
at::Tensor col2im(const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::col2im(self, output_size, kernel_size, dilation, padding, stride);
    } else {
        return acl_op::col2im(self, output_size, kernel_size, dilation, padding, stride);
    }
}
at::Tensor complex(const at::Tensor & real, const at::Tensor & imag){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(real) && at_npu::native::FormatHelper::IsOpInputBaseFormat(imag)) {
        return op_api::complex(real, imag);
    } else {
        return acl_op::complex(real, imag);
    }
}
at::Tensor constant_pad_nd(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::constant_pad_nd(self, pad, value);
    } else {
        return acl_op::constant_pad_nd(self, pad, value);
    }
}
at::Tensor conv_tbc(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::conv_tbc(self, weight, bias, pad);
    } else {
        return acl_op::conv_tbc(self, weight, bias, pad);
    }
}
at::Tensor conv_transpose2d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation){
    return acl_op::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
at::Tensor conv_transpose3d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation){
    return acl_op::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
at::Tensor convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
    } else {
        return acl_op::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
    }
}
at::Tensor convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
    } else {
        return acl_op::convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
    }
}
at::Tensor cos(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::cos(self);
    } else {
        return acl_op::cos(self);
    }
}
at::Tensor cosh(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::cosh(self);
    } else {
        return acl_op::cosh(self);
    }
}
at::Tensor count_nonzero(const at::Tensor & self, at::IntArrayRef dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::count_nonzero(self, dim);
    } else {
        return acl_op::count_nonzero(self, dim);
    }
}
at::Tensor count_nonzero(const at::Tensor & self, c10::optional<int64_t> dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::count_nonzero(self, dim);
    } else {
        return acl_op::count_nonzero(self, dim);
    }
}
at::Tensor ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(log_probs) && at_npu::native::FormatHelper::IsOpInputBaseFormat(targets)) {
        return op_api::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    } else {
        return acl_op::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    }
}
at::Tensor ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(log_probs) && at_npu::native::FormatHelper::IsOpInputBaseFormat(targets) && at_npu::native::FormatHelper::IsOpInputBaseFormat(input_lengths) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target_lengths)) {
        return op_api::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    } else {
        return acl_op::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    }
}
at::Tensor cumsum(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::cumsum(self, dim, dtype);
    } else {
        return acl_op::cumsum(self, dim, dtype);
    }
}
at::Tensor dequantize(const at::Tensor & self){
    return acl_op::dequantize(self);
}
at::Tensor diag(const at::Tensor & self, int64_t diagonal){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::diag(self, diagonal);
    } else {
        return acl_op::diag(self, diagonal);
    }
}
at::Tensor div(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::div(self, other);
    } else {
        return acl_op::div(self, other);
    }
}
at::Tensor div(const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::div(self, other, rounding_mode);
    } else {
        return acl_op::div(self, other, rounding_mode);
    }
}
at::Tensor div(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::div(self, other);
    } else {
        return acl_op::div(self, other);
    }
}
at::Tensor div(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::div(self, other, rounding_mode);
    } else {
        return acl_op::div(self, other, rounding_mode);
    }
}
at::Tensor dot(const at::Tensor & self, const at::Tensor & tensor){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor)) {
        return op_api::dot(self, tensor);
    } else {
        return acl_op::dot(self, tensor);
    }
}
at::Tensor dropout(const at::Tensor & input, double p, bool train){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input)) {
        return op_api::dropout(input, p, train);
    } else {
        return acl_op::dropout(input, p, train);
    }
}
at::Tensor dropout_with_byte_mask(const at::Tensor & self, double p, bool train){
    return acl_op::dropout_with_byte_mask(self, p, train);
}
at::Tensor elu(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::elu(self, alpha, scale, input_scale);
    } else {
        return acl_op::elu(self, alpha, scale, input_scale);
    }
}
at::Tensor elu_backward(const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self_or_result)) {
        return op_api::elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result);
    } else {
        return acl_op::elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result);
    }
}
at::Tensor embedding_backward_symint(const at::Tensor & grad, const at::Tensor & indices, c10::SymInt num_weights, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::embedding_backward_symint(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
    } else {
        return acl_op::embedding_backward_symint(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
    }
}
at::Tensor embedding_dense_backward(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
    } else {
        return acl_op::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
    }
}
at::Tensor embedding_symint(const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse);
    } else {
        return acl_op::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse);
    }
}
at::Tensor eq(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::eq(self, other);
    } else {
        return acl_op::eq(self, other);
    }
}
at::Tensor eq(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::eq(self, other);
    } else {
        return acl_op::eq(self, other);
    }
}
at::Tensor erf(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::erf(self);
    } else {
        return acl_op::erf(self);
    }
}
at::Tensor erfc(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::erfc(self);
    } else {
        return acl_op::erfc(self);
    }
}
at::Tensor erfinv(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::erfinv(self);
    } else {
        return acl_op::erfinv(self);
    }
}
at::Tensor exp(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::exp(self);
    } else {
        return acl_op::exp(self);
    }
}
at::Tensor exp2(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::exp2(self);
    } else {
        return acl_op::exp2(self);
    }
}
at::Tensor expm1(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::expm1(self);
    } else {
        return acl_op::expm1(self);
    }
}
at::Tensor eye(int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::eye(n, dtype, layout, device, pin_memory);
    } else {
        return acl_op::eye(n, dtype, layout, device, pin_memory);
    }
}
at::Tensor eye(int64_t n, int64_t m, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::eye(n, m, dtype, layout, device, pin_memory);
    } else {
        return acl_op::eye(n, m, dtype, layout, device, pin_memory);
    }
}
at::Tensor fast_gelu(const at::Tensor & self){
    return acl_op::fast_gelu(self);
}
at::Tensor fft_rfft(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm){
    return op_api::fft_rfft(self, n, dim, norm);
}
at::Tensor flip(const at::Tensor & self, at::IntArrayRef dims){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::flip(self, dims);
    } else {
        return acl_op::flip(self, dims);
    }
}
at::Tensor floor(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::floor(self);
    } else {
        return acl_op::floor(self);
    }
}
at::Tensor floor_divide(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::floor_divide(self, other);
    } else {
        return acl_op::floor_divide(self, other);
    }
}
at::Tensor floor_divide(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::floor_divide(self, other);
    } else {
        return acl_op::floor_divide(self, other);
    }
}
at::Tensor fmod(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::fmod(self, other);
    } else {
        return acl_op::fmod(self, other);
    }
}
at::Tensor fmod(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::fmod(self, other);
    } else {
        return acl_op::fmod(self, other);
    }
}
at::Tensor frac(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::frac(self);
    } else {
        return acl_op::frac(self);
    }
}
at::Tensor gather(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index)) {
        return op_api::gather(self, dim, index, sparse_grad);
    } else {
        return acl_op::gather(self, dim, index, sparse_grad);
    }
}
at::Tensor gather(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index)) {
        return op_api::gather(self, dim, index, sparse_grad);
    } else {
        return acl_op::gather(self, dim, index, sparse_grad);
    }
}
at::Tensor ge(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::ge(self, other);
    } else {
        return acl_op::ge(self, other);
    }
}
at::Tensor ge(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::ge(self, other);
    } else {
        return acl_op::ge(self, other);
    }
}
at::Tensor gelu(const at::Tensor & self, c10::string_view approximate){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::gelu(self, approximate);
    } else {
        return acl_op::gelu(self, approximate);
    }
}
at::Tensor gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::gelu_backward(grad_output, self, approximate);
    } else {
        return acl_op::gelu_backward(grad_output, self, approximate);
    }
}
at::Tensor glu(const at::Tensor & self, int64_t dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::glu(self, dim);
    } else {
        return acl_op::glu(self, dim);
    }
}
at::Tensor glu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::glu_backward(grad_output, self, dim);
    } else {
        return acl_op::glu_backward(grad_output, self, dim);
    }
}
at::Tensor grid_sampler_2d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grid)) {
        return op_api::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
    } else {
        return acl_op::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
    }
}
at::Tensor grid_sampler_3d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(grid)) {
        return op_api::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
    } else {
        return acl_op::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
    }
}
at::Tensor gt(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::gt(self, other);
    } else {
        return acl_op::gt(self, other);
    }
}
at::Tensor gt(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::gt(self, other);
    } else {
        return acl_op::gt(self, other);
    }
}
at::Tensor hardshrink(const at::Tensor & self, const at::Scalar & lambd){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardshrink(self, lambd);
    } else {
        return acl_op::hardshrink(self, lambd);
    }
}
at::Tensor hardshrink_backward(const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardshrink_backward(grad_out, self, lambd);
    } else {
        return acl_op::hardshrink_backward(grad_out, self, lambd);
    }
}
at::Tensor hardsigmoid(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardsigmoid(self);
    } else {
        return acl_op::hardsigmoid(self);
    }
}
at::Tensor hardsigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardsigmoid_backward(grad_output, self);
    } else {
        return acl_op::hardsigmoid_backward(grad_output, self);
    }
}
at::Tensor hardswish(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardswish(self);
    } else {
        return acl_op::hardswish(self);
    }
}
at::Tensor hardswish_backward(const at::Tensor & grad_output, const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardswish_backward(grad_output, self);
    } else {
        return acl_op::hardswish_backward(grad_output, self);
    }
}
at::Tensor hardtanh(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardtanh(self, min_val, max_val);
    } else {
        return acl_op::hardtanh(self, min_val, max_val);
    }
}
at::Tensor hardtanh_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::hardtanh_backward(grad_output, self, min_val, max_val);
    } else {
        return acl_op::hardtanh_backward(grad_output, self, min_val, max_val);
    }
}
at::Tensor histc(const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::histc(self, bins, min, max);
    } else {
        return acl_op::histc(self, bins, min, max);
    }
}
at::Tensor im2col(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::im2col(self, kernel_size, dilation, padding, stride);
    } else {
        return acl_op::im2col(self, kernel_size, dilation, padding, stride);
    }
}
at::Tensor index(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::index(self, indices);
    } else {
        return acl_op::index(self, indices);
    }
}
at::Tensor index_add(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha){
    return acl_op::index_add(self, dim, index, source, alpha);
}
at::Tensor index_add(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(source)) {
        return op_api::index_add(self, dim, index, source, alpha);
    } else {
        return acl_op::index_add(self, dim, index, source, alpha);
    }
}
at::Tensor index_copy(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(source)) {
        return op_api::index_copy(self, dim, index, source);
    } else {
        return acl_op::index_copy(self, dim, index, source);
    }
}
at::Tensor index_fill(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index)) {
        return op_api::index_fill(self, dim, index, value);
    } else {
        return acl_op::index_fill(self, dim, index, value);
    }
}
at::Tensor index_fill(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(value)) {
        return op_api::index_fill(self, dim, index, value);
    } else {
        return acl_op::index_fill(self, dim, index, value);
    }
}
at::Tensor index_put(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values)) {
        return op_api::index_put(self, indices, values, accumulate);
    } else {
        return acl_op::index_put(self, indices, values, accumulate);
    }
}
at::Tensor index_select(const at::Tensor & self, at::Dimname dim, const at::Tensor & index){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index)) {
        return op_api::index_select(self, dim, index);
    } else {
        return acl_op::index_select(self, dim, index);
    }
}
at::Tensor index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index)) {
        return op_api::index_select(self, dim, index);
    } else {
        return acl_op::index_select(self, dim, index);
    }
}
at::Tensor int_repr(const at::Tensor & self){
    return acl_op::int_repr(self);
}
at::Tensor inverse(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::inverse(self);
    } else {
        return acl_op::inverse(self);
    }
}
at::Tensor isclose(const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::isclose(self, other, rtol, atol, equal_nan);
    } else {
        return acl_op::isclose(self, other, rtol, atol, equal_nan);
    }
}
at::Tensor isfinite(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::isfinite(self);
    } else {
        return acl_op::isfinite(self);
    }
}
at::Tensor isin(const at::Tensor & element, const at::Scalar & test_elements, bool assume_unique, bool invert){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(element)) {
        return op_api::isin(element, test_elements, assume_unique, invert);
    } else {
        return acl_op::isin(element, test_elements, assume_unique, invert);
    }
}
at::Tensor kl_div(const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::kl_div(self, target, reduction, log_target);
    } else {
        return acl_op::kl_div(self, target, reduction, log_target);
    }
}
at::Tensor kl_div_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target){
    return acl_op::kl_div_backward(grad_output, self, target, reduction, log_target);
}
at::Tensor l1_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::l1_loss(self, target, reduction);
    } else {
        return acl_op::l1_loss(self, target, reduction);
    }
}
at::Tensor l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::l1_loss_backward(grad_output, self, target, reduction);
    } else {
        return acl_op::l1_loss_backward(grad_output, self, target, reduction);
    }
}
at::Tensor le(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::le(self, other);
    } else {
        return acl_op::le(self, other);
    }
}
at::Tensor le(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::le(self, other);
    } else {
        return acl_op::le(self, other);
    }
}
at::Tensor leaky_relu(const at::Tensor & self, const at::Scalar & negative_slope){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::leaky_relu(self, negative_slope);
    } else {
        return acl_op::leaky_relu(self, negative_slope);
    }
}
at::Tensor leaky_relu_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::leaky_relu_backward(grad_output, self, negative_slope, self_is_result);
    } else {
        return acl_op::leaky_relu_backward(grad_output, self, negative_slope, self_is_result);
    }
}
at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(end)) {
        return op_api::lerp(self, end, weight);
    } else {
        return acl_op::lerp(self, end, weight);
    }
}
at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(end) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight)) {
        return op_api::lerp(self, end, weight);
    } else {
        return acl_op::lerp(self, end, weight);
    }
}
at::Tensor linalg_cross(const at::Tensor & self, const at::Tensor & other, int64_t dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::linalg_cross(self, other, dim);
    } else {
        return acl_op::linalg_cross(self, other, dim);
    }
}
at::Tensor linalg_matrix_norm(const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    return acl_op::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
}
at::Tensor linalg_matrix_norm(const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    return acl_op::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
}
at::Tensor linalg_norm(const at::Tensor & self, c10::string_view ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    return acl_op::linalg_norm(self, ord, dim, keepdim, dtype);
}
at::Tensor linalg_norm(const at::Tensor & self, const c10::optional<at::Scalar> & ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    return acl_op::linalg_norm(self, ord, dim, keepdim, dtype);
}
at::Tensor linalg_solve_triangular(const at::Tensor & self, const at::Tensor & B, bool upper, bool left, bool unitriangular){
    return op_api::linalg_solve_triangular(self, B, upper, left, unitriangular);
}
at::Tensor linalg_svdvals(const at::Tensor & A, c10::optional<c10::string_view> driver){
    return acl_op::linalg_svdvals(A, driver);
}
at::Tensor linalg_vector_norm(const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::linalg_vector_norm(self, ord, dim, keepdim, dtype);
    } else {
        return acl_op::linalg_vector_norm(self, ord, dim, keepdim, dtype);
    }
}
at::Tensor linspace(const at::Scalar & start, const at::Scalar & end, int64_t steps, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::linspace(start, end, steps, dtype, layout, device, pin_memory);
    } else {
        return acl_op::linspace(start, end, steps, dtype, layout, device, pin_memory);
    }
}
at::Tensor log(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::log(self);
    } else {
        return acl_op::log(self);
    }
}
at::Tensor log10(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::log10(self);
    } else {
        return acl_op::log10(self);
    }
}
at::Tensor log1p(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::log1p(self);
    } else {
        return acl_op::log1p(self);
    }
}
at::Tensor log2(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::log2(self);
    } else {
        return acl_op::log2(self);
    }
}
at::Tensor log_sigmoid(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::log_sigmoid(self);
    } else {
        return acl_op::log_sigmoid(self);
    }
}
at::Tensor log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(buffer)) {
        return op_api::log_sigmoid_backward(grad_output, self, buffer);
    } else {
        return acl_op::log_sigmoid_backward(grad_output, self, buffer);
    }
}
at::Tensor log_softmax(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype){
    return acl_op::log_softmax(self, dim, dtype);
}
at::Tensor log_softmax(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype){
    return acl_op::log_softmax(self, dim, dtype);
}
at::Tensor logaddexp(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::logaddexp(self, other);
    } else {
        return acl_op::logaddexp(self, other);
    }
}
at::Tensor logaddexp2(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::logaddexp2(self, other);
    } else {
        return acl_op::logaddexp2(self, other);
    }
}
at::Tensor logical_and(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::logical_and(self, other);
    } else {
        return acl_op::logical_and(self, other);
    }
}
at::Tensor logical_not(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::logical_not(self);
    } else {
        return acl_op::logical_not(self);
    }
}
at::Tensor logical_or(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::logical_or(self, other);
    } else {
        return acl_op::logical_or(self, other);
    }
}
at::Tensor logical_xor(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::logical_xor(self, other);
    } else {
        return acl_op::logical_xor(self, other);
    }
}
at::Tensor logspace(const at::Scalar & start, const at::Scalar & end, int64_t steps, double base, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    return acl_op::logspace(start, end, steps, base, dtype, layout, device, pin_memory);
}
at::Tensor logsumexp(const at::Tensor & self, at::DimnameList dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::logsumexp(self, dim, keepdim);
    } else {
        return acl_op::logsumexp(self, dim, keepdim);
    }
}
at::Tensor logsumexp(const at::Tensor & self, at::IntArrayRef dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::logsumexp(self, dim, keepdim);
    } else {
        return acl_op::logsumexp(self, dim, keepdim);
    }
}
at::Tensor lt(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::lt(self, other);
    } else {
        return acl_op::lt(self, other);
    }
}
at::Tensor lt(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::lt(self, other);
    } else {
        return acl_op::lt(self, other);
    }
}
at::Tensor masked_select(const at::Tensor & self, const at::Tensor & mask){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mask)) {
        return op_api::masked_select(self, mask);
    } else {
        return acl_op::masked_select(self, mask);
    }
}
at::Tensor matmul(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::matmul(self, other);
    } else {
        return acl_op::matmul(self, other);
    }
}
at::Tensor max(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::max(self);
    } else {
        return acl_op::max(self);
    }
}
at::Tensor max_pool2d_with_indices_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
    } else {
        return acl_op::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
    }
}
at::Tensor max_pool3d_with_indices_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices){
    return acl_op::max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
at::Tensor max_sparse(const at::Tensor & self){
    return sparse::max_sparse(self);
}
at::Tensor max_unpool2d(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::max_unpool2d(self, indices, output_size);
    } else {
        return acl_op::max_unpool2d(self, indices, output_size);
    }
}
at::Tensor max_unpool3d(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::max_unpool3d(self, indices, output_size, stride, padding);
    } else {
        return acl_op::max_unpool3d(self, indices, output_size, stride, padding);
    }
}
at::Tensor maximum(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::maximum(self, other);
    } else {
        return acl_op::maximum(self, other);
    }
}
at::Tensor mean(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::mean(self, dim, keepdim, dtype);
    } else {
        return acl_op::mean(self, dim, keepdim, dtype);
    }
}
at::Tensor mean(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::mean(self, dim, keepdim, dtype);
    } else {
        return acl_op::mean(self, dim, keepdim, dtype);
    }
}
at::Tensor mean(const at::Tensor & self, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::mean(self, dtype);
    } else {
        return acl_op::mean(self, dtype);
    }
}
at::Tensor median(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::median(self);
    } else {
        return acl_op::median(self);
    }
}
at::Tensor min(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::min(self);
    } else {
        return acl_op::min(self);
    }
}
at::Tensor minimum(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::minimum(self, other);
    } else {
        return acl_op::minimum(self, other);
    }
}
at::Tensor mish(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::mish(self);
    } else {
        return acl_op::mish(self);
    }
}
at::Tensor mish_backward(const at::Tensor & grad_output, const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::mish_backward(grad_output, self);
    } else {
        return acl_op::mish_backward(grad_output, self);
    }
}
at::Tensor mm(const at::Tensor & self, const at::Tensor & mat2){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2)) {
        return op_api::mm(self, mat2);
    } else {
        return acl_op::mm(self, mat2);
    }
}
at::Tensor mse_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::mse_loss(self, target, reduction);
    } else {
        return acl_op::mse_loss(self, target, reduction);
    }
}
at::Tensor mse_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::mse_loss_backward(grad_output, self, target, reduction);
    } else {
        return acl_op::mse_loss_backward(grad_output, self, target, reduction);
    }
}
at::Tensor mul(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::mul(self, other);
    } else {
        return acl_op::mul(self, other);
    }
}
at::Tensor mul(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::mul(self, other);
    } else {
        return acl_op::mul(self, other);
    }
}
at::Tensor multilabel_margin_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::multilabel_margin_loss(self, target, reduction);
    } else {
        return acl_op::multilabel_margin_loss(self, target, reduction);
    }
}
at::Tensor multinomial(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::multinomial(self, num_samples, replacement, generator);
    } else {
        return acl_op::multinomial(self, num_samples, replacement, generator);
    }
}
at::Tensor mv(const at::Tensor & self, const at::Tensor & vec){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(vec)) {
        return op_api::mv(self, vec);
    } else {
        return acl_op::mv(self, vec);
    }
}
at::Tensor nan_to_num(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::nan_to_num(self, nan, posinf, neginf);
    } else {
        return acl_op::nan_to_num(self, nan, posinf, neginf);
    }
}
at::Tensor nanmedian(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::nanmedian(self);
    } else {
        return acl_op::nanmedian(self);
    }
}
at::Tensor nansum(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    return op_api::nansum(self, dim, keepdim, dtype);
}
at::Tensor native_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double scale){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mask)) {
        return op_api::native_dropout_backward(grad_output, mask, scale);
    } else {
        return acl_op::native_dropout_backward(grad_output, mask, scale);
    }
}
at::Tensor ne(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::ne(self, other);
    } else {
        return acl_op::ne(self, other);
    }
}
at::Tensor ne(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::ne(self, other);
    } else {
        return acl_op::ne(self, other);
    }
}
at::Tensor neg(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::neg(self);
    } else {
        return acl_op::neg(self);
    }
}
at::Tensor nll_loss(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index){
    return acl_op::nll_loss(self, target, weight, reduction, ignore_index);
}
at::Tensor nll_loss2d(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index){
    return acl_op::nll_loss2d(self, target, weight, reduction, ignore_index);
}
at::Tensor nll_loss2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight)) {
        return op_api::nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
    } else {
        return acl_op::nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
    }
}
at::Tensor nll_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight)) {
        return op_api::nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
    } else {
        return acl_op::nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
    }
}
at::Tensor nonzero(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::nonzero(self);
    } else {
        return acl_op::nonzero(self);
    }
}
at::Tensor norm(const at::Tensor & self, const at::Scalar & p){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::norm(self, p);
    } else {
        return acl_op::norm(self, p);
    }
}
at::Tensor norm(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::norm(self, p, dim, keepdim);
    } else {
        return acl_op::norm(self, p, dim, keepdim);
    }
}
at::Tensor norm(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::norm(self, p, dim, keepdim, dtype);
    } else {
        return acl_op::norm(self, p, dim, keepdim, dtype);
    }
}
at::Tensor norm(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::ScalarType dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::norm(self, p, dtype);
    } else {
        return acl_op::norm(self, p, dtype);
    }
}
at::Tensor normal(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean) && at_npu::native::FormatHelper::IsOpInputBaseFormat(std)) {
        return op_api::normal(mean, std, generator);
    } else {
        return acl_op::normal(mean, std, generator);
    }
}
at::Tensor normal(const at::Tensor & mean, double std, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(mean)) {
        return op_api::normal(mean, std, generator);
    } else {
        return acl_op::normal(mean, std, generator);
    }
}
at::Tensor normal(double mean, const at::Tensor & std, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(std)) {
        return op_api::normal(mean, std, generator);
    } else {
        return acl_op::normal(mean, std, generator);
    }
}
at::Tensor normal(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::normal(mean, std, size, generator, dtype, layout, device, pin_memory);
    } else {
        return acl_op::normal(mean, std, size, generator, dtype, layout, device, pin_memory);
    }
}
at::Tensor npu_alloc_float_status(const at::Tensor & self){
    return acl_op::npu_alloc_float_status(self);
}
at::Tensor npu_anchor_response_flags(const at::Tensor & self, at::IntArrayRef featmap_size, at::IntArrayRef stride, int64_t num_base_anchors){
    return acl_op::npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors);
}
at::Tensor npu_anti_quant(const at::Tensor & x, const at::Tensor & scale, const c10::optional<at::Tensor> & offset, c10::optional<at::ScalarType> dst_dtype, c10::optional<at::ScalarType> src_dtype){
    return op_api::npu_anti_quant(x, scale, offset, dst_dtype, src_dtype);
}
at::Tensor npu_binary_cross_entropy_with_logits_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight_opt, const c10::optional<at::Tensor> & pos_weight_opt, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_opt) && at_npu::native::FormatHelper::IsOpInputBaseFormat(pos_weight_opt)) {
        return op_api::npu_binary_cross_entropy_with_logits_backward(grad_output, self, target, weight_opt, pos_weight_opt, reduction);
    } else {
        return acl_op::npu_binary_cross_entropy_with_logits_backward(grad_output, self, target, weight_opt, pos_weight_opt, reduction);
    }
}
at::Tensor npu_bmmV2(const at::Tensor & self, const at::Tensor & mat2, at::IntArrayRef output_sizes){
    return acl_op::npu_bmmV2(self, mat2, output_sizes);
}
at::Tensor npu_bmm_v2_mat1_backward_symint(const at::Tensor & grad, const at::Tensor & mat1, const at::Tensor & mat2, c10::SymIntArrayRef size){
    return acl_op::npu_bmm_v2_mat1_backward_symint(grad, mat1, mat2, size);
}
at::Tensor npu_bmm_v2_mat2_backward_symint(const at::Tensor & grad, const at::Tensor & mat1, const at::Tensor & mat2, c10::SymIntArrayRef size){
    return acl_op::npu_bmm_v2_mat2_backward_symint(grad, mat1, mat2, size);
}
at::Tensor npu_bounding_box_decode(const at::Tensor & rois, const at::Tensor & deltas, double means0, double means1, double means2, double means3, double stds0, double stds1, double stds2, double stds3, at::IntArrayRef max_shape, double wh_ratio_clip){
    return acl_op::npu_bounding_box_decode(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip);
}
at::Tensor npu_bounding_box_encode(const at::Tensor & anchor_box, const at::Tensor & ground_truth_box, double means0, double means1, double means2, double means3, double stds0, double stds1, double stds2, double stds3){
    return acl_op::npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3);
}
at::Tensor npu_broadcast(const at::Tensor & self, at::IntArrayRef size){
    return acl_op::npu_broadcast(self, size);
}
at::Tensor npu_ciou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag){
    return acl_op::npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}
at::Tensor npu_clear_float_status(const at::Tensor & self, int64_t mode){
    return acl_op::npu_clear_float_status(self, mode);
}
at::Tensor npu_confusion_transpose(const at::Tensor & self, at::IntArrayRef perm, at::IntArrayRef shape, bool transpose_first){
    return acl_op::npu_confusion_transpose(self, perm, shape, transpose_first);
}
at::Tensor npu_confusion_transpose_backward_symint(const at::Tensor & grad, at::IntArrayRef perm, c10::SymIntArrayRef shape, bool transpose_first){
    return acl_op::npu_confusion_transpose_backward_symint(grad, perm, shape, transpose_first);
}
at::Tensor npu_conv2d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups){
    return acl_op::npu_conv2d(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor npu_conv3d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups){
    return acl_op::npu_conv3d(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor npu_conv_transpose2d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups){
    return acl_op::npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups);
}
at::Tensor npu_convert_weight_to_int4pack(const at::Tensor & weight, int64_t inner_k_tiles){
    return op_api::npu_convert_weight_to_int4pack(weight, inner_k_tiles);
}
at::Tensor npu_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups){
    return acl_op::npu_convolution(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor npu_convolution_transpose(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups){
    return acl_op::npu_convolution_transpose(input, weight, bias, padding, output_padding, stride, dilation, groups);
}
at::Tensor npu_diou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode){
    return acl_op::npu_diou(self, gtboxes, trans, is_cross, mode);
}
at::Tensor npu_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double p){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(mask)) {
        return op_api::npu_dropout_backward(grad_output, mask, p);
    } else {
        return acl_op::npu_dropout_backward(grad_output, mask, p);
    }
}
at::Tensor npu_dropout_gen_mask(at::IntArrayRef size, double p, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    return acl_op::npu_dropout_gen_mask(size, p, dtype, layout, device, pin_memory);
}
at::Tensor npu_dtype_cast(const at::Tensor & self, at::ScalarType dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::npu_dtype_cast(self, dtype);
    } else {
        return acl_op::npu_dtype_cast(self, dtype);
    }
}
at::Tensor npu_dtype_cast_backward(const at::Tensor & grad, at::ScalarType dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad)) {
        return op_api::npu_dtype_cast_backward(grad, dtype);
    } else {
        return acl_op::npu_dtype_cast_backward(grad, dtype);
    }
}
at::Tensor npu_fast_gelu(const at::Tensor & self){
    return acl_op::npu_fast_gelu(self);
}
at::Tensor npu_fast_gelu_backward(const at::Tensor & grad, const at::Tensor & self){
    return acl_op::npu_fast_gelu_backward(grad, self);
}
at::Tensor npu_ffn(const at::Tensor & x, const at::Tensor & weight1, const at::Tensor & weight2, c10::string_view activation, at::OptionalIntArrayRef expert_tokens, at::OptionalIntArrayRef expert_tokens_index, const c10::optional<at::Tensor> & bias1, const c10::optional<at::Tensor> & bias2, const c10::optional<at::Tensor> & scale, const c10::optional<at::Tensor> & offset, const c10::optional<at::Tensor> & deq_scale1, const c10::optional<at::Tensor> & deq_scale2, const c10::optional<at::Tensor> & antiquant_scale1, const c10::optional<at::Tensor> & antiquant_scale2, const c10::optional<at::Tensor> & antiquant_offset1, const c10::optional<at::Tensor> & antiquant_offset2, c10::optional<int64_t> inner_precise, c10::optional<at::ScalarType> output_dtype){
    return op_api::npu_ffn(x, weight1, weight2, activation, expert_tokens, expert_tokens_index, bias1, bias2, scale, offset, deq_scale1, deq_scale2, antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2, inner_precise, output_dtype);
}
at::Tensor npu_fused_attention_score(const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose){
    return acl_op::npu_fused_attention_score(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}
at::Tensor npu_geglu_grad(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & gelu, int64_t dim, int64_t approximate, bool activate_left){
    return op_api::npu_geglu_grad(grad_output, self, gelu, dim, approximate, activate_left);
}
at::Tensor npu_get_float_status(const at::Tensor & self, int64_t mode){
    return acl_op::npu_get_float_status(self, mode);
}
at::Tensor npu_giou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode){
    return acl_op::npu_giou(self, gtboxes, trans, is_cross, mode);
}
at::Tensor npu_grid_assign_positive(const at::Tensor & self, const at::Tensor & overlaps, const at::Tensor & box_responsible_flags, const at::Tensor & max_overlaps, const at::Tensor & argmax_overlaps, const at::Tensor & gt_max_overlaps, const at::Tensor & gt_argmax_overlaps, int64_t num_gts, double pos_iou_thr, double min_pos_iou, bool gt_max_assign_all){
    return acl_op::npu_grid_assign_positive(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all);
}
at::Tensor npu_incre_flash_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_lengths, const c10::optional<at::Tensor> & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & block_table, const c10::optional<at::Tensor> & dequant_scale1, const c10::optional<at::Tensor> & quant_scale1, const c10::optional<at::Tensor> & dequant_scale2, const c10::optional<at::Tensor> & quant_scale2, const c10::optional<at::Tensor> & quant_offset2, const c10::optional<at::Tensor> & kv_padding_size, int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise){
    return op_api::npu_incre_flash_attention(query, key, value, padding_mask, atten_mask, actual_seq_lengths, antiquant_scale, antiquant_offset, block_table, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, kv_padding_size, num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise);
}
at::Tensor npu_indexing(const at::Tensor & self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask){
    return acl_op::npu_indexing(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
}
at::Tensor npu_iou(const at::Tensor & bboxes, const at::Tensor & gtboxes, int64_t mode){
    return acl_op::npu_iou(bboxes, gtboxes, mode);
}
at::Tensor npu_layer_norm_eval(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps){
    return acl_op::npu_layer_norm_eval(input, normalized_shape, weight, bias, eps);
}
at::Tensor npu_linear(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::npu_linear(input, weight, bias);
    } else {
        return acl_op::npu_linear(input, weight, bias);
    }
}
at::Tensor npu_masked_fill_range(const at::Tensor & self, const at::Tensor & start, const at::Tensor & end, const at::Tensor & value, int64_t axis){
    return acl_op::npu_masked_fill_range(self, start, end, value, axis);
}
at::Tensor npu_masked_softmax_with_rel_pos_bias(const at::Tensor & x, const c10::optional<at::Tensor> & atten_mask, const at::Tensor & relative_pos_bias, double scale_value, int64_t inner_precision_mode){
    return op_api::npu_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode);
}
at::Tensor npu_max_backward_symint(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, c10::SymIntArrayRef sizes, bool keepdim){
    return acl_op::npu_max_backward_symint(grad, dim, indices, sizes, keepdim);
}
at::Tensor npu_min_backward_symint(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, c10::SymIntArrayRef sizes, bool keepdim){
    return acl_op::npu_min_backward_symint(grad, dim, indices, sizes, keepdim);
}
at::Tensor npu_mish(const at::Tensor & self){
    return acl_op::npu_mish(self);
}
at::Tensor npu_mish_backward(const at::Tensor & grad, const at::Tensor & input){
    return acl_op::npu_mish_backward(grad, input);
}
at::Tensor npu_mm_all_reduce_base(const at::Tensor & x1, const at::Tensor & x2, c10::string_view hcom, c10::string_view reduce_op, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & x3, const c10::optional<at::Tensor> & dequant_scale, int64_t antiquant_group_size, int64_t comm_turn){
    return op_api::npu_mm_all_reduce_base(x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3, dequant_scale, antiquant_group_size, comm_turn);
}
at::Tensor npu_mm_reduce_scatter_base(const at::Tensor & self, const at::Tensor & x2, c10::string_view hcom, int64_t world_size, c10::string_view reduce_op, const c10::optional<at::Tensor> & bias, int64_t comm_turn){
    return op_api::npu_mm_reduce_scatter_base(self, x2, hcom, world_size, reduce_op, bias, comm_turn);
}
at::Tensor npu_moe_compute_expert_tokens(const at::Tensor & sorted_expert_for_source_row, int64_t num_expert){
    return op_api::npu_moe_compute_expert_tokens(sorted_expert_for_source_row, num_expert);
}
at::Tensor npu_moe_finalize_routing(const at::Tensor & expanded_permuted_rows, const at::Tensor & skip1, const c10::optional<at::Tensor> & skip2, const at::Tensor & bias, const at::Tensor & scales, const at::Tensor & expanded_src_to_dst_row, const at::Tensor & export_for_source_row){
    return op_api::npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, export_for_source_row);
}
at::Tensor npu_normalize_batch(const at::Tensor & self, const at::Tensor & seq_len, int64_t normalize_type){
    return acl_op::npu_normalize_batch(self, seq_len, normalize_type);
}
at::Tensor npu_one_hot(const at::Tensor & self, int64_t num_classes, int64_t depth, const at::Scalar & on_value, const at::Scalar & off_value){
    return acl_op::npu_one_hot(self, num_classes, depth, on_value, off_value);
}
at::Tensor npu_pad(const at::Tensor & input, at::IntArrayRef paddings){
    return acl_op::npu_pad(input, paddings);
}
at::Tensor npu_prompt_flash_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_lengths, int64_t num_heads, double scale_value, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads){
    return op_api::npu_prompt_flash_attention(query, key, value, padding_mask, atten_mask, actual_seq_lengths, num_heads, scale_value, pre_tokens, next_tokens, input_layout, num_key_value_heads);
}
at::Tensor npu_ps_roi_pooling(const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t group_size, int64_t output_dim){
    return acl_op::npu_ps_roi_pooling(self, rois, spatial_scale, group_size, output_dim);
}
at::Tensor npu_ps_roi_pooling_backward_symint(const at::Tensor & output_grad, const at::Tensor & rois, double spatial_scale, int64_t group_size, int64_t output_dim, c10::SymIntArrayRef input_size){
    return acl_op::npu_ps_roi_pooling_backward_symint(output_grad, rois, spatial_scale, group_size, output_dim, input_size);
}
at::Tensor npu_ptiou(const at::Tensor & bboxes, const at::Tensor & gtboxes, int64_t mode){
    return acl_op::npu_ptiou(bboxes, gtboxes, mode);
}
at::Tensor npu_quant_conv2d(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & scale, at::IntArrayRef strides, at::IntArrayRef pads, at::IntArrayRef dilations, int64_t groups, int64_t offset_x, c10::string_view round_mode, c10::optional<at::ScalarType> output_dtype, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & offset){
    return acl_op::npu_quant_conv2d(input, weight, scale, strides, pads, dilations, groups, offset_x, round_mode, output_dtype, bias, offset);
}
at::Tensor npu_quant_matmul(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & scale, const c10::optional<at::Tensor> & offset, const c10::optional<at::Tensor> & pertoken_scale, const c10::optional<at::Tensor> & bias, c10::optional<at::ScalarType> output_dtype){
    return op_api::npu_quant_matmul(x1, x2, scale, offset, pertoken_scale, bias, output_dtype);
}
at::Tensor npu_quant_scatter(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, const at::Tensor & quant_scales, const c10::optional<at::Tensor> & quant_zero_points, int64_t axis, int64_t quant_axis, c10::string_view reduce){
    return op_api::npu_quant_scatter(self, indices, updates, quant_scales, quant_zero_points, axis, quant_axis, reduce);
}
at::Tensor npu_quantize(const at::Tensor & self, const at::Tensor & scales, const c10::optional<at::Tensor> & zero_points, at::ScalarType dtype, int64_t axis, bool div_mode){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(scales) && at_npu::native::FormatHelper::IsOpInputBaseFormat(zero_points)) {
        return op_api::npu_quantize(self, scales, zero_points, dtype, axis, div_mode);
    } else {
        return acl_op::npu_quantize(self, scales, zero_points, dtype, axis, div_mode);
    }
}
at::Tensor npu_reshape(const at::Tensor & self, at::IntArrayRef shape, bool can_refresh){
    return acl_op::npu_reshape(self, shape, can_refresh);
}
at::Tensor npu_roi_align(const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sample_num, int64_t roi_end_mode){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(rois)) {
        return op_api::npu_roi_align(self, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode);
    } else {
        return acl_op::npu_roi_align(self, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode);
    }
}
at::Tensor npu_roi_alignbk(const at::Tensor & self, const at::Tensor & rois, at::IntArrayRef xdiff_shape, int64_t pooled_width, int64_t pooled_height, double spatial_scale, int64_t sample_num, c10::optional<int64_t> roi_end_mode){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(rois)) {
        return op_api::npu_roi_alignbk(self, rois, xdiff_shape, pooled_width, pooled_height, spatial_scale, sample_num, roi_end_mode);
    } else {
        return acl_op::npu_roi_alignbk(self, rois, xdiff_shape, pooled_width, pooled_height, spatial_scale, sample_num, roi_end_mode);
    }
}
at::Tensor npu_rotary_mul(const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2){
    return acl_op::npu_rotary_mul(self, r1, r2);
}
at::Tensor npu_rotated_box_decode(const at::Tensor & self, const at::Tensor & deltas, const at::Tensor & weight){
    return acl_op::npu_rotated_box_decode(self, deltas, weight);
}
at::Tensor npu_rotated_box_encode(const at::Tensor & self, const at::Tensor & gt_bboxes, const at::Tensor & weight){
    return acl_op::npu_rotated_box_encode(self, gt_bboxes, weight);
}
at::Tensor npu_rotated_iou(const at::Tensor & self, const at::Tensor & query_boxes, bool trans, int64_t mode, bool is_cross, double v_threshold, double e_threshold){
    return acl_op::npu_rotated_iou(self, query_boxes, trans, mode, is_cross, v_threshold, e_threshold);
}
at::Tensor npu_rotated_overlaps(const at::Tensor & self, const at::Tensor & query_boxes, bool trans){
    return acl_op::npu_rotated_overlaps(self, query_boxes, trans);
}
at::Tensor npu_scaled_masked_softmax(const at::Tensor & x, const at::Tensor & mask, const at::Scalar & scale, bool fixed_triu_mask){
    return acl_op::npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask);
}
at::Tensor npu_scaled_masked_softmax_backward(const at::Tensor & y_grad, const at::Tensor & y, const at::Tensor & mask, const at::Scalar & scale, bool fixed_triu_mask){
    return acl_op::npu_scaled_masked_softmax_backward(y_grad, y, mask, scale, fixed_triu_mask);
}
at::Tensor npu_scatter(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t dim){
    return acl_op::npu_scatter(self, indices, updates, dim);
}
at::Tensor npu_scatter_nd_update(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates){
    return op_api::npu_scatter_nd_update(self, indices, updates);
}
at::Tensor npu_sign_bits_pack(const at::Tensor & self, int64_t size){
    return acl_op::npu_sign_bits_pack(self, size);
}
at::Tensor npu_sign_bits_unpack(const at::Tensor & input, int64_t size, at::ScalarType dtype){
    return acl_op::npu_sign_bits_unpack(input, size, dtype);
}
at::Tensor npu_silu(const at::Tensor & self){
    return acl_op::npu_silu(self);
}
at::Tensor npu_silu_backward(const at::Tensor & grad_output, const at::Tensor & x0, const at::Tensor & x1){
    return acl_op::npu_silu_backward(grad_output, x0, x1);
}
at::Tensor npu_slice(const at::Tensor & self, at::IntArrayRef offsets, at::IntArrayRef size){
    return acl_op::npu_slice(self, offsets, size);
}
at::Tensor npu_softmax_cross_entropy_with_logits(const at::Tensor & self, const at::Tensor & labels){
    return acl_op::npu_softmax_cross_entropy_with_logits(self, labels);
}
at::Tensor npu_softmax_cross_entropy_with_logits_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & labels){
    return acl_op::npu_softmax_cross_entropy_with_logits_backward(grad, self, labels);
}
at::Tensor npu_sort_v2(const at::Tensor & self, int64_t dim, bool descending){
    return acl_op::npu_sort_v2(self, dim, descending);
}
at::Tensor npu_stride_add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & offset1, const at::Scalar & offset2, const at::Scalar & c1_len){
    return acl_op::npu_stride_add(self, other, offset1, offset2, c1_len);
}
at::Tensor npu_stride_copy(const at::Tensor & self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar & storage_offset){
    return acl_op::npu_stride_copy(self, shape, stride, storage_offset);
}
at::Tensor npu_sub_sample(const at::Tensor & self, int64_t per_images, double positive_fraction){
    return acl_op::npu_sub_sample(self, per_images, positive_fraction);
}
at::Tensor npu_swiglu(const at::Tensor & self, int64_t dim){
    return op_api::npu_swiglu(self, dim);
}
at::Tensor npu_swiglu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim){
    return op_api::npu_swiglu_backward(grad_output, self, dim);
}
at::Tensor npu_trans_quant_param(const at::Tensor & scale, const c10::optional<at::Tensor> & offset){
    return op_api::npu_trans_quant_param(scale, offset);
}
at::Tensor npu_transpose(const at::Tensor & self, at::IntArrayRef perm, bool require_contiguous){
    return acl_op::npu_transpose(self, perm, require_contiguous);
}
at::Tensor npu_weight_quant_batchmatmul(const at::Tensor & x, const at::Tensor & weight, const at::Tensor & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & quant_scale, const c10::optional<at::Tensor> & quant_offset, const c10::optional<at::Tensor> & bias, int64_t antiquant_group_size){
    return op_api::npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size);
}
at::Tensor npu_yolo_boxes_encode(const at::Tensor & self, const at::Tensor & gt_bboxes, const at::Tensor & stride, bool performance_mode){
    return acl_op::npu_yolo_boxes_encode(self, gt_bboxes, stride, performance_mode);
}
at::Tensor one_hot(const at::Tensor & self, int64_t num_classes){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::one_hot(self, num_classes);
    } else {
        return acl_op::one_hot(self, num_classes);
    }
}
at::Tensor ones(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::ones(size, names, dtype, layout, device, pin_memory);
    } else {
        return acl_op::ones(size, names, dtype, layout, device, pin_memory);
    }
}
at::Tensor ones(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::ones(size, dtype, layout, device, pin_memory);
    } else {
        return acl_op::ones(size, dtype, layout, device, pin_memory);
    }
}
at::Tensor ones_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::ones_like(self, dtype, layout, device, pin_memory, memory_format);
    } else {
        return acl_op::ones_like(self, dtype, layout, device, pin_memory, memory_format);
    }
}
at::Tensor pdist(const at::Tensor & self, double p){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::pdist(self, p);
    } else {
        return acl_op::pdist(self, p);
    }
}
at::Tensor pow(const at::Scalar & self, const at::Tensor & exponent){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent)) {
        return op_api::pow(self, exponent);
    } else {
        return acl_op::pow(self, exponent);
    }
}
at::Tensor pow(const at::Tensor & self, const at::Scalar & exponent){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::pow(self, exponent);
    } else {
        return acl_op::pow(self, exponent);
    }
}
at::Tensor pow(const at::Tensor & self, const at::Tensor & exponent){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent)) {
        return op_api::pow(self, exponent);
    } else {
        return acl_op::pow(self, exponent);
    }
}
at::Tensor prod(const at::Tensor & self, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::prod(self, dtype);
    } else {
        return acl_op::prod(self, dtype);
    }
}
at::Tensor prod(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::prod(self, dim, keepdim, dtype);
    } else {
        return acl_op::prod(self, dim, keepdim, dtype);
    }
}
at::Tensor quantize_per_channel(const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype){
    return acl_op::quantize_per_channel(self, scales, zero_points, axis, dtype);
}
at::Tensor quantize_per_tensor(const at::Tensor & self, double scale, int64_t zero_point, at::ScalarType dtype){
    return acl_op::quantize_per_tensor(self, scale, zero_point, dtype);
}
at::Tensor randperm(int64_t n, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::randperm(n, generator, dtype, layout, device, pin_memory);
    } else {
        return acl_op::randperm(n, generator, dtype, layout, device, pin_memory);
    }
}
at::Tensor randperm(int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::randperm(n, dtype, layout, device, pin_memory);
    } else {
        return acl_op::randperm(n, dtype, layout, device, pin_memory);
    }
}
at::Tensor range(const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::range(start, end, dtype, layout, device, pin_memory);
    } else {
        return acl_op::range(start, end, dtype, layout, device, pin_memory);
    }
}
at::Tensor range(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::range(start, end, step, dtype, layout, device, pin_memory);
    } else {
        return acl_op::range(start, end, step, dtype, layout, device, pin_memory);
    }
}
at::Tensor reciprocal(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::reciprocal(self);
    } else {
        return acl_op::reciprocal(self);
    }
}
at::Tensor reflection_pad1d(const at::Tensor & self, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::reflection_pad1d(self, padding);
    } else {
        return acl_op::reflection_pad1d(self, padding);
    }
}
at::Tensor reflection_pad1d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::reflection_pad1d_backward(grad_output, self, padding);
    } else {
        return acl_op::reflection_pad1d_backward(grad_output, self, padding);
    }
}
at::Tensor reflection_pad2d(const at::Tensor & self, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::reflection_pad2d(self, padding);
    } else {
        return acl_op::reflection_pad2d(self, padding);
    }
}
at::Tensor reflection_pad2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::reflection_pad2d_backward(grad_output, self, padding);
    } else {
        return acl_op::reflection_pad2d_backward(grad_output, self, padding);
    }
}
at::Tensor reflection_pad3d(const at::Tensor & self, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::reflection_pad3d(self, padding);
    } else {
        return acl_op::reflection_pad3d(self, padding);
    }
}
at::Tensor reflection_pad3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    return op_api::reflection_pad3d_backward(grad_output, self, padding);
}
at::Tensor relu(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::relu(self);
    } else {
        return acl_op::relu(self);
    }
}
at::Tensor remainder(const at::Scalar & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::remainder(self, other);
    } else {
        return acl_op::remainder(self, other);
    }
}
at::Tensor remainder(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::remainder(self, other);
    } else {
        return acl_op::remainder(self, other);
    }
}
at::Tensor remainder(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::remainder(self, other);
    } else {
        return acl_op::remainder(self, other);
    }
}
at::Tensor renorm(const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::renorm(self, p, dim, maxnorm);
    } else {
        return acl_op::renorm(self, p, dim, maxnorm);
    }
}
at::Tensor repeat(const at::Tensor & self, at::IntArrayRef repeats){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::repeat(self, repeats);
    } else {
        return acl_op::repeat(self, repeats);
    }
}
at::Tensor repeat_interleave_backward_int_symint(const at::Tensor & grad, const at::Tensor & self, c10::SymInt repeats, c10::optional<int64_t> dim){
    return op_api::repeat_interleave_backward_int_symint(grad, self, repeats, dim);
}
at::Tensor repeat_interleave_backward_tensor(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & repeats, c10::optional<int64_t> dim){
    return op_api::repeat_interleave_backward_tensor(grad, self, repeats, dim);
}
at::Tensor repeat_interleave_symint(const at::Tensor & self, c10::SymInt repeats, c10::optional<int64_t> dim, c10::optional<c10::SymInt> output_size){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::repeat_interleave_symint(self, repeats, dim, output_size);
    } else {
        return acl_op::repeat_interleave_symint(self, repeats, dim, output_size);
    }
}
at::Tensor repeat_interleave_symint(const at::Tensor & self, const at::Tensor & repeats, c10::optional<int64_t> dim, c10::optional<c10::SymInt> output_size){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(repeats)) {
        return op_api::repeat_interleave_symint(self, repeats, dim, output_size);
    } else {
        return acl_op::repeat_interleave_symint(self, repeats, dim, output_size);
    }
}
at::Tensor replication_pad1d(const at::Tensor & self, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::replication_pad1d(self, padding);
    } else {
        return acl_op::replication_pad1d(self, padding);
    }
}
at::Tensor replication_pad1d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::replication_pad1d_backward(grad_output, self, padding);
    } else {
        return acl_op::replication_pad1d_backward(grad_output, self, padding);
    }
}
at::Tensor replication_pad2d(const at::Tensor & self, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::replication_pad2d(self, padding);
    } else {
        return acl_op::replication_pad2d(self, padding);
    }
}
at::Tensor replication_pad2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::replication_pad2d_backward(grad_output, self, padding);
    } else {
        return acl_op::replication_pad2d_backward(grad_output, self, padding);
    }
}
at::Tensor replication_pad3d(const at::Tensor & self, at::IntArrayRef padding){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::replication_pad3d(self, padding);
    } else {
        return acl_op::replication_pad3d(self, padding);
    }
}
at::Tensor replication_pad3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    return op_api::replication_pad3d_backward(grad_output, self, padding);
}
at::Tensor roll(const at::Tensor & self, at::IntArrayRef shifts, at::IntArrayRef dims){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::roll(self, shifts, dims);
    } else {
        return acl_op::roll(self, shifts, dims);
    }
}
at::Tensor round(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::round(self);
    } else {
        return acl_op::round(self);
    }
}
at::Tensor round(const at::Tensor & self, int64_t decimals){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::round(self, decimals);
    } else {
        return acl_op::round(self, decimals);
    }
}
at::Tensor rrelu_with_noise(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(noise)) {
        return op_api::rrelu_with_noise(self, noise, lower, upper, training, generator);
    } else {
        return acl_op::rrelu_with_noise(self, noise, lower, upper, training, generator);
    }
}
at::Tensor rsqrt(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::rsqrt(self);
    } else {
        return acl_op::rsqrt(self);
    }
}
at::Tensor rsub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::rsub(self, other, alpha);
    } else {
        return acl_op::rsub(self, other, alpha);
    }
}
at::Tensor rsub(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::rsub(self, other, alpha);
    } else {
        return acl_op::rsub(self, other, alpha);
    }
}
at::Tensor scaled_dot_product_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, c10::optional<double> scale){
    return op_api::scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale);
}
at::Tensor scatter_add(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(src)) {
        return op_api::scatter_add(self, dim, index, src);
    } else {
        return acl_op::scatter_add(self, dim, index, src);
    }
}
at::Tensor scatter_add(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index) && at_npu::native::FormatHelper::IsOpInputBaseFormat(src)) {
        return op_api::scatter_add(self, dim, index, src);
    } else {
        return acl_op::scatter_add(self, dim, index, src);
    }
}
at::Tensor scatter_update(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t axis){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices) && at_npu::native::FormatHelper::IsOpInputBaseFormat(updates)) {
        return op_api::scatter_update(self, indices, updates, axis);
    } else {
        return acl_op::scatter_update(self, indices, updates, axis);
    }
}
at::Tensor searchsorted(const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<at::Tensor> & sorter){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(sorted_sequence) && at_npu::native::FormatHelper::IsOpInputBaseFormat(sorter)) {
        return op_api::searchsorted(sorted_sequence, self, out_int32, right, side, sorter);
    } else {
        return acl_op::searchsorted(sorted_sequence, self, out_int32, right, side, sorter);
    }
}
at::Tensor searchsorted(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<at::Tensor> & sorter){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(sorted_sequence) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(sorter)) {
        return op_api::searchsorted(sorted_sequence, self, out_int32, right, side, sorter);
    } else {
        return acl_op::searchsorted(sorted_sequence, self, out_int32, right, side, sorter);
    }
}
at::Tensor selu(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::selu(self);
    } else {
        return acl_op::selu(self);
    }
}
at::Tensor selu_backward(const at::Tensor & grad_output, const at::Tensor & result){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(result)) {
        return op_api::selu_backward(grad_output, result);
    } else {
        return acl_op::selu_backward(grad_output, result);
    }
}
at::Tensor sgn(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sgn(self);
    } else {
        return acl_op::sgn(self);
    }
}
at::Tensor sigmoid(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sigmoid(self);
    } else {
        return acl_op::sigmoid(self);
    }
}
at::Tensor sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & output){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output)) {
        return op_api::sigmoid_backward(grad_output, output);
    } else {
        return acl_op::sigmoid_backward(grad_output, output);
    }
}
at::Tensor sign(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sign(self);
    } else {
        return acl_op::sign(self);
    }
}
at::Tensor silu(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::silu(self);
    } else {
        return acl_op::silu(self);
    }
}
at::Tensor silu_backward(const at::Tensor & grad_output, const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::silu_backward(grad_output, self);
    } else {
        return acl_op::silu_backward(grad_output, self);
    }
}
at::Tensor sin(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sin(self);
    } else {
        return acl_op::sin(self);
    }
}
at::Tensor sinc(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sinc(self);
    } else {
        return acl_op::sinc(self);
    }
}
at::Tensor sinh(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sinh(self);
    } else {
        return acl_op::sinh(self);
    }
}
at::Tensor slow_conv3d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding){
    return acl_op::slow_conv3d(self, weight, kernel_size, bias, stride, padding);
}
at::Tensor slow_conv3d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding){
    return acl_op::slow_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
}
at::Tensor slow_conv_dilated2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
    } else {
        return acl_op::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
    }
}
at::Tensor slow_conv_transpose2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias)) {
        return op_api::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    } else {
        return acl_op::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    }
}
at::Tensor smooth_l1_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::smooth_l1_loss(self, target, reduction, beta);
    } else {
        return acl_op::smooth_l1_loss(self, target, reduction, beta);
    }
}
at::Tensor smooth_l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::smooth_l1_loss_backward(grad_output, self, target, reduction, beta);
    } else {
        return acl_op::smooth_l1_loss_backward(grad_output, self, target, reduction, beta);
    }
}
at::Tensor soft_margin_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::soft_margin_loss(self, target, reduction);
    } else {
        return acl_op::soft_margin_loss(self, target, reduction);
    }
}
at::Tensor soft_margin_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(target)) {
        return op_api::soft_margin_loss_backward(grad_output, self, target, reduction);
    } else {
        return acl_op::soft_margin_loss_backward(grad_output, self, target, reduction);
    }
}
at::Tensor softmax(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::softmax(self, dim, dtype);
    } else {
        return acl_op::softmax(self, dim, dtype);
    }
}
at::Tensor softmax(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::softmax(self, dim, dtype);
    } else {
        return acl_op::softmax(self, dim, dtype);
    }
}
at::Tensor softplus(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::softplus(self, beta, threshold);
    } else {
        return acl_op::softplus(self, beta, threshold);
    }
}
at::Tensor softshrink(const at::Tensor & self, const at::Scalar & lambd){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::softshrink(self, lambd);
    } else {
        return acl_op::softshrink(self, lambd);
    }
}
at::Tensor softshrink_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::softshrink_backward(grad_output, self, lambd);
    } else {
        return acl_op::softshrink_backward(grad_output, self, lambd);
    }
}
at::Tensor sqrt(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sqrt(self);
    } else {
        return acl_op::sqrt(self);
    }
}
at::Tensor stack(at::TensorList tensors, int64_t dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors)) {
        return op_api::stack(tensors, dim);
    } else {
        return acl_op::stack(tensors, dim);
    }
}
at::Tensor std(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::std(self, dim, correction, keepdim);
    } else {
        return acl_op::std(self, dim, correction, keepdim);
    }
}
at::Tensor stft(const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool normalized, c10::optional<bool> onesided, c10::optional<bool> return_complex){
    return op_api::stft(self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
}
at::Tensor sub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sub(self, other, alpha);
    } else {
        return acl_op::sub(self, other, alpha);
    }
}
at::Tensor sub(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::sub(self, other, alpha);
    } else {
        return acl_op::sub(self, other, alpha);
    }
}
at::Tensor sum(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sum(self, dim, keepdim, dtype);
    } else {
        return acl_op::sum(self, dim, keepdim, dtype);
    }
}
at::Tensor sum(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sum(self, dim, keepdim, dtype);
    } else {
        return acl_op::sum(self, dim, keepdim, dtype);
    }
}
at::Tensor sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::sum(self, dtype);
    } else {
        return acl_op::sum(self, dtype);
    }
}
at::Tensor take(const at::Tensor & self, const at::Tensor & index){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(index)) {
        return op_api::take(self, index);
    } else {
        return acl_op::take(self, index);
    }
}
at::Tensor tan(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::tan(self);
    } else {
        return acl_op::tan(self);
    }
}
at::Tensor tanh(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::tanh(self);
    } else {
        return acl_op::tanh(self);
    }
}
at::Tensor tanh_backward(const at::Tensor & grad_output, const at::Tensor & output){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(output)) {
        return op_api::tanh_backward(grad_output, output);
    } else {
        return acl_op::tanh_backward(grad_output, output);
    }
}
at::Tensor threshold(const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::threshold(self, threshold, value);
    } else {
        return acl_op::threshold(self, threshold, value);
    }
}
at::Tensor threshold_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::threshold_backward(grad_output, self, threshold);
    } else {
        return acl_op::threshold_backward(grad_output, self, threshold);
    }
}
at::Tensor trace(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::trace(self);
    } else {
        return acl_op::trace(self);
    }
}
at::Tensor tril(const at::Tensor & self, int64_t diagonal){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::tril(self, diagonal);
    } else {
        return acl_op::tril(self, diagonal);
    }
}
at::Tensor triu(const at::Tensor & self, int64_t diagonal){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::triu(self, diagonal);
    } else {
        return acl_op::triu(self, diagonal);
    }
}
at::Tensor trunc(const at::Tensor & self){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::trunc(self);
    } else {
        return acl_op::trunc(self);
    }
}
at::Tensor upsample_bicubic2d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::upsample_bicubic2d(self, output_size, align_corners, scales_h, scales_w);
    } else {
        return acl_op::upsample_bicubic2d(self, output_size, align_corners, scales_h, scales_w);
    }
}
at::Tensor upsample_bicubic2d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output)) {
        return op_api::upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    } else {
        return acl_op::upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    }
}
at::Tensor upsample_bilinear2d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::upsample_bilinear2d(self, output_size, align_corners, scales_h, scales_w);
    } else {
        return acl_op::upsample_bilinear2d(self, output_size, align_corners, scales_h, scales_w);
    }
}
at::Tensor upsample_bilinear2d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output)) {
        return op_api::upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    } else {
        return acl_op::upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    }
}
at::Tensor upsample_linear1d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::upsample_linear1d(self, output_size, align_corners, scales);
    } else {
        return acl_op::upsample_linear1d(self, output_size, align_corners, scales);
    }
}
at::Tensor upsample_linear1d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output)) {
        return op_api::upsample_linear1d_backward(grad_output, output_size, input_size, align_corners, scales);
    } else {
        return acl_op::upsample_linear1d_backward(grad_output, output_size, input_size, align_corners, scales);
    }
}
at::Tensor upsample_nearest1d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::upsample_nearest1d(self, output_size, scales);
    } else {
        return acl_op::upsample_nearest1d(self, output_size, scales);
    }
}
at::Tensor upsample_nearest1d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output)) {
        return op_api::upsample_nearest1d_backward(grad_output, output_size, input_size, scales);
    } else {
        return acl_op::upsample_nearest1d_backward(grad_output, output_size, input_size, scales);
    }
}
at::Tensor upsample_nearest2d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::upsample_nearest2d(self, output_size, scales_h, scales_w);
    } else {
        return acl_op::upsample_nearest2d(self, output_size, scales_h, scales_w);
    }
}
at::Tensor upsample_nearest2d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output)) {
        return op_api::upsample_nearest2d_backward(grad_output, output_size, input_size, scales_h, scales_w);
    } else {
        return acl_op::upsample_nearest2d_backward(grad_output, output_size, input_size, scales_h, scales_w);
    }
}
at::Tensor upsample_nearest3d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::upsample_nearest3d(self, output_size, scales_d, scales_h, scales_w);
    } else {
        return acl_op::upsample_nearest3d(self, output_size, scales_d, scales_h, scales_w);
    }
}
at::Tensor upsample_nearest3d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output)) {
        return op_api::upsample_nearest3d_backward(grad_output, output_size, input_size, scales_d, scales_h, scales_w);
    } else {
        return acl_op::upsample_nearest3d_backward(grad_output, output_size, input_size, scales_d, scales_h, scales_w);
    }
}
at::Tensor upsample_trilinear3d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::upsample_trilinear3d(self, output_size, align_corners, scales_d, scales_h, scales_w);
    } else {
        return acl_op::upsample_trilinear3d(self, output_size, align_corners, scales_d, scales_h, scales_w);
    }
}
at::Tensor upsample_trilinear3d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output)) {
        return op_api::upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
    } else {
        return acl_op::upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
    }
}
at::Tensor var(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::var(self, dim, correction, keepdim);
    } else {
        return acl_op::var(self, dim, correction, keepdim);
    }
}
at::Tensor vdot(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::vdot(self, other);
    } else {
        return acl_op::vdot(self, other);
    }
}
at::Tensor view_as_complex(const at::Tensor & self){
    return acl_op::view_as_complex(self);
}
at::Tensor view_as_real(const at::Tensor & self){
    return acl_op::view_as_real(self);
}
at::Tensor where(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(condition) && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::where(condition, self, other);
    } else {
        return acl_op::where(condition, self, other);
    }
}
at::Tensor xlogy(const at::Scalar & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::xlogy(self, other);
    } else {
        return acl_op::xlogy(self, other);
    }
}
at::Tensor xlogy(const at::Tensor & self, const at::Scalar & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::xlogy(self, other);
    } else {
        return acl_op::xlogy(self, other);
    }
}
at::Tensor xlogy(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::xlogy(self, other);
    } else {
        return acl_op::xlogy(self, other);
    }
}
at::Tensor zeros(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::zeros(size, names, dtype, layout, device, pin_memory);
    } else {
        return acl_op::zeros(size, names, dtype, layout, device, pin_memory);
    }
}
at::Tensor zeros_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self)) {
        return op_api::zeros_like(self, dtype, layout, device, pin_memory, memory_format);
    } else {
        return acl_op::zeros_like(self, dtype, layout, device, pin_memory, memory_format);
    }
}
at::Tensor zeros_symint(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    if (at_npu::native::env::CheckJitDisable()) {
        return op_api::zeros_symint(size, dtype, layout, device, pin_memory);
    } else {
        return acl_op::zeros_symint(size, dtype, layout, device, pin_memory);
    }
}
bool _amp_foreach_non_finite_check(at::TensorList scaled_grads){
    return acl_op::_amp_foreach_non_finite_check(scaled_grads);
}
bool equal(const at::Tensor & self, const at::Tensor & other){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(other)) {
        return op_api::equal(self, other);
    } else {
        return acl_op::equal(self, other);
    }
}
const at::Tensor & _conv_depthwise2d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, const at::Tensor & out){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight) && at_npu::native::FormatHelper::IsOpInputBaseFormat(bias) && at_npu::native::FormatHelper::IsOpInputBaseFormat(out)) {
        return op_api::_conv_depthwise2d_out(self, weight, kernel_size, bias, stride, padding, dilation, out);
    } else {
        return acl_op::_conv_depthwise2d_out(self, weight, kernel_size, bias, stride, padding, dilation, out);
    }
}
void _amp_foreach_non_finite_check_and_unscale_(at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(found_inf) && at_npu::native::FormatHelper::IsOpInputBaseFormat(inv_scale)) {
        return op_api::_amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale);
    } else {
        return acl_op::_amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale);
    }
}
void _cummax_helper(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::_cummax_helper(self, values, indices, dim);
    } else {
        return acl_op::_cummax_helper(self, values, indices, dim);
    }
}
void _cummin_helper(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim){
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(self) && at_npu::native::FormatHelper::IsOpInputBaseFormat(values) && at_npu::native::FormatHelper::IsOpInputBaseFormat(indices)) {
        return op_api::_cummin_helper(self, values, indices, dim);
    } else {
        return acl_op::_cummin_helper(self, values, indices, dim);
    }
}
void _foreach_add_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_add_(self, scalars);
}
void _foreach_add_(at::TensorList self, at::TensorList other, const at::Scalar & alpha){
    return op_api::_foreach_add_(self, other, alpha);
}
void _foreach_add_(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_add_(self, scalar);
}
void _foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_addcdiv_(self, tensor1, tensor2, scalars);
}
void _foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value){
    return op_api::_foreach_addcdiv_(self, tensor1, tensor2, value);
}
void _foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars){
    return op_api::_foreach_addcdiv_(self, tensor1, tensor2, scalars);
}
void _foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_addcmul_(self, tensor1, tensor2, scalars);
}
void _foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value){
    return op_api::_foreach_addcmul_(self, tensor1, tensor2, value);
}
void _foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars){
    return op_api::_foreach_addcmul_(self, tensor1, tensor2, scalars);
}
void _foreach_ceil_(at::TensorList self){
    return op_api::_foreach_ceil_(self);
}
void _foreach_cos_(at::TensorList self){
    return op_api::_foreach_cos_(self);
}
void _foreach_div_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_div_(self, scalars);
}
void _foreach_div_(at::TensorList self, at::TensorList other){
    return op_api::_foreach_div_(self, other);
}
void _foreach_div_(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_div_(self, scalar);
}
void _foreach_exp_(at::TensorList self){
    return op_api::_foreach_exp_(self);
}
void _foreach_floor_(at::TensorList self){
    return op_api::_foreach_floor_(self);
}
void _foreach_frac_(at::TensorList self){
    return op_api::_foreach_frac_(self);
}
void _foreach_maximum_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_maximum_(self, scalars);
}
void _foreach_maximum_(at::TensorList self, at::TensorList other){
    return op_api::_foreach_maximum_(self, other);
}
void _foreach_maximum_(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_maximum_(self, scalar);
}
void _foreach_minimum_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_minimum_(self, scalars);
}
void _foreach_minimum_(at::TensorList self, at::TensorList other){
    return op_api::_foreach_minimum_(self, other);
}
void _foreach_minimum_(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_minimum_(self, scalar);
}
void _foreach_mul_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_mul_(self, scalars);
}
void _foreach_mul_(at::TensorList self, at::TensorList other){
    return op_api::_foreach_mul_(self, other);
}
void _foreach_mul_(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_mul_(self, scalar);
}
void _foreach_neg_(at::TensorList self){
    return op_api::_foreach_neg_(self);
}
void _foreach_pow_(at::TensorList self, at::ArrayRef<at::Scalar> exponent){
    return op_api::_foreach_pow_(self, exponent);
}
void _foreach_pow_(at::TensorList self, at::TensorList exponent){
    return op_api::_foreach_pow_(self, exponent);
}
void _foreach_pow_(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_pow_(self, scalar);
}
void _foreach_round_(at::TensorList self){
    return op_api::_foreach_round_(self);
}
void _foreach_sigmoid_(at::TensorList self){
    return op_api::_foreach_sigmoid_(self);
}
void _foreach_sqrt_(at::TensorList self){
    return op_api::_foreach_sqrt_(self);
}
void _foreach_sub_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    return op_api::_foreach_sub_(self, scalars);
}
void _foreach_sub_(at::TensorList self, at::TensorList other, const at::Scalar & alpha){
    return op_api::_foreach_sub_(self, other, alpha);
}
void _foreach_sub_(at::TensorList self, const at::Scalar & scalar){
    return op_api::_foreach_sub_(self, scalar);
}
void _foreach_trunc_(at::TensorList self){
    return op_api::_foreach_trunc_(self);
}
void npu_scatter_list_(at::TensorList self, const at::Tensor & indices, const at::Tensor & updates, const c10::optional<at::Tensor> & mask, c10::string_view reduce, int64_t axis){
    return op_api::npu_scatter_list_(self, indices, updates, mask, reduce, axis);
}
}  // namespace acl_op
