// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

#include "aten/common/InnerNpuNativeFunction.h"
#include "aten/utils/op_api_common.h"
#include "core/NPUPeerToPeerAccess.h"
#include "csrc/aten/generated/NPUNativeFunctions.h"
#include "csrc/aten/generated/NPUOpApiNativeFunctions.h"
#include "csrc/backend/NPUCachingHostAllocator.h"
#include "framework/contiguous/ContiguousOpt.h"
#include "framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

// the format of dst and src is base format now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_between_host_and_device_opapi(
    at::Tensor& dst,
    const at::Tensor& src,
    aclrtMemcpyKind kind,
    bool non_blocking) {
  int64_t nbytes = dst.numel() * dst.element_size();
  c10::backend::Stream stream = c10::backend::getCurrentNPUStream();

  if (non_blocking) {
    auto ret = CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
        dst, nbytes, src, nbytes, kind);
    NPU_CHECK_ERROR(ret);
    ASCEND_LOGD("non_blocking copy without StreamSynchronize.");
    const auto& host_tensor = torch_backend::utils::is_npu(dst) ? src : dst;
    void* ptr = host_tensor.data_ptr();
    void* ctx = host_tensor.storage().data_ptr().get_context();
    c10::backend::HostAllocator::recordEvent(ptr, ctx, stream);
  } else {
    aclError error = aclrtSynchronizeStream(stream);
    auto ret = CalcuOpUtil::AclrtMemcpyWithModeSwitch(
        std::make_pair(
            dst.storage().unsafeGetStorageImpl(),
            dst.storage_offset() * dst.itemsize()),
        nbytes,
        std::make_pair(
            src.storage().unsafeGetStorageImpl(),
            src.storage_offset() * src.itemsize()),
        nbytes,
        kind);
    NPU_CHECK_ERROR(ret, "aclrtMemcpy");
    if (error != ACL_ERROR_NONE) {
      C10_NPU_SHOW_ERR_MSG();
      if (c10::npu::option::OptionsManager::IsResumeModeEnable()) {
        TORCH_NPU_WARN(
            "ACL stream synchronize failed, error code:",
            error,
            ". But in checkpoint-resume mode will not throw exceptions.");
      } else {
        AT_ERROR("ACL stream synchronize failed, error code:", error);
      }
    }
  }
}

// the format of dst and src is base format now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_h2d_baseformat_dtype_contigous_opapi(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking) {
  c10::DeviceGuard guard(dst.device());
  aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE;
  copy_between_host_and_device_opapi(dst, src, kind, non_blocking);
}

// the format of dst and src is baseformat now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_d2h_baseformat_dtype_contigous_opapi(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking) {
  c10::DeviceGuard guard(src.device());
  aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST;
  copy_between_host_and_device_opapi(dst, src, kind, non_blocking);
}

// the format of dst and src is baseformat now
void copy_h2d_baseformat_opapi(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking,
    bool dst_must_be_contiguous = false) {
  c10::DeviceGuard guard(dst.device());
  bool same_type = (src.dtype() == dst.dtype());
  bool same_size = (src.sizes() == dst.sizes());
  bool dst_is_contiguous = dst_must_be_contiguous ? true : dst.is_contiguous();
  if (same_type && dst_is_contiguous && src.is_contiguous() && same_size) {
    copy_h2d_baseformat_dtype_contigous_opapi(dst, src, non_blocking);
    return;
  }

  at::Tensor dst_contig = dst_is_contiguous ? dst : at::empty_like(dst);
  at::Tensor src_contig;
  if (!same_type) {
    src_contig = src.to(dst.dtype()).expand_as(dst).contiguous();
  } else {
    src_contig = src.expand_as(dst).contiguous();
  }
  // perform a same-dtype copy on contiguous tensors
  TORCH_INTERNAL_ASSERT(
      dst_contig.sizes().equals(src_contig.sizes()), OPS_ERROR(ErrCode::VALUE));
  TORCH_INTERNAL_ASSERT(
      dst_contig.scalar_type() == src_contig.scalar_type(),
      OPS_ERROR(ErrCode::VALUE));
  copy_h2d_baseformat_dtype_contigous_opapi(
      dst_contig, src_contig, non_blocking);
  // if necessary, copy back into dst
  if (!dst_contig.is_same(dst)) {
    TORCH_INTERNAL_ASSERT(
        dst_contig.device() == dst.device(), OPS_ERROR(ErrCode::VALUE));
    copy_d2d_dtype(dst, dst_contig, non_blocking);
  }
}

// the format of dst and src is baseformat now
void copy_d2h_baseformat_opapi(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking) {
  c10::DeviceGuard guard(src.device());
  bool same_type = (src.dtype() == dst.dtype());
  bool same_size = (src.sizes() == dst.sizes());
  bool dst_is_contiguous = dst.is_contiguous();
  if (same_type && dst_is_contiguous && src.is_contiguous() && same_size) {
    copy_d2h_baseformat_dtype_contigous_opapi(dst, src, non_blocking);
    return;
  }
  at::Tensor dst_contig =
      (dst_is_contiguous && same_type) ? dst : at::empty_like(dst, src.dtype());
  at::Tensor src_contig = src.expand_as(dst).contiguous();
  // perform a same-dtype copy on contiguous tensors
  TORCH_INTERNAL_ASSERT(
      dst_contig.sizes().equals(src_contig.sizes()), OPS_ERROR(ErrCode::VALUE));
  TORCH_INTERNAL_ASSERT(
      dst_contig.scalar_type() == src_contig.scalar_type(),
      OPS_ERROR(ErrCode::VALUE));
  copy_d2h_baseformat_dtype_contigous_opapi(
      dst_contig, src_contig, non_blocking);
  // if necessary, copy back into dst
  if (!dst_contig.is_same(dst)) {
    TORCH_INTERNAL_ASSERT(
        dst_contig.device() == dst.device(), OPS_ERROR(ErrCode::VALUE));
    dst.copy_(dst_contig, non_blocking); // h2h, use cpu copy
  }
}

// the format of dst and src is baseformat now, copy d2d
void copy_d2d_baseformat_opapi(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking) {
  c10::DeviceGuard guard(src.device());
  if (dst.device().index() != src.device().index()) {
    bool warning_flag = false;
    NpuP2pCtrl::get_instance().get_p2p_access(
        src.device().index(), dst.device().index(), warning_flag);
    // In the same 'os', tensor can copy even if the enable fails
    if (warning_flag) {
      ASCEND_LOGW(
          "p2p enable from %d to %d is fails",
          src.device().index(),
          dst.device().index());
    }
    guard.reset_device(dst.device());
    c10::backend::Stream dst_stream =
        c10::backend::getCurrentNPUStream(dst.device().index());
    NPU_CHECK_ERROR(aclrtSynchronizeStreamWithTimeout(dst_stream, -1));
    guard.reset_device(src.device());
  } else {
    c10::SmallVector<at::Tensor, N> inputs = {src};
    c10::SmallVector<at::Tensor, N> outputs = {dst};
    CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
  }
  EXEC_NPU_CMD(aclnnInplaceCopy, dst, src);
  if (dst.device().index() != src.device().index()) {
    c10::backend::Stream copy_stream = c10::backend::getCurrentNPUStream();
    NPU_CHECK_ERROR(aclrtSynchronizeStreamWithTimeout(copy_stream, -1));
  }
}

at::Tensor& NPUNativeOpApiFunctions::copy_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  DO_COMPATIBILITY(
      aclnnInplaceCopy, NPUNativeFunctions::copy_(self, src, non_blocking));
  if (self.numel() == 0) {
    return self;
  }
  auto maybe_outnames =
      at::namedinference::compute_broadcast_outnames(self, src);

  if (torch_backend::utils::is_npu(self)) {
    if (torch_backend::utils::is_npu(src)) {
      copy_d2d_baseformat_opapi(self, src, non_blocking);
    } else {
      copy_h2d_baseformat_opapi(self, src, non_blocking);
    }
  } else {
    if (torch_backend::utils::is_npu(src)) {
      copy_d2h_baseformat_opapi(self, src, non_blocking);
    }
  }
  at::namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

} // namespace native
} // namespace at_npu
