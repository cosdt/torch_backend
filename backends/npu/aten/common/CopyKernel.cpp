#include <ATen/ATen.h>

#include "aten/OpInterface.h"
#include "aten/common/FormatCastHelper.h"
#include "aten/common/InnerNpuNativeFunction.h"
#include "core/NPUException.h"
#include "core/NPUPeerToPeerAccess.h"
#include "core/register/OptionsManager.h"
#include "csrc/aten/generated/CustomFunctions.h"
#include "csrc/aten/generated/NPUNativeFunctions.h"
#include "csrc/backend/CachingHostAllocator.h"
#include "framework/FormatHelper.h"
#include "framework/StorageDescHelper.h"
#include "framework/contiguous/ContiguousOpt.h"
#include "framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

namespace {
// NOTE: helper function of copy, the input parameter is not checked, The caller
// needs to ensure that the parameters are correct.

// the caller should ensure the tensor.is_npu == true
bool is_same_format(const at::Tensor& a, const at::Tensor& b) {
  bool isSameFormat = FormatHelper::GetFormat(a) == FormatHelper::GetFormat(b);
  if (!isSameFormat) {
    bool isBaseFormat =
        FormatHelper::IsBaseFormatType(a) && FormatHelper::IsBaseFormatType(b);
    return isBaseFormat;
  }
  return true;
}

bool try_to_optimize_copy_with_any_format(
    at::Tensor& self,
    const at::Tensor& src) {
  // Some Ops support inputs with 5HD/NZ format, Transdata is redundant
  // Record:
  // Op:Reshape; SliceD || Supportformat: 5HD/NZ
  return TransContiguous::ContiguousOptimizeWithAnyFormat(self, src);
}

// the dst and src are same format now
// the dst and src are base format now
// the dst and src maybe non-contiguous
void copy_d2d_last_method(
    at::Tensor& self,
    const at::Tensor& src,
    bool same_type,
    bool non_blocking) {
  // general copy method but Low performance
  RECORD_FUNCTION("contiguous_d_ViewCopy", std::vector<c10::IValue>({src}));
  op_plugin::npu_view_copy(self, src, non_blocking);
}

// the dst and src are same format now
void copy_d2d_dtype_format(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  // Note: Src & Self have the same format.
  if (try_to_optimize_copy_with_any_format(self, src)) {
    return;
  }

  if (!FormatHelper::IsBaseFormatType(self)) { // 必须要非NCHW的才行？
    if (can_use_memcpy(self, src)) {
      RECORD_FUNCTION(
          "d2dCopyAsync with format", std::vector<c10::IValue>({src}));
      return copy_d2d_by_memcpy(self, src);
    }
  }

  if (!FormatHelper::IsBaseFormatType(self)) {
    at::Tensor src_4D = FormatCastHelper::ApplyBaseFormatTensorBy(src);
    at::Tensor dst_4D = FormatCastHelper::ApplyBaseFormatTensorBy(self);
    copy_d2d_dtype_baseformat(dst_4D, src_4D, non_blocking);
    NPUNativeFunctions::npu_format_cast_(self, dst_4D);
    return;
  }
  copy_d2d_dtype_baseformat(self, src, non_blocking);
}

void copy_d2d(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  c10::DeviceGuard guard(src.device());
  // p2p enable and synchronize self stream
  if (self.device().index() != src.device().index()) {
    bool warning_flag = false;
    NpuP2pCtrl::get_instance().get_p2p_access(
        src.device().index(), self.device().index(), warning_flag);
    // In the same 'os', tensor can copy even if the enable fails
    if (warning_flag) {
      ASCEND_LOGW(
          "p2p enable from %d to %d is fails",
          src.device().index(),
          self.device().index());
    }
    guard.reset_device(self.device());
    c10::backend::Stream dst_stream =
        c10::backend::getCurrentStream(self.device().index());
    NPU_CHECK_ERROR(aclrtSynchronizeStreamWithTimeout(dst_stream, -1));
    guard.reset_device(src.device());
  }
  if (self.dtype() != src.dtype()) {
    custom_ops::npu_dtype_cast_(
        self, src); // npu_dtype_cast_ will call copy function.
    return;
  }
  copy_d2d_dtype(self, src, non_blocking);
  // synchronize src stream for different devices copy
  if (self.device().index() != src.device().index()) {
    c10::backend::Stream copy_stream = c10::backend::getCurrentStream();
    NPU_CHECK_ERROR(aclrtSynchronizeStreamWithTimeout(copy_stream, -1));
  }
}

// the format of dst and src is base format now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_between_host_and_device(
    at::Tensor& dst,
    const at::Tensor& src,
    aclrtMemcpyKind kind,
    bool non_blocking) {
  int64_t nbytes = dst.numel() * dst.element_size();
  c10::backend::Stream stream = c10::backend::getCurrentStream();

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
    aclError error = aclrtSynchronizeStreamWithTimeout(stream, -1);
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
void copy_h2d_baseformat_dtype_contigous(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking) {
  c10::DeviceGuard guard(dst.device());
  aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE;
  copy_between_host_and_device(dst, src, kind, non_blocking);
}

// the format of dst and src is baseformat now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_d2h_baseformat_dtype_contigous(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking) {
  c10::DeviceGuard guard(dst.device());
  aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST;
  copy_between_host_and_device(dst, src, kind, non_blocking);
}

// the format of dst and src is baseformat now
void copy_h2d_baseformat(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking,
    bool dst_must_be_contiguous = false) {
  bool same_type = (src.dtype() == dst.dtype());
  bool same_size = (src.sizes() == dst.sizes());
  bool dst_is_contiguous = dst_must_be_contiguous ? true : dst.is_contiguous();
  if (same_type && dst_is_contiguous && src.is_contiguous() && same_size) {
    copy_h2d_baseformat_dtype_contigous(dst, src, non_blocking);
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
  TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
  TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
  copy_h2d_baseformat_dtype_contigous(dst_contig, src_contig, non_blocking);
  // if necessary, copy back into dst
  if (!dst_contig.is_same(dst)) {
    TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
    copy_d2d_dtype(dst, dst_contig, non_blocking);
  }
}

// the format of dst and src is baseformat now
void copy_d2h_baseformat(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking) {
  bool same_type = (src.dtype() == dst.dtype());
  bool same_size = (src.sizes() == dst.sizes());
  bool dst_is_contiguous = dst.is_contiguous();
  if (same_type && dst_is_contiguous && src.is_contiguous() && same_size) {
    copy_d2h_baseformat_dtype_contigous(dst, src, non_blocking);
    return;
  }
  at::Tensor dst_contig =
      (dst_is_contiguous && same_type) ? dst : at::empty_like(dst, src.dtype());
  at::Tensor src_contig = src.expand_as(dst).contiguous();
  // perform a same-dtype copy on contiguous tensors
  TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
  TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
  copy_d2h_baseformat_dtype_contigous(dst_contig, src_contig, non_blocking);
  // if necessary, copy back into dst
  if (!dst_contig.is_same(dst)) {
    TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
    dst.copy_(dst_contig, non_blocking); // h2h, use cpu copy
  }
}

void copy_h2d(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  c10::DeviceGuard guard(self.device());
  if (!FormatHelper::IsBaseFormatType(self)) {
    at::Tensor dst =
        OpPreparation::ApplyTensorWithSizes(self.sizes(), self.options());
    copy_h2d_baseformat(dst, src, non_blocking, true);
    NPUNativeFunctions::npu_format_cast_(self, dst);
    return;
  }
  copy_h2d_baseformat(self, src, non_blocking);
}

void copy_d2h(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  c10::DeviceGuard guard(src.device());
  if (!FormatHelper::IsBaseFormatType(src)) {
    at::Tensor src_4D = FormatCastHelper::ApplyBaseFormatTensorBy(src);
    copy_d2h_baseformat(self, src_4D, non_blocking);
    return;
  }
  copy_d2h_baseformat(self, src, non_blocking);
}
} // namespace

// the caller should guarantee that the format and dtype are same
bool can_use_memcpy(at::Tensor& dst, const at::Tensor& src) {
  if (StorageDescHelper::IsSameDesc(dst, src)) {
    // Make sure that the metadata are same.
    if (!dst.sizes().equals(src.sizes())) {
      return false;
    }
    if (!dst.strides().equals(src.strides())) {
      return false;
    }
    // Make sure that copy the whole memory.
    // we just need to compare one of them, because of the StorageDesc
    // and metadata(sizes and stride) of src and dst are same.
    if (StorageDescHelper::GetValidMemorySize(src) != src.numel()) {
      return false;
    }
    if ((dst.storage_offset() != 0) || (src.storage_offset() != 0)) {
      return false;
    }
    return true;
  }
  return false;
}

at::Tensor copy_d2d_format_cast(at::Tensor& dst, const at::Tensor& src) {
  string srcFormat = FormatHelper::GetFormatName(src);
  string dstFormat = FormatHelper::GetFormatName(dst);

  if (!FormatCastHelper::IsSameGroupType(src, dst)) {
    bool res = FormatCastHelper::format_cast_between_group(
        dst, src, copy_d2d_format_cast);
    if (!res) {
      AT_ERROR("unsupport cast from ", srcFormat, " to ", dstFormat);
    }
    return dst;
  }

  OpCommand cmd;
  cmd.Name("Identity").InputWithoutContiguous(src).Output(dst).Run();
  return dst;
}

// the dst and src are same dtype now
void copy_d2d_dtype(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  if (!is_same_format(self, src)) {
    auto src_desc =
        c10::backend::NPUBridge::GetNpuStorageImpl(src)->storage_desc_;
    if (src.is_contiguous() && FormatHelper::IsBaseFormatType(src) &&
        src_desc.base_sizes_.size() == 1) {
      StorageDescHelper::ReflushDescBySelf(src);
      copy_d2d_format_cast(self, src);
      c10::backend::NPUBridge::GetNpuStorageImpl(src)->storage_desc_ =
          std::move(src_desc);
      return;
    }
    at::Tensor src_4D = FormatCastHelper::ApplyBaseFormatTensorBy(src);
    // ApplyBaseFormatTensorBy is redundant for self tensor with base format.
    if (FormatHelper::IsBaseFormatType(self)) {
      copy_d2d_dtype_baseformat(self, src_4D, non_blocking);
      return;
    }
    at::Tensor dst_4D = FormatCastHelper::ApplyBaseFormatTensorBy(self);
    copy_d2d_dtype_baseformat(dst_4D, src_4D, non_blocking);
    NPUNativeFunctions::npu_format_cast_(self, dst_4D);
    return;
  }
  copy_d2d_dtype_format(self, src, non_blocking);
}

// the dst and src are same format now
// the dst and src are base format now
// the dst and src maybe non-contiguous
void copy_d2d_dtype_baseformat(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  if (!self.is_contiguous()) {
    // Contiguous/discontiguous source tensor copy to discontiguous self tensor
    return copy_d2d_last_method(self, src, true, non_blocking);
  }

  if (!src.is_contiguous()) {
    // Discontiguous source tensor copy to contiguous self tensor
    if (TransContiguous::ContiguousOptimizeWithBaseFormat(self, src)) {
      // Optimized trans-contiguous method
      return;
    } else {
      // General trans-contiguous method
      RECORD_FUNCTION(
          "contiguous_d_AsStrided", std::vector<c10::IValue>({src}));
      op_plugin::npu_stride_copy_out(
          src, src.sizes(), src.strides(), src.storage_offset(), self);
      return;
    }
  } else {
    // Contiguous source tensor copy to contiguous self tensor
    int64_t numel = self.numel();
    if (numel == src.numel()) {
      RECORD_FUNCTION("d2dCopyAsync", std::vector<c10::IValue>({src}));
      ASCEND_LOGD("copy contiguous tensor inside device");
      return copy_d2d_by_memcpy(self, src, numel);
    }
  }
  // such as discontiguous tensor copy to unmatched tensor
  copy_d2d_last_method(self, src, true, non_blocking);
}

bool try_to_optimize_copy_with_any_format(
    at::Tensor& self,
    const at::Tensor& src) {
  // Some Ops support inputs with 5HD/NZ format, Transdata is redundant
  // Record:
  // Op:Reshape; SliceD || Supportformat: 5HD/NZ
  return TransContiguous::ContiguousOptimizeWithAnyFormat(self, src);
}

at::Tensor& NPUNativeFunctions::copy_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  if (self.numel() == 0) {
    return self;
  }
  // save tensor dim name
  c10::optional<at::DimnameList> names = src.opt_names();
  if (names.has_value()) {
    internal_set_names_inplace(self, names);
  }

  if (torch_backend::utils::is_npu(self)) {
    if (torch_backend::utils::is_npu(src)) {
      copy_d2d(self, src, non_blocking);
    } else {
      copy_h2d(self, src, non_blocking);
    }
  } else {
    if (torch_backend::utils::is_npu(src)) {
      copy_d2h(self, src, non_blocking);
    }
  }
  return self;
}

} // namespace native
} // namespace at_npu
