#include "aten/common/from_blob.h"
#include <ATen/Utils.h>
#include <c10/core/Allocator.h>
#include "aten/common/TensorFactories.h"
#include "core/NPUException.h"
#include "csrc/backend/NPUCachingAllocator.h"
#include "csrc/backend/NPUFunctions.h"
#include "csrc/backend/NPUStorageImpl.h"
#include "framework/StorageDescHelper.h"
#include "framework/utils/OpAdapter.h"

namespace at_npu {

namespace native {

at::Tensor TensorMaker::make_tensor() {
  if (device_ == c10::nullopt) {
    device_ = c10::Device(
        at::DeviceType::PrivateUse1, c10::backend::current_device());
  }

  if (opts_.device().has_index()) {
    TORCH_CHECK_VALUE(
        opts_.device() == *device_,
        "Specified device ",
        opts_.device(),
        " does not match device of data ",
        *device_);
  }
  AT_ASSERT(
      (*device_).type() == c10::DeviceType::PrivateUse1,
      OPS_ERROR(ErrCode::PARAM));
  at::globalContext().lazyInitPrivateUse1();

  auto dtype = c10::scalarTypeToTypeMeta(
      dtype_or_default(c10::optTypeMetaToScalarType(opts_.dtype_opt())));
  at_npu::native::check_size_nonnegative(sizes_);
  c10::DeviceGuard guard(*device_);
  c10::Allocator* allocator = c10::backend::Allocator::get();

  std::size_t size_bytes = computeStorageSize();

  c10::DataPtr data_ptr{data_, *device_};

  c10::intrusive_ptr<c10::StorageImpl> storage_impl =
      c10::make_intrusive<c10::backend::NPUStorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size_bytes,
          std::move(data_ptr),
          allocator,
          true);

  auto tensor =
      at::detail::make_tensor<c10::backend::NPUTensorImpl>(storage_impl, dtype);

  at_npu::native::StorageDescHelper::SetDesc(tensor, sizes_, tensor.strides());

  at::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  if (strides_) {
    tensor_impl->set_sizes_and_strides(sizes_, *strides_);
  } else {
    tensor_impl->set_sizes_contiguous(sizes_);
  }
  if (storage_offset_) {
    tensor_impl->set_storage_offset(*storage_offset_);
  }

  return tensor;
}

std::size_t TensorMaker::computeStorageSize() const noexcept {
  std::size_t itemsize = opts_.dtype().itemsize();

  if (strides_) {
    auto storage_size =
        at::detail::computeStorageNbytes(sizes_, *strides_, itemsize);
    if (storage_offset_) {
      storage_size += storage_offset_.value();
    }
    return storage_size;
  }

  std::size_t size = 1;
  for (std::int64_t s : sizes_) {
    size *= static_cast<std::size_t>(s);
  }
  auto storage_size = size * itemsize;
  if (storage_offset_) {
    storage_size += storage_offset_.value();
  }
  return storage_size;
}

at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    int64_t storage_offset,
    const at::TensorOptions& options,
    const c10::optional<c10::Device> target_device) {
  return for_blob(data, sizes)
      .strides(strides)
      .storage_offset(storage_offset)
      .options(options)
      .target_device(target_device)
      .make_tensor();
}

at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const c10::optional<c10::Device> target_device,
    const at::TensorOptions& options) {
  return for_blob(data, sizes)
      .options(options)
      .target_device(target_device)
      .make_tensor();
}

at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  return for_blob(data, sizes).strides(strides).options(options).make_tensor();
}

at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const at::TensorOptions& options) {
  return for_blob(data, sizes).options(options).make_tensor();
}

} // namespace native

} // namespace at_npu
