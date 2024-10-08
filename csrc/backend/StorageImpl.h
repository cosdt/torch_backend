#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/order_preserving_flat_hash_map.h>
#include <c10/util/typeid.h>

// TODO(FFFrog):
// Remove later
#include "acl/include/acl/acl_base.h"
#include "acl/include/acl/acl_rt.h"

namespace c10::backend {

struct StorageDesc {
 public:
  struct use_byte_size_t {};

  c10::SmallVector<int64_t, 5> base_sizes_;
  c10::SmallVector<int64_t, 5> base_strides_;
  c10::SmallVector<int64_t, 5> storage_sizes_;
  int64_t base_offset_ = 0; // no use
  use_byte_size_t base_dtype_ = {}; // no use
  aclFormat origin_format_ = ACL_FORMAT_UNDEFINED;
  aclFormat npu_format_ = ACL_FORMAT_ND;
  // used to make CANN GE tensor from storagImpl
  caffe2::TypeMeta data_type_;
};

struct DeviceStorageImpl : public c10::StorageImpl {
  explicit DeviceStorageImpl(
      use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);
  ~DeviceStorageImpl() override = default;

  void release_resources() override;

  StorageDesc storage_desc_;

  StorageDesc get_device_desc() const {
    return storage_desc_;
  }
};

c10::intrusive_ptr<c10::StorageImpl> make_device_storage_impl(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable);

} // namespace c10::backend
