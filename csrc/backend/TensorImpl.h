#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include "csrc/backend/StorageImpl.h"

namespace c10::backend {

// TensorImpl class is derived from c10::TensorImpl, and it is only used to
// handle an device tensor.
class TensorImpl : public c10::TensorImpl {
 public:
  explicit TensorImpl(
      c10::Storage&& storage,
      const caffe2::TypeMeta& data_type);

  void shallow_copy_from(const c10::intrusive_ptr<c10::TensorImpl>& impl) final;

  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const final;
  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const final;

 public:
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = default;
  TensorImpl& operator=(TensorImpl&&) = default;
  ~TensorImpl();
};

} // namespace c10::backend
