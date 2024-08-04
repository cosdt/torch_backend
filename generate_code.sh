#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

python_execute="$1"

source_yaml="$CDIR/csrc/aten/npu_native_functions.yaml"
op_plugin_yaml_path="$CDIR/csrc/aten/op_plugin_functions.yaml"

${python_execute} -m codegen.gen_backend_stubs  \
  --output_dir="csrc/aten/generated" \
  --source_yaml="$source_yaml" \
  --op_plugin_yaml_path="$op_plugin_yaml_path"

${python_execute} -m codegen.autograd.gen_autograd \
  --out_dir="$CDIR/csrc/aten/generated" \
  --autograd_dir="$CDIR/codegen/autograd" \
  --npu_native_function_dir="$source_yaml"
