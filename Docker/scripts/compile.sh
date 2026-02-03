#!/bin/bash

# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

set -e

_curfile=$(realpath $0)
script_dir=$(dirname $_curfile)

function usage() {
    echo "usage:"
    echo "compile.sh [-m model_file (required)] [-n name (optional)] [-c calib_input_list (optional)] [-o output_dir (optional)] [-i custom_io_file (optional)] [-h show help message]"
}
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

model_file=""
calib_input_list=""
name=""
output=""

# Add to the variables section at the beginning
custom_io=""

# Add to the getopts section
while getopts "m:n:c:o:i:h:" arg
do
    case $arg in 
        m)
            model_file=${OPTARG}
            ;;
        n)
            name=${OPTARG}
            ;;
        c)
            calib_input_list=${OPTARG}
            ;;
        o)
            output=${OPTARG}
            ;;
        i)
            custom_io=${OPTARG}
            ;;
        h)
            usage
            exit
            ;;
        ?)
            usage
            exit 1
            ;;
    esac
done

if [ -z "${model_file}" ]; then
    echo -e "${YELLOW}Model file not provided. Please specify with '-m <model_file>'${NC}"
    exit 1
fi

if [ ! -f "${model_file}" ]; then
    echo -e "${YELLOW}Error: Model file '${model_file}' not found.${NC}"
    exit 1
fi

filename=$(basename -- "$model_file")
extension="${filename##*.}"

if [ -z "${name}" ]; then
    name="${filename%.*}"
fi

if [ -z "${output}" ]; then
    output="./${name}_output"
fi
output=$(realpath "$output")

tmp_output="/tmp/tmp_${name}_output"

# Check Model Type
if [[ "$extension" == "onnx" ]]; then
    model_type="onnx"
elif [[ "$extension" == "tflite" ]]; then
    model_type="tflite"
else
    echo -e "${YELLOW}Unsupported model format: ${extension}. Please provide an ONNX or TFLite model.${NC}"
    exit 1
fi

# Double check if QNN_SDK_ROOT is set
if [ ! -n "${QNN_SDK_ROOT}" ]; then
    echo -e "${YELLOW}QNN_SDK_ROOT not found.${NC}"
    exit 1
fi

# Activate QNN Virtual Environment
if [ -f "${QNN_SDK_ROOT}/qnn229_env/bin/activate" ]; then
    source "${QNN_SDK_ROOT}/qnn229_env/bin/activate"
else
    echo -e "${YELLOW}Warning: Virtual environment not found. Proceeding without activation.${NC}"
fi

# Ensure QNN SDK Environment is Sourced
if ! hash qnn-onnx-converter 2>/dev/null && ! hash qnn-tflite-converter 2>/dev/null; then
    echo -e "${YELLOW}QNN SDK environment is not set. Run:${NC}"
    echo -e "${YELLOW}>> cd ${QNN_SDK_ROOT}/bin; source envsetup.sh; cd -${NC}"
    exit 1
fi

# Remove Existing Output Directory
if [ -d "$output" ]; then
    rm -rf "$output"
fi
mkdir -p "$output"

# Remove Existing Temporary Output Directory
if [ -d "$tmp_output" ]; then
    rm -rf "$tmp_output"
fi
mkdir -p "$tmp_output"

# Extra Arguments for Quantization
extra_args=""
if [ -n "${calib_input_list}" ]; then
    echo "Calibration input list: ${calib_input_list}"
    extra_args="--input_list ${calib_input_list}"
fi

# Model Conversion
if [ "$model_type" == "onnx" ]; then
    echo "🚀 Converting ONNX model..."
    custom_io_args=""
    if [ -n "${custom_io}" ]; then
        custom_io_args="--custom_io ${custom_io}"
    fi
    qnn-onnx-converter --input_network "${model_file}" --debug ${extra_args} \
        --output_path "${tmp_output}/${name}.cpp" --float_bw 16 ${custom_io_args}
elif [ "$model_type" == "tflite" ]; then
    echo "🚀 Converting TFLite model..."
    custom_io_args=""
    if [ -n "${custom_io}" ]; then
        custom_io_args="--custom_io ${custom_io}"
    fi
    qnn-tflite-converter --input_network "${model_file}" --debug ${extra_args} \
        --output_path "${tmp_output}/${name}.cpp" --float_bw 16 ${custom_io_args}
fi

# Generate Model Library
qnn-model-lib-generator -c "${tmp_output}/${name}.cpp" -b "${tmp_output}/${name}.bin" -o "${tmp_output}"

# Quantized Model Context Binary Generation
#if [ -n "${calib_input_list}" ]; then
backend="${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so"
qnn-context-binary-generator --backend "${backend}" \
    --model "${tmp_output}/x86_64-linux-clang/lib${name}.so" --binary_file "${name}.serialized" \
    --config_file /opt/config/HtpConfigFile.json \
    --output_dir "${tmp_output}"
#fi

# Generate Waterdrop Model Config
context_binary_file="${tmp_output}/${name}.serialized.bin"
if [ -f "${context_binary_file}" ]; then
    python3 "${script_dir}/qnn_to_waterdrop_model_config.py" "${tmp_output}/${name}_net.json" "${context_binary_file}"
fi

# Copy context_binary_file to output directory
if [ -f "${tmp_output}/waterdrop/0/${name}.serialized.bin" ]; then
    cp "${tmp_output}/waterdrop/0/${name}.serialized.bin" "${output}"
fi

# Copy ${name}_net.json to output directory
if [ -f "${tmp_output}/waterdrop/0/model.json" ]; then
    cp "${tmp_output}/waterdrop/0/model.json" "${output}"
fi

echo -e "${GREEN}🎉 Done! Converted model saved in ${output}${NC}"
