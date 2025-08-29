#include <iostream>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_data.h"
#include "defines.h"
#include "torch_helper.h"
#include "aclrtlaunch_pp_matmul_einsum.h"

namespace sglang {
namespace npu_kernel {
using namespace pp_matmul;

std::unordered_map<c10::string_view, uint16_t> quantModeMap = {
    {"per_channel_symm", 0},
    {"per_channel_asymm", 1},
    {"per_token_symm", 2},
};

std::unordered_map<c10::string_view, uint16_t> formatModeMap = {
    {"nd", 0},
    {"nz", 1},
};

template<typename MapType>
inline int GetModeVal(const MapType& mode_map,
                       c10::optional<c10::string_view> mode_opt,
                       c10::string_view default_mode,
                       const char* mode_name)
{
    c10::string_view mode_str = mode_opt.value_or(default_mode);
    auto it = mode_map.find(mode_str);
    // if input mode is unsupported, use default value
    OP_CHECK(it != mode_map.end(), "Unsupported mode value" + mode_str, 0);
    return it->second;
}

HOST_API bool pp_matmul_einsum(const at::Tensor &tensor_a, const at::Tensor &tensor_b, at::Tensor &tensor_c,
    c10::optional<c10::string_view> format_mode, c10::optional<c10::string_view> quant_mode)
{
    auto tensorAShape = tensor_a.sizes();
    auto tensorBShape = tensor_b.sizes();
    auto tensorCShape = tensor_c.sizes();
    uint32_t n;
    uint32_t block_dim;
    HardwareInfo hwInfo;
    std::map<TensorDType, float> dTypeMap = {
        {TENSOR_DTYPE_INT8, 1.0}, {TENSOR_DTYPE_FLOAT16, 2.0}, {TENSOR_DTYPE_BF16, 2.0}, {TENSOR_DTYPE_FLOAT, 4.0}};

    at::ScalarType aType = tensor_a.scalar_type();
    at::ScalarType bType = tensor_b.scalar_type();
    at::ScalarType cType = tensor_c.scalar_type();
    OP_CHECK(aType == bType && bType == cType, "tensor type is not the same", return false);
    OP_CHECK(aType != at::ScalarType::BFloat16 && aType != at::ScalarType::Half, "tensor type only support half or bf16", return false);

    TensorFormat formatMode = static_cast<TensorFormat>(GetModeVal(formatModeMap, format_mode, "nd", "format_mode"));
    MatMul::QuantMode quantMode = static_cast<MatMul::QuantMode>(GetModeVal(quantModeMap, quant_mode, "per_channel_symm", "quant_mode"));

    OP_CHECK(tensorAShape.size() == 3, "batch size is not same between srcTensor and dstTensor", return false);
    if (formatMode == TensorFormat::TENSOR_FORMAT_ND) {
        OP_CHECK(tensorBShape.size() == 3, "tensor shape should be dim3 in nd format", return false);
        OP_CHECK(tensorAShape[2] == tensorBShape[1], "tensor shape is wrong", return false);
        n = tensorBShape[2];
    } else {
        OP_CHECK(tensorBShape.size() == 4, "tensor shape should be dim4 in nz format", return false);
        OP_CHECK(tensorAShape[2] == tensorBShape[2], "tensor shape is wrong", return false);
        n = tensorBShape[1] * tensorBShape[3];
    }
    OP_CHECK(tensorAShape[1] == tensorBShape[0], "tensor shape is wrong", return false);

    OpShape opShape = {
        .batchSize = tensorAShape[1],
        .m = tensorAShape[0],
        .k = tensorAShape[2],
        .n = n
    };
    PpMatmulTilingData matmulTilingData = {
        .opShape = opShape,
    }
    MatMulInfo mmInfo = {
        .batchSize = opShape.batchSize,
        .m = opShape.m,
        .k = opShape.k,
        .n = opShape.n,
        .dtypeA = aType,
        .dtypeB = bType,
        .dtypeC = cType,
        .formatB = formatMode,
        .mmType = MatMul::MatMulType::MATMUL_EIN_SUM,
        .inDtype = dTypeMap[aType],
        .outDtype = dTypeMap[cType],
        .quantMode = quantMode
    };
    GetPpMatmulTiling(mmInfo, hwInfo, block_dim, matmulTilingData);
    PpMatmulTilingCheck(matmulTilingData);

    at::Tensor buffer = at::from_blob((uint8_t *)&matmulTilingData, sizeof(PpMatmulTilingData), at::kByte);
    at::Tensor tiling = TorchNpuHepler::CopyTensorHostToDevice(buffer);
    EXEC_KERNEL_CMD(pp_matmul_einsum, block_dim, tensor_a, tensor_b, tensor_c, tiling);
    return true;
}

} // namespace npu_kernel

} // namespace sglang
