#include <cstdio>
#include <string>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/lightning_indexer_tiling.h"
#include "defines.h"
#include "torch_helper.h"
#include "ge_helper.h"
#include "common_tiling.h"
#include "lightning_indexer_def.h"
#include "aclrtlaunch_lightning_indexer.h"

namespace sglang::LIHost {

using namespace ge_helper;
constexpr uint32_t MAX_CAPTURE_NUM = 1024;
uint32_t actualCaptureNum = 0;
std::unordered_map<uint32_t, uint32_t> captureMap;
at::Tensor workspace;

}

namespace sglang {
namespace npu_kernel {
HOST_API void lightning_indexer(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
                                const at::Tensor &actual_seq_lengths_q, const at::Tensor &actual_seq_lengths,
                                const at::Tensor &blocktable, c10::optional<c10::string_view> layout_query,
                                c10::optional<c10::string_view> layout_key, at::Tensor &sparse_indices)
{
    using namespace LIHost;
    std::cout << "0" << std::endl;
    LightningIndexer indexer("lightning_indexer");
    auto context = std::make_shared<TilingContext>("lightning_indexer");
    TORCH_CHECK(context != nullptr, "TilingContext is null");

    auto qScalarType = query.scalar_type();
    std::cout << "1" << std::endl;
    indexer.SetToContext(context, qScalarType);
    std::cout << "2" << std::endl;
    context->RegisterTensor(query, true);
    context->RegisterTensor(key, true);
    context->RegisterTensor(weights, true);
    context->RegisterTensor(actual_seq_lengths_q, true);
    context->RegisterTensor(actual_seq_lengths, true);
    context->RegisterTensor(blocktable, true);
    context->RegisterTensor(sparse_indices, false);
    std::cout << "3" << std::endl;

    LITilingInfo liInfo;
    LIInfoParser LIInfoParser(context.get());
    std::cout << "4" << std::endl;
    TORCH_CHECK(LIInfoParser.ParseAndCheck(liInfo) == ge::GRAPH_SUCCESS, "lightning_indexer ParseAndCheck failed")

    LightningIndexerTiling liTiling(context.get());
    liTiling.DoTiling(&liInfo);
    std::cout << "5" << std::endl;
    const auto &tilingData = liTiling.GetTilingData();

    uint32_t tilingSize = sizeof(LITilingData);
    auto blockDim = tilingData.usedCoreNum;
    auto bs = query.sizes()[0];

    static auto globalTilingData = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                             at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    if (captureMap.find(bs) == captureMap.end()) {
        TORCH_CHECK(actualCaptureNum < MAX_CAPTURE_NUM,
                    "lightning_indexer captureNum overflow")
        captureMap[bs] = actualCaptureNum;
        aclrtMemcpy(globalTilingData.data_ptr<uint8_t>() + actualCaptureNum * tilingSize, tilingSize,
                  &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
        actualCaptureNum++;
    }
    at::Tensor tilingTensor =
        at::from_blob(globalTilingData.data_ptr<uint8_t>() + (tilingSize * captureMap[bs]), tilingSize, at::kByte);

    size_t userWorkspaceSize = *context->GetWorkspaceSizes(1);
    workspace =
        at::empty({userWorkspaceSize}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    EXEC_KERNEL_CMD(lightning_indexer, blockDim, query, key, weights, actual_seq_lengths_q, actual_seq_lengths, blocktable,
                    sparse_indices, workspace, tilingTensor);
}
}  // namespace LIHost
}  // namespace sglang
