#ifndef SGLANG_KERNEL_GE_HELPER_H
#define SGLANG_KERNEL_GE_HELPER_H
#include <stdint>
#include <vector>
#include <any>
#include <map>
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"

namespace sglang {
namespace ge_helper {

enum class ParamType : uint32_t {
    REQUIRED = 0,
    OPTIONAL,
};
using AttrType = ParamType;

#define MAP_SCALAR_TYPE_TO_GE_DATATYPE(scalar_type)                                                                    \
    [&]() {                                                                                                            \
        switch (scalar_type) {                                                                                         \
            case at::ScalarType::Float:                                                                                \
                return ge::DT_FLOAT;                                                                                   \
            case at::ScalarType::Half:                                                                                 \
                return ge::DT_FLOAT16;                                                                                 \
            case at::ScalarType::Char:                                                                                 \
                return ge::DT_INT8;                                                                                    \
            case at::ScalarType::Int:                                                                                  \
                return ge::DT_INT32;                                                                                   \
            case at::ScalarType::Byte:                                                                                 \
                return ge::DT_UINT8;                                                                                   \
            case at::ScalarType::Short:                                                                                \
                return ge::DT_INT16;                                                                                   \
            case at::ScalarType::UInt16:                                                                               \
                return ge::DT_UINT16;                                                                                  \
            case at::ScalarType::UInt32:                                                                               \
                return ge::DT_UINT32;                                                                                  \
            case at::ScalarType::Long:                                                                                 \
                return ge::DT_INT64;                                                                                   \
            case at::ScalarType::UInt64:                                                                               \
                return ge::DT_UINT64;                                                                                  \
            case at::ScalarType::Double:                                                                               \
                return ge::DT_DOUBLE;                                                                                  \
            case at::ScalarType::Bool:                                                                                 \
                return ge::DT_BOOL;                                                                                    \
            case at::ScalarType::BFloat16:                                                                             \
                return ge::DT_BF16;                                                                                    \
            default:                                                                                                   \
                throw std::runtime_error("Unsupported scalar type: " + std::to_string(static_cast<int>(scalar_type))); \
        }                                                                                                              \
    }()

class InputDef
{
public:
    InputDef &ParamType(ParamType type)
    {
        paramType_ = type;
        return *this;
    }

    InputDef &DataType(const std::vector<ge::DataType> &types)
    {
        dataTypes = types;
        return *this;
    }

    InputDef &DataTypeList(const std::vector<ge::DataType> &types)
    {
        useDataTypeList_ = true;
        dataTypes = types;
        return *this;
    }

    InputDef &Format(const std::vector<ge::Format> &formats)
    {
        formats_ = formats;
        return *this;
    }
    InputDef &FormatList(const std::vector<ge::Format> &formats)
    {
        useFormatList_ = true;
        formats_ = formats;
        return *this;
    }

    InputDef &AutoContiguous()
    {
        autoContiguous_ = true;
        return *this;
    }

    ge::DataType GetDataType(uint32_t index) const
    {
        if (useDataTypeList_) {
            return dataTypes[0];
        }
        if (index >= dataTypes.size()) {
            throw std::out_of_range("InputDef::GetDataType index out of range");
        }
        return dataTypes[index];
    }

    const std::vector<ge::DataType> &GetDataTypes() const
    {
        return dataTypes;
    }

    ge::Format GetFormat(uint32_t index) const
    {
        if (useFormatList_) {
            return formats_[0];
        }
        if (index >= formats_.size()) {
            throw std::out_of_range("InputDef::GetFormat index out of range");
        }
        return formats_[index];
    }

private:
    ParamType paramType_;
    std::vector<ge::DataType> dataTypes;
    std::vector<ge::Format> formats_;
    bool autoContiguous_ = false;
    bool useFormatList_ = false;
    bool useDataTypeList_ = false;
};

class AttrDef
{
public:
    AttrDef &AttrType(AttrType type)
    {
        attrType_ = type;
        return *this;
    }

    AttrDef &String(const std::string &value)
    {
        if (valueInitialized_) {
            throw std::runtime_error("Cannot set default value for an attribute that has already been initialized.");
        }
        defaultValue_ = value;
        valueInitialized_ = true;
        return *this;
    }

    AttrDef &Int(int value)
    {
        if (valueInitialized_) {
            throw std::runtime_error("Cannot set default value for an attribute that has already been initialized.");
        }
        defaultValue_ = value;
        valueInitialized_ = true;
        return *this;
    }

    std::any GetValue() const
    {
        return defaultValue_;
    }

private:
    AttrType attrType_;      // REQUIRED or OPTIONAL
    std::any defaultValue_;  // need C++17
    bool valueInitialized_ = false;
};

// TODO: Do automatic registery template class at compile time
class OpDef
{
public:
    explicit OpDef(const std::string &name) : opName_(name) {}

    InputDef &Input(const std::string &name)
    {
        inputs_.emplace_back(name, InputDef());
        return inputs_.back().second;
    }

    AttrDef &Attr(const std::string &name)
    {
        attrs_.emplace_back(name, AttrDef());
        return attrs_.back().second;
    }

    InputDef &Output(const std::string &name)
    {
        outputs_.emplace_back(name, OutputDef());
        return outputs_.back().second;
    }

    void SetToContext(std::shared_ptr<TilingContext> &context, at::ScalarType &scalarType)
    {
        auto geType = MAP_SCALAR_TYPE_TO_GE_DATATYPE(scalarType);
        auto &firstParamTypes = inputs_[0].second.GetDataTypes() uint32_t index = 0;
        for (; index < firstParamTypes.size(); index++) {
            if (firstParamTypes[index] == geType) {
                break;
            }
        }
        if (index == firstParamTypes.size()) {
            throw std::runtime_error("Invalid input type, please check the definition file");
        }

        for (auto &input : inputs_) {
            auto tensorDesc = std::make_shared<gert::CompileTimeTensorDesc>()
                                  tensorDesc->SetDataType(input.second.GetDataType(index));
            tensorDesc->SetOriginFormat(input.second.GetFormat(index));
            context->AddInputDesc(tensorDesc);
        }
        for (auto &output : outputs_) {
            auto tensorDesc = std::make_shared<gert::CompileTimeTensorDesc>()
                                  tensorDesc->SetDataType(output.second.GetDataType(index));
            tensorDesc->SetOriginFormat(output.second.GetFormat(index));
            context->AddOutputDesc(tensorDesc);
        }
        auto runtimeAttrs = std::make_shared<RuntimeAttrs>();
        for (auto &attr : attrs_) {
            if (attr.second.GetValue().type() == typeid(std::string)) {
                runtimeAttrs->SetStr(attr.second.GetValue());
                runtimeAttrs->SetAny(std::any{});
            } else {
                runtimeAttrs->SetAny(attr.second.GetValue());
                runtimeAttrs->SetStr("");
            }
        }
        context->SetAttrs(runtimeAttrs);
    }

private:
    std::string opName_;
    std::vector<std::pair<std::string, InputDef>> inputs_;
    std::vector<std::pair<std::string, InputDef>> outputs_;
    std::vector<std::pair<std::string, AttrDef>> attrs_;
};

class RuntimeAttrs
{
public:
    RuntimeAttrs() = default;

    void SetStr(const std::string &value)
    {
        strValues.push_back(value);
    }

    void SetAny(const std::any &value)
    {
        values.push_back(value);
    }

    const char *GetStr(const size_t index) const
    {
        return strValues[index].c_str();
    }

    template <typename T>
    const T *GetAttrPointer(size_t index) const
    {
        std::any anyValue = values[index];
        if (anyValue.type() != typeid(T)) {
            throw std::runtime_error("Invalid attribute type.");
        }
        return &std::any_cast<T>(anyValue);
    }

private:
    std::vector<std::string> strValues;
    std::vector<std::any> values;
};

class TilingContext
{
public:
    TilingContext(const std::string &nodeName) : nodeName_(nodeName) {}

    void RegisterTensor(const at::Tensor &tensor, bool isInput)
    {
        // convert to gert::Tensor and add to inputTensor_
        // get shape and convert to gert::StorageShape, then add to inputShape_
        std::vector<std::shared_ptr<gert::StorageShape>> *shapePtr;
        std::vector<std::shared_ptr<gert::Tensor>> *tensorPtr;
        std::vector<std::shared_ptr<gert::CompileTimeTensorDesc>> *descPtr;

        if (isInput) {
            shapePtr = &inputShape_;
            tensorPtr = &inputTensor_;
            descPtr = &inputDesc_;
        } else {
            shapePtr = &outputShape_;
            tensorPtr = &outputTensor_;
            descPtr = &outputDesc_;
        }

        auto shape = tensor.sizes();
        std::vector<int64_t> shapeVec(shape.begin(), shape.end());

        auto storageShape = std::make_shared<gert::StorageShape>(shapeVec, shapeVec);
        shapePtr->push_back(storageShape);

        // Safety check to avoid underflow
        if (descPtr->empty()) {
            throw std::runtime_error("No tensor description available");
        }

        auto index = descPtr->size() - 1;
        // storageFormat == originFormat
        auto storageFormat = (*descPtr)[index]->GetOriginFormat();
        auto dataType = (*descPtr)[index]->GetDataType();
        auto geTensor = std::make_shared<gert::Tensor>(storageShape, storageFormat, dataType);
        tensorPtr->push_back(geTensor);
    }

    std::shared_ptr<gert::CompileTimeTensorDesc> &GetInputDesc(uint32_t index) const
    {
        return inputDesc_[index];
    }

    std::shared_ptr<gert::StorageShape> &GetInputShape(uint32_t index) const
    {
        return inputShape_[index];
    }

    std::shared_ptr<gert::Tensor> &GetInputTensor(uint32_t index) const
    {
        return inputTensor_[index];
    }

    std::shared_ptr<gert::CompileTimeTensorDesc> &GetOptionalInputDesc(uint32_t index) const
    {
        return inputDesc_[index];
    }

    std::shared_ptr<gert::StorageShape> &GetOptionalInputShape(uint32_t index) const
    {
        return inputShape_[index];
    }

    std::shared_ptr<gert::Tensor> &GetOptionalInputTensor(uint32_t index) const
    {
        return inputTensor_[index];
    }

    std::shared_ptr<gert::CompileTimeTensorDesc> &GetOutputDesc(uint32_t index) const
    {
        return outputDesc_[index];
    }

    std::shared_ptr<gert::StorageShape> &GetOutputShape(uint32_t index) const
    {
        return outputShape_[index];
    }

    std::shared_ptr<gert::Tensor> &GetOutputTensor(uint32_t index) const
    {
        return outputTensor_[index];
    }

    const char *GetNodeName() const
    {
        return nodeName_.c_str();
    }

    auto GetPlatformInfo()
    {
        return platform_ascendc::PlatformAscendCManager::GetInstance();
    }

    std::shared_ptr<RuntimeAttrs> &GetAttrs() const
    {
        return runtimeAttrs_;
    }

    void AddInputDesc(std::shared_ptr<gert::CompileTimeTensorDesc> desc)
    {
        inputDesc_.push_back(desc);
    }

    void AddOutputDesc(std::shared_ptr<gert::CompileTimeTensorDesc> desc)
    {
        outputDesc_.push_back(desc);
    }

    void SetAttrs(std::shared_ptr<RuntimeAttrs> runtimeAttrs)
    {
        runtimeAttrs_ = runtimeAttrs;
    }

    void SetWorkspaceSize

        // Deleted, do not need to use these functions
        void
        SetBlockDim(int blockDim) = delete;
    void SetTilingKey(int tilingKey) = delete;
    size_t *GetWorkspaceSizes(uint32_t index) const = delete;
    gert::TilingData *GetRawTilingData() const = delete;

private:
    // init from user definition
    // input include input and optional input (for adapt aclnn)
    std::vector<std::shared_ptr<gert::CompileTimeTensorDesc>> inputDesc_;
    std::vector<std::shared_ptr<gert::CompileTimeTensorDesc>> outputDesc_;
    std::shared_ptr<RuntimeAttrs> runtimeAttrs_;

    // init from constructor
    std::vector<std::shared_ptr<gert::Tensor>> inputTensor_;
    std::vector<std::shared_ptr<gert::Tensor>> outputTensor_;
    std::vector<std::shared_ptr<gert::StorageShape>> inputShape_;
    std::vector<std::shared_ptr<gert::StorageShape>> outputShape_;

    std::string nodeName_;
    gert::TilingData *rawTilingData_ = nullptr;
    size_t systemWorkSpaceSize_ = 0;
    size_t userWorkSpaceSize_ = 0;
    std::vector<size_t *> workSpaceSize_{&systemWorkSpaceSize_, &userWorkSpaceSize_};
};

}  // namespace ge_helper
}  // namespace sglang
#endif
