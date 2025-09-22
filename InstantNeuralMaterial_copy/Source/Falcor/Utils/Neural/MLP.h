// MLP.h

#pragma once
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/Program/ShaderVar.h"
#include <memory>
#include <random>

namespace Falcor
{

class FALCOR_API MLP
{
public:

    MLP(ref<Device> pDevice, std::string networkPath);

    void bindShaderData(const ShaderVar& var) const;
    void bindDebugData(const ShaderVar& var, ref<Buffer> w, ref<Buffer> b) const;
    void bindDebugData(const ShaderVar& var, ref<Buffer> w) const;


private:
    ref<Buffer> mpWeights;
    ref<Buffer> mpBias;
    ref<Buffer> mpMeta;
    ref<Texture> mpFeatureTex;
    std::string mNetworkName;
    int mLayerNum;
    int mMaxDim;
};

}
