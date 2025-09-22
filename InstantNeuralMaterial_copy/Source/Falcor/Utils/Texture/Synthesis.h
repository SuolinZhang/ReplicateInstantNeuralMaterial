#pragma once
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/Program/ShaderVar.h"
#include "SynthesisUtils.h"
#include <memory>
#include <random>

namespace Falcor
{




/**
 * Implements Texture Synthesis
 */
class FALCOR_API TextureSynthesis
{
public:
    /**
     * Create an alias table.
     * The weights don't need to be normalized to sum up to 1.
     * @param[in] pDevice GPU device.
     * @param[in] weights The weights we'd like to sample each entry proportional to.
     * @param[in] rng The random number generator to use when creating the table.
     */
    TextureSynthesis();

    void bindShaderData(const ShaderVar& var) const;

    void readHFData(std::string hfPath, ref<Device> pDevice) ;
    void bindHFData(const ShaderVar& var);
    void bindMap(const ShaderVar& var);
    void updateMap(uint dim, ref<Device> pDevice);
    void updateMap(uint dim, ref<Device> pDevice, float2* ctrl_point, ACFCurve curve);
    void precomputeFeatureData(std::vector<float> data, uint2 dataDim, ref<Device> pDevice);
    void bindFeatureData(const ShaderVar& var);

    std::vector<float> getTData() { return TData; }
    std::vector<float> getInvTData() { return invTData; }
    std::vector<float> getAcfWeight() { return acfWeight; }
    std::vector<float> getSampleUV() { return sample_uv_list; }

    ref<Texture> mpHFT;
    ref<Texture> mpACF;

private:
    float HTRotStength = 0.5f;
    ref<Texture> mpColor;
    ref<Texture> mpHFInvT;
    ref<Texture> mpFeatureT;
    ref<Texture> mpFeatureInvT;
    ref<Buffer> mpACFBuffer;
    ref<Buffer> mpACFInputBuffer;

    ref<ComputePass> mpComputeACFPass;

    std::vector<float> TData;
    std::vector<float> invTData;
    std::vector<float> acfWeight;
    std::vector<float> acfPDF;
    std::vector<float> acfPDFImg;
    std::vector<float> sample_uv_list;
    ref<Buffer> mpMap;
};

}
