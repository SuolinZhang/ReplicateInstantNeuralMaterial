/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Texture/Synthesis.h"
#include "Utils/Neural/MLP.h"
#include "Utils/Neural/NBTF.h"
#include "Utils/Neural/MLPCuda.h"
#include "Utils/Neural/cuda/CUDADefines.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace Falcor;

enum class ModelName : uint32_t
{
    LEATHER11
};

FALCOR_ENUM_INFO(
    ModelName,
    {{ModelName::LEATHER11, "UBO Leather11"}
    }
);
FALCOR_ENUM_REGISTER(ModelName);

struct ModelInfo
{
    std::string name;
    std::string hfName;
    bool HDRBTF;
    float scales[8]; // quantization scales
};

/**
 * Minimal path tracer.
 *
 * This pass implements a minimal brute-force path tracer. It does purposely
 * not use any importance sampling or other variance reduction techniques.
 * The output is unbiased/consistent ground truth images, against which other
 * renderers can be validated.
 *
 * Note that transmission and nested dielectrics are not yet supported.
 */
class NeuralMatRendering : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(NeuralMatRendering, "NeuralMatRendering", "Minimal path tracer.");

    static ref<NeuralMatRendering> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<NeuralMatRendering>(pDevice, props);
    }

    NeuralMatRendering(ref<Device> pDevice, const Properties& props);
    virtual ~NeuralMatRendering();

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return mpPixelDebug->onMouseEvent(mouseEvent); }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    void tracingPass(RenderContext* pRenderContext, const RenderData& renderData);
    void cudaInferPass(RenderContext* pRenderContext, const RenderData& renderData);
    void displayPass(RenderContext* pRenderContext, const RenderData& renderData);
    void loadNetwork(RenderContext* pRenderContext);

private:
    void prepareVars();

    /// Current scene.
    ref<Scene> mpScene;
    /// GPU sample generator.
    ref<SampleGenerator> mpSampleGenerator;
    /// Frame count since scene was loaded.
    uint mFrameCount = 0;
    bool mOptionsChanged = false;

    // Ray tracing program.
    struct
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    } mTracer;
    ref<ComputePass> mpDisplayPass;
    std::string mProjectPath = getProjectDirectory().string();

    ModelName mModelName = ModelName::LEATHER11;

    ModelInfo mModelInfo[1] = {
        {"leather11_int8",
         "leather11.png",
         false,
         {0.0015868720f,
          0.0000177248f,
          0.0030352981f,
          0.0000292827f,
          0.0051905843f,
          0.0000358183f,
          0.0066363174f,
          0.0000363468f}}
        };


    // displacement map
    ref<Texture> mpHF;
    // max filter sampler for HF texel fetch.
    ref<Sampler> mpMaxSampler;
    std::unique_ptr<PixelDebug> mpPixelDebug;

    // cuda inference output buffer
    ref<Buffer> mpOutputBuffer;
    ref<Buffer> mpValidBuffer;
    ref<Buffer> mpPackedInputBuffer;
    ref<Buffer> mpScaleBuffer;

    Falcor::float4 mControlParas = Falcor::float4(0.1, 10, 0, 0.099);

    ACFCurve mCurveType = ACFCurve::X;
    Falcor::float2 point_data[5] = {
        Falcor::float2(0.0f, 1.0f),
        Falcor::float2(0.0f, 1.0f),
        Falcor::float2(1.0f, 0.0f),
        Falcor::float2(1.0f, 0.0f),
        Falcor::float2(0.0f, 0.0f)};

    float point_data_curve[1] = {0};

    bool mApplySyn = true;

    /// GPU fence for synchronizing readback.
    ref<Fence> mpFence;

    std::unique_ptr<TextureSynthesis> mpTextureSynthesis;
    std::shared_ptr<NBTF> mpNBTFInt8;
    std::shared_ptr<NBTF> mpNBTF[1];

    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;

    bool mShowTracedHF = false;
    bool mTracedShadowRay = true;
    bool mHDRBTF = false;

    Falcor::float3 mEnvRotAngle = Falcor::float3(0.0f, 0.0f, 0.0f);
    // cuda
    float mCudaTime = 0.0;
    double mCudaAvgTime = 0.0;
    int cudaInferTimes = 1;
    cudaEvent_t mCudaStart, mCudaStop;

    uint mCudaAccumulatedFrames = 1;
};
