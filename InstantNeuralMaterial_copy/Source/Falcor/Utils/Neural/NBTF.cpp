#include "NBTF.h"

#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include "Utils/Math/FormatConversion.h"
#include "IOHelper.h"
#include "cuda/TextureHelper.h"
#include <fstream>
#include <cuda_runtime.h>
namespace Falcor
{
std::vector<float> readBinaryFile(const char* filename);

void NBTF::loadFeature(ref<Device> pDevice, std::string featurePath)
{
    std::filesystem::path projectDir = getProjectDirectory();
    std::vector<float> PlaneMetaBuffer =
        readBinaryFile(fmt::format("{}/media/neural_materials/networks/PlaneMeta_{}.bin", projectDir.string(), featurePath).c_str());
    mUP.texDim = int2(PlaneMetaBuffer[0], PlaneMetaBuffer[1]);
    mHP.texDim = int2(PlaneMetaBuffer[2], PlaneMetaBuffer[3]);
    mDP.texDim = int2(PlaneMetaBuffer[4], PlaneMetaBuffer[5]);
    logInfo("[NBTF] Plane Dims");
    logInfo("[NBTF] U: {}, H: {}, D: {}", mUP.texDim, mHP.texDim, mDP.texDim);

    std::vector<float> DPlaneBuffer =
        readBinaryFile(fmt::format("{}/media/neural_materials/networks/DPlane_{}.bin", projectDir.string(), featurePath).c_str());
    std::vector<float> UPlaneBuffer =
        readBinaryFile(fmt::format("{}/media/neural_materials/networks/UPlane_{}.bin", projectDir.string(), featurePath).c_str());
    std::vector<float> HPlaneBuffer =
        readBinaryFile(fmt::format("{}/media/neural_materials/networks/HPlane_{}.bin", projectDir.string(), featurePath).c_str());

    ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;

    if (mHistogram)
    {
        mpTextureSynthesis->precomputeFeatureData(UPlaneBuffer, mUP.texDim, pDevice);
    }

    // std::vector<float> temp;
    // temp.resize(UPlaneBuffer.size());
    // for (int j = 0; j < mUP.texDim.y; j++)
    // {
    //     uint offset = j * mUP.texDim.x * mUP.texDim.x * 4;
    //     for (uint i = 0; i < UPlaneBuffer.size() / (4 * mUP.texDim.y); i++)
    //     {
    //         uint id = i / mUP.texDim.x + (i % mUP.texDim.x) * mUP.texDim.x;
    //         temp[offset + i * 4] = UPlaneBuffer[offset + id * 4];
    //         temp[offset + i * 4 + 1] = UPlaneBuffer[offset + id * 4 + 1];
    //         temp[offset + i * 4 + 2] = UPlaneBuffer[offset + id * 4 + 2];
    //         temp[offset + i * 4 + 3] = UPlaneBuffer[offset + id * 4 + 3];
    //     }
    // }
    // UPlaneBuffer = temp;

    // Save for cuda
    mUP.featureData = UPlaneBuffer;
    mHP.featureData = HPlaneBuffer;
    mDP.featureData = DPlaneBuffer;

    mTP.featureData = mpTextureSynthesis->getTData();
    mTPInv.featureData = mpTextureSynthesis->getInvTData();
    mTP.texDim = int2(PlaneMetaBuffer[0], PlaneMetaBuffer[1]);
    mTPInv.texDim = int2(mTPInv.featureData.size() / (4 * int(PlaneMetaBuffer[1])), PlaneMetaBuffer[1]);
    // mTPInv.texDim = int2(8192, PlaneMetaBuffer[1]);
    logInfo("[NBTF] T: {}, InvT: {}", mTP.texDim, mTPInv.texDim);

    mUP.featureTex = pDevice->createTexture2D(
        mUP.texDim.x, mUP.texDim.x, ResourceFormat::RGBA32Float, mUP.texDim.y, Resource::kMaxPossible, UPlaneBuffer.data(), bindFlags
    );

    mDP.featureTex =
        pDevice->createTexture2D(mDP.texDim.x, mDP.texDim.x, ResourceFormat::RGBA32Float, mDP.texDim.y, 1, DPlaneBuffer.data(), bindFlags);

    mHP.featureTex =
        pDevice->createTexture2D(mHP.texDim.x, mHP.texDim.x, ResourceFormat::RGBA32Float, mHP.texDim.y, 1, HPlaneBuffer.data(), bindFlags);

    // std::vector<float>().swap(DPlaneBuffer);
    // std::vector<float>().swap(HPlaneBuffer);
    // std::vector<float>().swap(UPlaneBuffer);
}

NBTF::NBTF(ref<Device> pDevice, std::string networkName, bool buildCuda)
{
    mNetworkName = networkName;
    mBuildCuda = buildCuda;
    mHistogram = true;

    mpTextureSynthesis = std::make_unique<TextureSynthesis>();
    loadFeature(pDevice, networkName);

    if (buildCuda)
    {
        mpMLPCuda = std::make_unique<MLPCuda>();
        mpMLPCuda->loadInt8(pDevice, fmt::format("{}/media/neural_materials/networks/Weight_int8_{}.bin", getProjectDirectory(), networkName));
        // mpMLPCuda->loadFP32(pDevice, fmt::format("{}/media/neural_materials/networks/Weight_fp32_{}.bin", getProjectDirectory(), networkName));

        mpMLPCuda->mUTexObj = createCudaTextureArray(mUP.featureData, mUP.texDim.x, mUP.texDim.x, mUP.texDim.y);
        mpMLPCuda->mHTexObj = createCudaTextureArray(mHP.featureData, mHP.texDim.x, mHP.texDim.x, mHP.texDim.y);
        mpMLPCuda->mDTexObj = createCudaTextureArray(mDP.featureData, mDP.texDim.x, mDP.texDim.x, mDP.texDim.y);

        mpMLPCuda->mTTexObj = createCudaTextureArray(mTP.featureData, mTP.texDim.x, mTP.texDim.x, mTP.texDim.y);
        mpMLPCuda->mInvTexObj = createCudaTextureArray(mTPInv.featureData, mTPInv.texDim.x, 1, mTPInv.texDim.y);
        std::vector<float> sample_data = mpTextureSynthesis->getSampleUV();
        mpMLPCuda->mpSampleBuffer = pDevice->createBuffer(
            sample_data.size() * sizeof(float), ResourceBindFlags::Shared, MemoryType::DeviceLocal, sample_data.data()
        );
    }
    // else
    // {
    // mpMLP = std::make_unique<MLP>(pDevice, networkName);
    // }
}

void NBTF::bindShaderData(const ShaderVar& var) const
{
    // if (mBuildCuda)
    //     return;

    // mpMLP->bindShaderData(var["mlp"]);
    if (mHistogram)
        mpTextureSynthesis->bindFeatureData(var["histoFeatureData"]);

    var["uDims"] = mUP.texDim;
    var["hDims"] = mHP.texDim;
    var["dDims"] = mDP.texDim;

    var["uP"].setSrv(mUP.featureTex->getSRV());
    var["hP"].setSrv(mHP.featureTex->getSRV());
    var["dP"].setSrv(mDP.featureTex->getSRV());
    var["inputSize"] = (mDP.texDim.y + mHP.texDim.y + mUP.texDim.y) * 4;
}

NBTF::~NBTF()
{
    if (mpMLPCuda)
    {
        // Clean up CUDA texture objects and arrays
        destroyCudaTextureArray(mpMLPCuda->mUTexObj);
        destroyCudaTextureArray(mpMLPCuda->mHTexObj);
        destroyCudaTextureArray(mpMLPCuda->mDTexObj);
        destroyCudaTextureArray(mpMLPCuda->mTTexObj);
        destroyCudaTextureArray(mpMLPCuda->mInvTexObj);

        mpMLPCuda->mUTexObj = 0;
        mpMLPCuda->mHTexObj = 0;
        mpMLPCuda->mDTexObj = 0;
        mpMLPCuda->mTTexObj = 0;
        mpMLPCuda->mInvTexObj = 0;
    }
}

} // namespace Falcor
