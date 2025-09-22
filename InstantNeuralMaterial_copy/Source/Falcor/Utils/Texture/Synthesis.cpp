#include "Synthesis.h"

#include "Core/Error.h"
#include "Core/API/Device.h"
#include "Utils/Neural/IOHelper.h"
namespace Falcor
{
TextureSynthesis::TextureSynthesis()
{
    // std::string hfPath = "D:/textures/ubo/leather11.png";
}
void TextureSynthesis::bindShaderData(const ShaderVar& var) const
{
    var["color"] = mpColor;
}
void TextureSynthesis::readHFData(std::string hfPath, ref<Device> pDevice)
{
    Bitmap::UniqueConstPtr pBitmap = Bitmap::createFromFile(hfPath, true);
    FALCOR_ASSERT(pBitmap);
    logInfo("[Synthesis] Input Image Path: {}", hfPath);
    logInfo("[Synthesis] Input Image Format: {}", to_string(pBitmap->getFormat()));
    logInfo("[Synthesis] Input Image Width:  {}", pBitmap->getWidth());
    logInfo("[Synthesis] Input Image Height: {}", pBitmap->getHeight());

    TextureDataFloat input(pBitmap->getHeight(), pBitmap->getWidth(), 1);
    int bitMapChannels = 1;
    if (pBitmap->getFormat() == ResourceFormat::BGRX8Unorm)
    {
        bitMapChannels = 4;
        for (size_t i = 0; i < pBitmap->getWidth() * pBitmap->getHeight(); i++)
        {
            input.data[i] = pBitmap->getData()[i * bitMapChannels + 0] / 255.0f;
        }
    }

    else if (pBitmap->getFormat() == ResourceFormat::R16Unorm)
    {
        bitMapChannels = 1;
        auto pBitData = reinterpret_cast<const uint16_t*>(pBitmap->getData());
        for (size_t i = 0; i < pBitmap->getWidth() * pBitmap->getHeight(); i++)
        {
            input.data[i] = pBitData[i * bitMapChannels + 0] / 65535.0f;
        }
    }


    else if (pBitmap->getFormat() == ResourceFormat::R8Unorm)
    {
        bitMapChannels = 1;
        auto pBitData = reinterpret_cast<const uint8_t*>(pBitmap->getData());
        for (size_t i = 0; i < pBitmap->getWidth() * pBitmap->getHeight(); i++)
        {
            input.data[i] = pBitData[i * bitMapChannels + 0] / 255.0f;
        }
    }

    TextureDataFloat Tinput;
    TextureDataFloat lut;
    logInfo("[Synthesis] Precomputing Gaussian T and Inv.");
    Precomputations(input, Tinput, lut);
    logInfo("[Synthesis] Precomputation done!");
    // TODO generate max mipmap
    mpHFT = pDevice->createTexture2D(
        Tinput.width,
        Tinput.height,
        ResourceFormat::R32Float,
        1,
        Resource::kMaxPossible,
        Tinput.data.data(),
        ResourceBindFlags::ShaderResource
    );

    mpHFInvT =
        pDevice->createTexture2D(lut.width, lut.height, ResourceFormat::R32Float, 1, 1, lut.data.data(), ResourceBindFlags::ShaderResource);
}

void TextureSynthesis::precomputeFeatureData(std::vector<float> data, uint2 dataDim, ref<Device> pDevice)
{
    logInfo("[Synthesis] Input Feature Dim: {}", dataDim);
    logInfo("[Synthesis] Input Feature Size:  {}", data.size());

    // std::vector<float> temp;
    // temp.resize(data.size());
    // for (uint j = 0; j < dataDim.y; j++)
    //{
    //     uint offset = j * dataDim.x * dataDim.x * 4;
    //     for (uint i = 0; i < data.size() / (4 * dataDim.y); i++)
    //     {
    //         uint id = i / dataDim.x + (i % dataDim.x) * dataDim.x;
    //         temp[offset + i * 4] = data[offset + id * 4];
    //         temp[offset + i * 4 + 1] = data[offset + id * 4 + 1];
    //         temp[offset + i * 4 + 2] = data[offset + id * 4 + 2];
    //         temp[offset + i * 4 + 3] = data[offset + id * 4 + 3];
    //     }
    // }
    // data = temp;

    TData.resize(data.size());
    invTData.resize(LUT_WIDTH * 4 * dataDim.y);
    TextureDataFloat acf = TextureDataFloat(dataDim.x, dataDim.x, 1);
    logInfo("[Synthesis] Precomputing Feature Gaussian T and Inv.");
    for (uint i = 0; i < dataDim.y; i++)
    {
        uint offset = i * dataDim.x * dataDim.x * 4;
        uint singleDataSize = dataDim.x * dataDim.x * 4;
        TextureDataFloat input(dataDim.x, dataDim.x, 4);
        std::copy(data.begin() + offset, data.begin() + offset + singleDataSize, input.data.begin());

        TextureDataFloat Tinput;
        TextureDataFloat lut;
        Precomputations(input, Tinput, lut);
        std::copy(Tinput.data.begin(), Tinput.data.end(), TData.begin() + offset);
        std::copy(lut.data.begin(), lut.data.end(), invTData.begin() + LUT_WIDTH * i * 4);

        if (i == 0)
        {
            mpComputeACFPass = ComputePass::create(pDevice, "Utils/Texture/ComputeACF.cs.slang", "csMain");
            mpACFBuffer = pDevice->createBuffer(
                dataDim.x * dataDim.x * 1 * sizeof(float),
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr
            );

            std::vector<float> acfInput(dataDim.x * dataDim.x);
            for (uint m = 0; m < dataDim.x; m++)
            {
                for (uint n = 0; n < dataDim.x; n++)
                {
                    acfInput[m * dataDim.x + n] = input.GetPixel(m, n, 0);
                }
            }

            mpACFInputBuffer = pDevice->createBuffer(
                dataDim.x * dataDim.x * 1 * sizeof(float),
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                acfInput.data()
            );
            float mean = calculateMean(input);

            auto vars = mpComputeACFPass->getRootVar();
            vars["PerFrameCB"]["gTargetDim"] = uint2(dataDim.x, dataDim.x);
            vars["PerFrameCB"]["gMean"] = mean;
            vars["gInput"] = mpACFInputBuffer;
            vars["gACF"] = mpACFBuffer;
            mpComputeACFPass->execute(pDevice->getRenderContext(), dataDim.x, dataDim.x);

            // calculateAutocovariance(input, acf, acfWeight);
            // writeToBinaryFile(acfWeight, "D:/acf_TILE2.bin");
        }
    }
    acfWeight.resize(dataDim.x * dataDim.x );
    acfPDF.resize(dataDim.x * dataDim.x );
    acfPDFImg.resize(dataDim.x * dataDim.x*3);
    mpACFBuffer->getBlob(acfWeight.data(), 0, dataDim.x * dataDim.x * 1 * sizeof(float));
    logInfo("[Synthesis] Precomputing ACF done! {} {}", acfWeight.size(), acfWeight[1]);

    logInfo("[Synthesis] Precomputation done!");
    sample_uv_list.resize(2048 * 2);
    updateMap(dataDim.x, pDevice);

    // // TODO generate max mipmap
    mpFeatureT = pDevice->createTexture2D(
        dataDim.x,  dataDim.x, ResourceFormat::RGBA32Float, dataDim.y, 1, TData.data(), ResourceBindFlags::ShaderResource
    );
    mpFeatureInvT =
        pDevice->createTexture2D(LUT_WIDTH, 1, ResourceFormat::RGBA32Float, dataDim.y, 1, invTData.data(),
        ResourceBindFlags::ShaderResource);
    // mpACF =
    //     pDevice->createTexture2D(dataDim.x, dataDim.x, ResourceFormat::R32Float, 1, 1, acf.data.data(),
    //     ResourceBindFlags::ShaderResource);
}

void TextureSynthesis::updateMap(uint dim, ref<Device> pDevice)
{

    updateSample(acfWeight, acfPDF, sample_uv_list, dim);
    logInfo("[Synthesis] Precomputing ACF done!updateMap {} {}", sample_uv_list.size(), acfPDF.size());
    mpMap = pDevice->createTypedBuffer(
        ResourceFormat::R32Float, sample_uv_list.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, sample_uv_list.data()
    );
    for (int i = 0; i < acfPDF.size(); i++)
    {
        acfPDFImg[i * 3] = acfPDF[i];
        acfPDFImg[i * 3 + 1] = acfPDF[i];
        acfPDFImg[i * 3 + 2] = acfPDF[i];
    }


    mpACF = pDevice->createTexture2D(dim, dim, ResourceFormat::RGB32Float, 1, 1, acfPDFImg.data(), ResourceBindFlags::ShaderResource);
}

void TextureSynthesis::updateMap(uint dim, ref<Device> pDevice, float2* ctrl_point, ACFCurve curve)
{
    if (curve == ACFCurve::BEZIER)
        updateSample(acfWeight, acfPDF, sample_uv_list, dim, ctrl_point);
    else
    updateSample(acfWeight, acfPDF, sample_uv_list, dim, curve);


    mpMap = pDevice->createTypedBuffer(
        ResourceFormat::R32Float, sample_uv_list.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, sample_uv_list.data()
    );
    for (int i = 0; i < acfPDF.size(); i++)
    {
        acfPDFImg[i * 3] = acfPDF[i];
        acfPDFImg[i * 3 + 1] = acfPDF[i];
        acfPDFImg[i * 3 + 2] = acfPDF[i];
    }
    mpACF = pDevice->createTexture2D(dim, dim, ResourceFormat::RGB32Float, 1, 1, acfPDFImg.data(), ResourceBindFlags::ShaderResource);
}

void TextureSynthesis::bindHFData(const ShaderVar& var)
{
    var["tex"] = mpHFT;
    var["invTex"] = mpHFInvT;
}
void TextureSynthesis::bindMap(const ShaderVar& var)
{
    var["sampleMap"] = mpMap;
    // std::vector<float> test;
    // test.resize(2);
    // mpMap->getBlob(test.data(), 0, 8);
    // std::cout << sample_uv_list.size() << " " << sample_uv_list[0] << " " << sample_uv_list[1] << std::endl;
    // std::cout << test[0] << " " << test[1] << std::endl;
}
void TextureSynthesis::bindFeatureData(const ShaderVar& var)
{
    var["tex"] = mpFeatureT;
    var["invTex"] = mpFeatureInvT;
    var["acf"] = mpACF;
}
} // namespace Falcor
