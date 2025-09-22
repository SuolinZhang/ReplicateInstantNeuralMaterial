
#include "Utils.h"
namespace Falcor
{
// create a texture if it does not exist, or recreate it if the size is different
void createTex(ref<Texture>& tex, ref<Device> device, Falcor::uint2 targetDim, bool buildCuda, bool isUint)
{
    ResourceBindFlags flag = ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess;
    if (buildCuda)
        flag |= ResourceBindFlags::Shared;

    if (tex.get() == nullptr)
    {
        if (isUint)
            tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Uint, 1, 1, nullptr, flag);
        else
            tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, flag);
    }
    else
    {
        if (tex.get()->getWidth() != targetDim.x || tex.get()->getHeight() != targetDim.y)
        {
            logInfo("Recreating texture");
            if (isUint)
                tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Uint, 1, 1, nullptr, flag);
            else
                tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, flag);
        }
    }
};
// create a buffer if it does not exist, or recreate it if the size is different
void createBuffer(ref<Buffer>& buf, ref<Device> device, Falcor::uint2 targetDim, uint itemSize )
{
    if (buf.get() == nullptr)
    {
        buf = device->createBuffer(
            targetDim.x * targetDim.y * itemSize * sizeof(float),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr
        );
    }
    else
    {
        if (buf.get()->getElementCount() != targetDim.x * targetDim.y * itemSize * sizeof(float))
        {
            logInfo("Recreating Buffer");
            buf = device->createBuffer(
                targetDim.x * targetDim.y * itemSize * sizeof(float),
                ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr
            );
        }
    }
};

// generate max mip for a texture
void generateMaxMip(RenderContext* pRenderContext, ref<Texture> pTex)
{
    for (uint32_t m = 0; m < pTex->getMipCount() - 1; m++)
    {
        auto srv = pTex->getSRV(m, 1, 0, 1);
        auto rtv = pTex->getRTV(m + 1, 0, 1);
        // only the first channel is used
        const TextureReductionMode redModes[] = {
            TextureReductionMode::Max,
            TextureReductionMode::Min,
            TextureReductionMode::Max,
            TextureReductionMode::Standard,
        };
        const Falcor::float4 componentsTransform[] = {
            Falcor::float4(1.0f, 0.0f, 0.0f, 0.0f),
            Falcor::float4(1.0f, 0.0f, 0.0f, 0.0f),
            Falcor::float4(1.0f, 0.0f, 0.0f, 0.0f),
            Falcor::float4(1.0f, 0.0f, 0.0f, 0.0f),
        };
        pRenderContext->blit(
            srv, rtv, RenderContext::kMaxRect, RenderContext::kMaxRect, TextureFilteringMode::Linear, redModes, componentsTransform
        );
    }
}

// curve related methods
float getPoint(void* data, int32_t index)
{
    return ((float*)data)[index];
}

float getPointX6(void* data, int32_t index)
{
    float val =  (index/40.0f);
    return val * val * val * val * val * val;
}
float getPointX(void* data, int32_t index)
{
    float val =  (index/40.0f);
    return val;
}

}
