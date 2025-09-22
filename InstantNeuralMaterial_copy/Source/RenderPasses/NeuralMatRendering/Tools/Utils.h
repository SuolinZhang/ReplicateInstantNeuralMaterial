
#include "Falcor.h"
namespace Falcor
{
// create a texture if it does not exist, or recreate it if the size is different
void createTex(ref<Texture>& tex, ref<Device> device, Falcor::uint2 targetDim, bool buildCuda = false, bool isUint = false);
// create a buffer if it does not exist, or recreate it if the size is different
void createBuffer(ref<Buffer>& buf, ref<Device> device, Falcor::uint2 targetDim, uint itemSize = 4);
// generate max mip for a texture
void generateMaxMip(RenderContext* pRenderContext, ref<Texture> pTex);
// curve related methods
float getPoint(void* data, int32_t index);
float getPointX6(void* data, int32_t index);
float getPointX(void* data, int32_t index);

}
