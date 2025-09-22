#include "TextureHelper.h"
#include <iostream>
cudaTextureObject_t createCudaTextureArray(std::vector<float> data, int width, int height, int layers)
{

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    cudaExtent extent = make_cudaExtent(width, height, layers);
    cudaArray_t cuArray;
    cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArrayLayered);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)data.data(), width * sizeof(float4), width, height);
    copyParams.dstArray = cuArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[2] = cudaAddressModeWrap;
    // texDesc.filterMode = cudaFilterModePoint;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    return texObj;
}


cudaTextureObject_t createCuda2DTexture(std::vector<float> data, int width, int height)
{

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    std::cout <<"[cudaTextureHelper] "<< "width: " << width << " height: " << height <<  std::endl;
    std::cout <<"[cudaTextureHelper] "<< "data size: " << data.size() << std::endl;

    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    cudaMemcpy2DToArray(cuArray, 0, 0, data.data(), width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    return texObj;
}

void destroyCudaTextureArray(cudaTextureObject_t texObj)
{
    if (texObj)
    {
        cudaResourceDesc resDesc;
        cudaGetTextureObjectResourceDesc(&resDesc, texObj);

        if (resDesc.resType == cudaResourceTypeArray && resDesc.res.array.array)
        {
            cudaFreeArray(resDesc.res.array.array);
        }

        cudaDestroyTextureObject(texObj);
    }
}

