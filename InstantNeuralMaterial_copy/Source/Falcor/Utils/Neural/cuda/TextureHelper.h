#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
cudaTextureObject_t createCudaTextureArray(std::vector<float> data, int width, int height, int depth);
void destroyCudaTextureArray(cudaTextureObject_t texObj);

