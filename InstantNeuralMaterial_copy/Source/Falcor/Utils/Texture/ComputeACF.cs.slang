
cbuffer PerFrameCB
{
    uint2 gTargetDim;
    float gMean;
}

Buffer<float> gInput;
RWBuffer<float> gACF;



[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gTargetDim))
        return;
    const uint offset = dispatchThreadId.y * gTargetDim.x + dispatchThreadId.x;
    float sum = 0.0f;
    uint i = dispatchThreadId.y;
    uint j = dispatchThreadId.x;

    for (int m = 0; m < gTargetDim.x; m++)
    {
        for (int n = 0; n < gTargetDim.x; n++)
        {
            sum += (gInput[m * gTargetDim.x + n] - gMean) * (gInput[((m + i) % gTargetDim.y) * gTargetDim.x + ((n + j) % gTargetDim.x)] - gMean);
        }
    }
    gACF[offset] = sum / (gTargetDim.x * gTargetDim.y);

}

