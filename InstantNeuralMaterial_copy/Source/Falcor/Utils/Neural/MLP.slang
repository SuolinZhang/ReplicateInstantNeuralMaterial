import Utils.Math.MathHelpers;
import Utils.Neural.ActivationFunc;
import Utils.Math.FormatConversion;
#ifndef MAX_NN_DIM
#define MAX_NN_DIM 8
#endif
struct MLP
{
    int layerNum;
    StructuredBuffer<float4x4> weights;
    StructuredBuffer<float4> bias;
    StructuredBuffer<float> meta;

    Buffer<float> debugWeights;
    Buffer<float> debugBias;

    float4 eval(inout float4 x[MAX_NN_DIM])
    {
        float4 y[MAX_NN_DIM];
        int offset = 0;
        int biasOffset = 0;
        for (int i = 0; i < layerNum; ++i)
        {
            if (i % 2 == 0)
            {
                float h = meta[2 * i + 1];
                float w = meta[2 * i + 2];

                int outNum = floor(w / 4);
                for (int j = 0; j < outNum; ++j)
                {
                    y[j] = float4(0.0f);
                }
                for (int j = 0; j < h / 4; ++j)
                {
                    for (int k = 0; k < w / 4; ++k)
                    {
                        y[k] += mul(weights[outNum * j + k + offset], x[j]);
                    }
                }
                for (int j = 0; j < outNum; ++j)
                {
                    y[j] += bias[j + biasOffset];
                    y[j] = relu(y[j]);
                }
                biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
            else
            {
                float h = meta[2 * i + 1];
                float w = meta[2 * i + 2];

                int outNum = floor(w / 4);
                for (int j = 0; j < outNum; ++j)
                {
                    x[j] = float4(0.0f);
                }
                for (int j = 0; j < h / 4; ++j)
                {
                    for (int k = 0; k < w / 4; ++k)
                    {
                        x[k] += mul(weights[outNum * j + k + offset], y[j]);
                    }
                }
                for (int j = 0; j < outNum; ++j)
                {
                    x[j] += bias[j + biasOffset];
                    x[j] = relu(x[j]);
                }
                biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
        }
        // final layer
        if (layerNum % 2 == 1)
        {
            return relu(y[0]);
        }
        else
        {
            return relu(x[0]);
        }
    }

    float4 eval(inout float inputVal[24])
    {
        float val1[32];
        float val2[32];
        int offset = 0;
        int biasOffset = 0;

        int inNumFirst = 24;
        int inNum = 32;
        int outNum = 32;

        for (int k = 0; k < outNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNumFirst; ++j)
            {
                sum += debugWeights[inNumFirst * k + j + offset] * inputVal[j];
            }
            // val2[k] = sum + debugBias[k + biasOffset];
            val2[k] = relu(sum);
        }
        offset += outNum * inNumFirst;
        biasOffset += outNum;

        for (int k = 0; k < outNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += debugWeights[inNum * k + j + offset] * val2[j];
            }
            // val1[k] = sum + debugBias[k + biasOffset];
            val1[k] = relu(sum);
        }
        offset += outNum * inNum;
        biasOffset += outNum;

        for (int k = 0; k < outNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += debugWeights[inNum * k + j + offset] * val1[j];
            }
            // val2[k] = sum + debugBias[k + biasOffset];
            val2[k] = relu(sum);
        }
        offset += outNum * inNum;
        biasOffset += outNum;

        for (int k = 0; k < 3; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += debugWeights[inNum * k + j + offset] * val2[j];
            }
            // val1[k] = sum + debugBias[k + biasOffset];
            val1[k] = relu(sum);
        }
        return float4(val1[0], val1[1], val1[2], 1.0);
    }

    float4 evalDummy(inout float4 x[MAX_NN_DIM])
    {
        float4 y[MAX_NN_DIM];
        int offset = 0;
        int biasOffset = 0;
        for (int i = 0; i < layerNum; ++i)
        {
            if (i % 2 == 0)
            {
                float h = meta[2 * i + 1] - 8;
                float w = meta[2 * i + 2];

                int outNum = floor(w / 4);
                for (int j = 0; j < outNum; ++j)
                {
                    y[j] = float4(0.0f);
                }
                for (int j = 0; j < h / 4; ++j)
                {
                    for (int k = 0; k < w / 4; ++k)
                    {
                        y[k] += mul(weights[outNum * j + k + offset], x[j]);
                    }
                }
                for (int j = 0; j < outNum; ++j)
                {
                    y[j] += bias[j + biasOffset];
                    y[j] = relu(y[j]);
                }
                biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
            else
            {
                float h = meta[2 * i + 1] - 8;
                float w = meta[2 * i + 2] - 8;

                int outNum = floor(w / 4);
                for (int j = 0; j < outNum; ++j)
                {
                    x[j] = float4(0.0f);
                }
                for (int j = 0; j < h / 4; ++j)
                {
                    for (int k = 0; k < w / 4; ++k)
                    {
                        x[k] += mul(weights[outNum * j + k + offset], y[j]);
                    }
                }
                for (int j = 0; j < outNum; ++j)
                {
                    x[j] += bias[j + biasOffset];
                    x[j] = relu(x[j]);
                }
                biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
        }
        // final layer
        if (layerNum % 2 == 1)
        {
            return relu(y[0]);
        }
        else
        {
            return relu(x[0]);
        }
    }
    float4 evalDummy2(inout float4 x[MAX_NN_DIM])
    {
        float4 y[MAX_NN_DIM];
        int offset = 0;
        int biasOffset = 0;
        for (int i = 0; i < layerNum; ++i)
        {
            if (i % 2 == 0)
            {
                float h = meta[2 * i + 1] - 8;
                float w = meta[2 * i + 2];

                int outNum = floor(w / 4);
                for (int j = 0; j < outNum; ++j)
                {
                    y[j] = float4(0.0f);
                }
                for (int j = 0; j < h / 4; ++j)
                {
                    for (int k = 0; k < w / 4; ++k)
                    {
                        y[k] += mul(weights[outNum * j + k + offset], x[j]);
                    }
                }
                for (int j = 0; j < outNum; ++j)
                {
                    y[j] += bias[j + biasOffset];
                    y[j] = relu(y[j]);
                }
                biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
            else
            {
                float h = meta[2 * i + 1] - 8;
                float w = meta[2 * i + 2] - 8;

                int outNum = floor(w / 4);
                for (int j = 0; j < outNum; ++j)
                {
                    x[j] = float4(0.0f);
                }
                for (int j = 0; j < h / 4; ++j)
                {
                    for (int k = 0; k < w / 4; ++k)
                    {
                        x[k] += mul(weights[outNum * j + k + offset], y[j]);
                    }
                }
                for (int j = 0; j < outNum; ++j)
                {
                    x[j] += bias[j + biasOffset];
                    x[j] = relu(x[j]);
                }
                biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
        }
        // final layer
        if (layerNum % 2 == 1)
        {
            return relu(y[0]);
        }
        else
        {
            return relu(x[0]);
        }
    }
}
