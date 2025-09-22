import SynthesisUtils;

struct HistogramBlendingHFData
{
    Texture2D<float> tex;
    Texture2D<float> invTex;

    Buffer<float> sampleMap;
};

struct HistogramBlendingFeatureData
{
    Texture2DArray<float4> tex;
    Texture2DArray<float4> invTex;
    Texture2D<float> acf;
};

float HistorgramBlending(
    HistogramBlendingHFData data,
    SamplerState sampler,
    float2 ddx,
    float2 ddy,
    float2 uv
)
{
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);

    float2 st1 = uv + hash(vertex1);
    float2 st2 = uv + hash(vertex2);
    float2 st3 = uv + hash(vertex3);

    float g1 = data.tex.SampleGrad(sampler, frac(st1), ddx, ddy);
    float g2 = data.tex.SampleGrad(sampler, frac(st2), ddx, ddy);
    float g3 = data.tex.SampleGrad(sampler, frac(st3), ddx, ddy);

    // float G = w1 * g1 + w2 * g2 + w3 * g3;
    float G = max(max(w1 * g1, w2 * g2), w3 * g3);
    G -= 0.5f;
    G *= inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
    G += 0.5f;
    G = abs(G);
    G = clamp(G, 0.001, 0.999);
    float res;
    res = data.invTex.SampleLevel(sampler, float2(G, 0.0f), 0.f);
    return res;
}

float4 HistorgramBlending(
    HistogramBlendingFeatureData data,
    SamplerState sampler,
    float2 uv,
    uint layer,
    float2 ddx,
    float2 ddy
)
{
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);

    float2 st1 = uv + hash(vertex1);
    float2 st2 = uv + hash(vertex2);
    float2 st3 = uv + hash(vertex3);
    float4 g1 = data.tex.SampleLevel(sampler, float3(st1, layer), 0);
    float4 g2 = data.tex.SampleLevel(sampler, float3(st2, layer), 0);
    float4 g3 = data.tex.SampleLevel(sampler, float3(st3, layer), 0);

    float4 G = w1 * g1 + w2 * g2 + w3 * g3;
    G -= 0.5f;
    G *= inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
    G += 0.5f;
    G = abs(G);
    G = clamp(G, 0.0001, 0.999);
    float4 res;
    res.x = data.invTex.SampleLevel(sampler, float3(G.x, 0.0f, layer), 0.f).x;
    res.y = data.invTex.SampleLevel(sampler, float3(G.y, 0.0f, layer), 0.f).y;
    res.z = data.invTex.SampleLevel(sampler, float3(G.z, 0.0f, layer), 0.f).z;
    res.w = data.invTex.SampleLevel(sampler, float3(G.w, 0.0f, layer), 0.f).w;
    return res;
}




float AutocovarianceBlendingDual(
    HistogramBlendingHFData data,
    SamplerState sampler,
    float2 ddx,
    float2 ddy,
    float2 uv
)
{
    float G;
	float3 b;
    float bSum;

    b = float3(B0cos(uv),
               B1cos(uv),
               BSingularity(uv));
    bSum = b.x + b.y + b.z;
    b /= bSum;
    float2 t0 = floor(uv) * 2.0f;
    float2 t1 = floor(uv + float2(0.5f, 0.5f)) * 2.0f + 1.0f;

    int id0 = floor(rnd21(t0) * 2048);
    int id1 = floor(rnd21(t1) * 2048);

    G = (data.tex.SampleGrad(sampler, uv - float2(data.sampleMap[2 * id0], data.sampleMap[2 * id0 + 1]), ddx, ddy) - 0.5f) * b.x
      + (data.tex.SampleGrad(sampler, uv - float2(data.sampleMap[2 * id1], data.sampleMap[2 * id1 + 1]), ddx, ddy) - 0.5f) * b.y;

    G = 0.5f + G / length(b);
    G = clamp(G, 0.0001, 0.999);

    float res;
    res = data.invTex.SampleLevel(sampler, float2(G, 0.0f), 0.f);
    return res;
}

float AutocovarianceBlendingDual(
    HistogramBlendingHFData data,
    SamplerState sampler,
    float lod,
    float2 uv
)
{
    float G;
    float3 b;
    float bSum;

    b = float3(B0cos(uv),
               B1cos(uv),
               BSingularity(uv));
    bSum = b.x + b.y + b.z;
    b /= bSum;
    float2 t0 = floor(uv) * 2.0f;
    float2 t1 = floor(uv + float2(0.5f, 0.5f)) * 2.0f + 1.0f;

    int id0 = floor(rnd21(t0) * 2048);
    int id1 = floor(rnd21(t1) * 2048);

    G = (data.tex.SampleLevel(sampler, uv - float2(data.sampleMap[2 * id0], data.sampleMap[2 * id0 + 1]), lod) - 0.5f) * b.x
      + (data.tex.SampleLevel(sampler, uv - float2(data.sampleMap[2 * id1], data.sampleMap[2 * id1 + 1]), lod) - 0.5f) * b.y;

    G = 0.5f + G / length(b);
    G = clamp(G, 0.0001, 0.999);

    float res;
    res = data.invTex.SampleLevel(sampler, float2(G, 0.0f), 0.f);
    return res;
}

float AutocovarianceBlendingTriangle(
    HistogramBlendingHFData data,
    SamplerState sampler,
    float2 ddx,
    float2 ddy,
    float2 uv
)
{
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);

    int id1 = floor(rnd21(vertex1) * 2048);
    int id2 = floor(rnd21(vertex2) * 2048);
    int id3 = floor(rnd21(vertex3) * 2048);
    float g1 = data.tex.SampleGrad(sampler, uv - float2(data.sampleMap[2 * id1], data.sampleMap[2 * id1 + 1]), ddx, ddy);
    float g2 = data.tex.SampleGrad(sampler, uv - float2(data.sampleMap[2 * id2], data.sampleMap[2 * id2 + 1]), ddx, ddy);
    float g3 = data.tex.SampleGrad(sampler, uv - float2(data.sampleMap[2 * id3], data.sampleMap[2 * id3 + 1]), ddx, ddy);

    float G = w1 * g1 + w2 * g2 + w3 * g3;
    G -= 0.5f;
    G *= inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
    G += 0.5f;
    G = abs(G);
    G = clamp(G, 0.0001, 0.999);

    float res;
    res = data.invTex.SampleLevel(sampler, float2(G, 0.0f), 0.f);
    return res;
}


float4 getHashUV(HistogramBlendingHFData data, float2 uv)
{

    float2 t0 = floor(uv) * 2.0f;
    float2 t1 = floor(uv + float2(0.5f, 0.5f)) * 2.0f + 1.0f;

    int id0 = floor(rnd21(t0) * 2048);
    int id1 = floor(rnd21(t1) * 2048);

    float2 uv_hash0 = uv - float2(data.sampleMap[2 * id0], data.sampleMap[2 * id0 + 1]);
    float2 uv_hash1 = uv - float2(data.sampleMap[2 * id1], data.sampleMap[2 * id1 + 1]);

    return float4(uv_hash0.x ,uv_hash0.y, uv_hash1.x, uv_hash1.y);
}
