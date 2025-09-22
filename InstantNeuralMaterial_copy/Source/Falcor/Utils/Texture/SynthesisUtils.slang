#include "Utils/Math/MathConstants.slangh"
import Rendering.Materials.TexLODHelpers;
import Scene.Material.TextureSampler;

inline float inversesqrt(float x) { return 1.0 / sqrt(x); }
float2 MakeCenST(int2 Vertex)
{
    float2x2 invSkewMat = float2x2(1.0, 0.5, 0.0, 1.0 / 1.15470054);
    return mul(invSkewMat, Vertex) / (2 * sqrt(3));
}
float2x2 LoadRot2x2(int2 idx, float rotStrength)
{
    float angle = abs(idx.x * idx.y) + abs(idx.x + idx.y) + M_PI;
    // Remap to +/-pi.
    angle = fmod(angle, 2 * M_PI);
    if (angle < 0)
        angle += 2 * M_PI;
    if (angle > M_PI)
        angle -= 2 * M_PI;
    angle *= rotStrength;
    float cs = cos(angle), si = sin(angle);
    return float2x2(cs, -si, si, cs);
}
float3 ProduceHexWeights(float3 W, int2 vertex1, int2 vertex2, int2 vertex3)
{
    float3 res = 0.0;
    int v1 = (vertex1.x - vertex1.y) % 3;
    if (v1 < 0)
        v1 += 3;
    int vh = v1 < 2 ? (v1 + 1) : 0;
    int vl = v1 > 0 ? (v1 - 1) : 2;
    int v2 = vertex1.x < vertex3.x ? vl : vh;
    int v3 = vertex1.x < vertex3.x ? vh : vl;
    res.x = v3 == 0 ? W.z : (v2 == 0 ? W.y : W.x);
    res.y = v3 == 1 ? W.z : (v2 == 1 ? W.y : W.x);
    res.z = v3 == 2 ? W.z : (v2 == 2 ? W.y : W.x);
    return res;
}
float3 Gain3(float3 x, float r)
{
    // Increase contrast when r > 0.5 and
    // reduce contrast if less.
    float k = log(1 - r) / log(0.5);
    float3 s = 2 * step(0.5, x);
    float3 m = 2 * (1 - s);
    float3 res = 0.5 * s + 0.25 * m * pow(max(0.0, s + x * m), k);
    return res.xyz / (res.x + res.y + res.z);
}
// Output: weights associated with each hex tile and integer centers
void TriangleGrid(out float w1, out float w2, out float w3, out int2 vertex1, out int2 vertex2, out int2 vertex3, float2 st)
{
    // Scaling of the input
    st *= 2 * sqrt(3);
    // Skew input space into simplex triangle grid.
    const float2x2 gridToSkewedGrid = float2x2(1.0, 0.0, -0.57735027, 1.15470054);
    float2 skewedCoord = mul(gridToSkewedGrid, st);
    int2 baseId = int2(floor(skewedCoord));
    float3 temp = float3(frac(skewedCoord), 0);
    temp.z = 1.0 - temp.x - temp.y;

    if (temp.z > 0.0)
    {
        w1 = temp.z;
        w2 = temp.y;
        w3 = temp.x;
        vertex1 = baseId;
        vertex2 = baseId + int2(0, 1);
        vertex3 = baseId + int2(1, 0);
    }
    else
    {

        w1 = -temp.z;
        w2 = 1.0 - temp.y;
        w3 = 1.0 - temp.x;
        vertex1 = baseId + int2(1, 1);
        vertex2 = baseId + int2(1, 0);
        vertex3 = baseId + int2(0, 1);
    }
}
float2 hash(float2 p)
{
    //float2 r = mul(float2x2(127.1, 311.7, 269.5, 183.3), p);
    float2 r = float2(127.1f * p.x + 311.7f * p.y, 269.5f * p.x + 183.3f * p.y);
    float2 temp = sin(r) * 43758.5453f;
    return (temp - floor(temp));
}

#define PI 3.14159265f

float B0cos(float2 uv)
{
    float cosu = sin(uv.x * PI);
    float cosv = sin(uv.y * PI);
    return pow(cosu * cosv * cosu * cosv, 0.5f);
}

float B1cos(float2 uv)
{
    uv += float2(0.5f, 0.5f);
    float cosu = sin(uv.x * PI);
    float cosv = sin(uv.y * PI);
    return pow(cosu * cosv * cosu * cosv, 0.5f);
}

float BSingularity(float2 uv)
{
    uv -= float2(0.5f, 0.5f);
    uv *= 1.41421356237f;
    const float a = 0.78539816f; // PI / 4.0;
    float cosA = cos(a);
    float sinA = sin(a);
    float4 rotateA = float4(cosA, sinA, -sinA, cosA);
    float2 V = float2(rotateA.x * uv.x + rotateA.y * uv.y, rotateA.z * uv.x + rotateA.w * uv.y);
    float cosu = sin(V.x * PI);
    float cosv = sin(V.y * PI);
    return 0.02f * cosu * cosv * cosu * cosv;
}

float rnd21(float2 p)
{
    float temp = sin(12.9898f * p.x + 78.233f * p.y) * 43758.5453f;
    return (temp - floor(temp));
}
