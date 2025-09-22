import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;


// pack inputs into one uint4 using float16
inline uint4 packInput(float3 wi, float3 wo, float2 uv, float2 ddx, float2 ddy)
{
    uint4 ret;
    float2 h, d, ddxy;
    dirToRusinkiewicz(wi, wo, h, d);

    // Use uniform sampling (same as training) instead of Shirley disk mapping
    // Training uses simple uniform sampling: u = random(), v = random()
    // So we keep the Rusinkiewicz angles as-is without disk mapping
    // h and d remain as (θ_h, φ_h) and (θ_d, φ_d) in radians, normalized to [0,1]
    h = h * float2(0.3183098861837907, 0.15915494309189535);  // 1/π, 1/(2π) - θ_h ∈ [0,π] → [0,1], φ_h ∈ [0,2π] → [0,1]
    d = d * float2(0.3183098861837907, 0.15915494309189535);  // 1/π, 1/(2π) - θ_d ∈ [0,π] → [0,1], φ_d ∈ [0,2π] → [0,1]


    ddxy.x = abs(ddx.x) > abs(ddx.y) ? ddx.x : ddx.y;
    ddxy.y = abs(ddy.x) > abs(ddy.y) ? ddy.x : ddy.y;
    ret.x = packUnorm2x16(h);
    ret.y = packUnorm2x16(d);
    ret.z = packUnorm2x16(uv);
    ret.w = packSnorm2x16(ddxy);
    return ret;
}

inline uint4 packInputSyn(float3 wi, float3 wo, float2 uv1, float2 uv2)
{
    uint4 ret;
    float2 h, d;
    dirToRusinkiewicz(wi, wo, h, d);

    // Use uniform sampling (same as training) instead of Shirley disk mapping
    // Training uses simple uniform sampling: u = random(), v = random()
    // So we keep the Rusinkiewicz angles as-is without disk mapping
    // h and d remain as (θ_h, φ_h) and (θ_d, φ_d) in radians, normalized to [0,1]
    h = h * float2(0.3183098861837907, 0.15915494309189535);  // 1/π, 1/(2π) - θ_h ∈ [0,π] → [0,1], φ_h ∈ [0,2π] → [0,1]
    d = d * float2(0.3183098861837907, 0.15915494309189535);  // 1/π, 1/(2π) - θ_d ∈ [0,π] → [0,1], φ_d ∈ [0,2π] → [0,1]


    ret.x = packUnorm2x16(h);
    ret.y = packUnorm2x16(d);
    ret.z = packUnorm2x16(uv1);
    ret.w = packSnorm2x16(uv2);
    return ret;
}

// unpack input from the packed uint4
inline void unpackInput(uint4 packed, out float2 h, out float2 d, out float2 uv, out float2 ddx, out float2 ddy)
{
    float2 ddxy;
    h =     unpackUnorm2x16(packed.x);
    d =     unpackUnorm2x16(packed.y);
    uv =    unpackUnorm2x16(packed.z);
    ddxy =  unpackSnorm2x16(packed.w);
    ddx = ddxy.xx;
    ddy = ddxy.yy;
}
