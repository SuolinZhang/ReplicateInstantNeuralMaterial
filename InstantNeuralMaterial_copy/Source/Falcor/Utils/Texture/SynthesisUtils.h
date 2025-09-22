#pragma once
/*****************************************************************************/
/*****************************************************************************/
/********************************* Includes **********************************/
/*****************************************************************************/
/*****************************************************************************/
#include "external/jacobi.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
//using namespace std;

/*****************************************************************************/
/*****************************************************************************/
/******************************** Parameters *********************************/
/*****************************************************************************/
/*****************************************************************************/

#define USE_DXT_COMPRESSION false // Use DXT1 (true) or GL_RGB8 (false) (Section 1.6)
#define GAUSSIAN_AVERAGE 0.5f     // Expectation of the Gaussian distribution
#define GAUSSIAN_STD 0.16666f     // Std of the Gaussian distribution
#define LUT_WIDTH 8192         // Size of the look-up table
namespace Falcor
{
enum class ACFCurve : uint32_t
{
    X,
    X2,
    X3,
    // X5,
    X6,
    // INV_X6,
    // INV_X3,
    // INV_X2,
    BEZIER
};

FALCOR_ENUM_INFO(
    ACFCurve,
    {{ACFCurve::X, "x"},
     {ACFCurve::X2, "x^2"},
     {ACFCurve::X3, "x^3"},
    //  {ACFCurve::X5, "x^5"},
     {ACFCurve::X6, "x^6"},
    //  {ACFCurve::INV_X6, "invx^6"},
    //  {ACFCurve::INV_X3, "invx^3"},
    //  {ACFCurve::INV_X2, "invx^2"},
     {ACFCurve::BEZIER, "Bezier Curve"}}
);
FALCOR_ENUM_REGISTER(ACFCurve);


/**
/*****************************************************************************/
/*****************************************************************************/
/*************************** Types and Structures ****************************/
/*****************************************************************************/
/*****************************************************************************/

struct TextureDataFloat
{
    inline TextureDataFloat() : data(), width(0), height(0), channels(0) {}
    inline TextureDataFloat(const int w, const int h, const int c) : data(w * h * c), width(w), height(h), channels(c) {}

    inline float GetPixel(int w, int h, int c) { return data[h * width * channels + w * channels + c]; }

    inline float3 GetColorAt(int w, int h)
    {
        return float3(
            data[h * width * channels + w * channels + 0],
            data[h * width * channels + w * channels + 1],
            data[h * width * channels + w * channels + 2]
        );
    }

    inline void SetPixel(int w, int h, int c, float value) { data[h * width * channels + w * channels + c] = value; }

    inline void SetColorAt(int w, int h, float3 value)
    {
        data[h * width * channels + w * channels + 0] = value.x;
        data[h * width * channels + w * channels + 1] = value.y;
        data[h * width * channels + w * channels + 2] = value.z;
    }

    inline void VerticalReversePNG()
    {
        uint8_t* data_png = (uint8_t*)data.data();
        for (int h = 0; h < height / 2; h++)
        {
            for (int w = 0; w < width; w++)
            {
                uint8_t x = data_png[h * width * channels + w * channels + 0];
                uint8_t y = data_png[h * width * channels + w * channels + 1];
                uint8_t z = data_png[h * width * channels + w * channels + 2];
                data_png[h * width * channels + w * channels + 0] = data_png[(height - 1 - h) * width * channels + w * channels + 0];
                data_png[h * width * channels + w * channels + 1] = data_png[(height - 1 - h) * width * channels + w * channels + 1];
                data_png[h * width * channels + w * channels + 2] = data_png[(height - 1 - h) * width * channels + w * channels + 2];
                data_png[(height - 1 - h) * width * channels + w * channels + 0] = x;
                data_png[(height - 1 - h) * width * channels + w * channels + 1] = y;
                data_png[(height - 1 - h) * width * channels + w * channels + 2] = z;
            }
        }
    }
    ~TextureDataFloat(){

    }
    std::vector<float> data;

    int width;
    int height;
    int channels;
};

struct PixelSortStruct
{
    int x;
    int y;
    float value;

    inline bool operator<(const PixelSortStruct& other) const { return (value < other.value); }
};

/*****************************************************************************/
/*****************************************************************************/
/**************** Section 1.3.1 Target Gaussian distribution *****************/
/*****************************************************************************/
/*****************************************************************************/
float Erf(float x);
float ErfInv(float x);
float CDF(float x, float mu, float sigma);
float invCDF(float U, float mu, float sigma);

/*****************************************************************************/
/*****************************************************************************/
/************** NEW: REMOVING THE SPATIALLY-VARYING MEAN *********************/
/*****************************************************************************/
/*****************************************************************************/

void PreRemoveMean(TextureDataFloat& input, TextureDataFloat& inputMeanRemoved, TextureDataFloat& mean, int channel);

/*****************************************************************************/
/*****************************************************************************/
/**** Section 1.3.2 Applying the histogram transformation T on the input *****/
/*****************************************************************************/
/*****************************************************************************/

void ComputeTinput(TextureDataFloat& input, TextureDataFloat& T_input, int channel);

/*****************************************************************************/
/*****************************************************************************/
/*  Section 1.3.3 Precomputing the inverse histogram transformation T^{-1}   */
/*****************************************************************************/
/*****************************************************************************/

void ComputeinvT(TextureDataFloat& input, TextureDataFloat& Tinv, int channel);
/*****************************************************************************/
/*****************************************************************************/
/******** Section 1.4 Improvement: using a decorrelated color space **********/
/*****************************************************************************/
/*****************************************************************************/

// Compute the eigen vectors of the histogram of the input
void ComputeEigenVectors(TextureDataFloat& input, float3 eigenVectors[3]);

// Main function of Section 1.4
void DecorrelateColorSpace(
    TextureDataFloat& input,              // input: example image
    TextureDataFloat& input_decorrelated, // output: decorrelated input
    float3& colorSpaceVector1,              // output: color space vector1
    float3& colorSpaceVector2,              // output: color space vector2
    float3& colorSpaceVector3,              // output: color space vector3
    float3& colorSpaceOrigin
); // output: color space origin

/*****************************************************************************/
/*****************************************************************************/
/* ===== Section 1.5 Improvement: prefiltering the look-up table =========== */
/*****************************************************************************/
/*****************************************************************************/

// Compute average subpixel variance at a given LOD
float ComputeLODAverageSubpixelVariance(TextureDataFloat& image, int LOD, int channel);

// Filter LUT by sampling a Gaussian N(mu, std\B2)
float FilterLUTValueAtx(TextureDataFloat& LUT, float x, float std, int channel);

// Main function of section 1.5
void PrefilterLUT(TextureDataFloat& image_T_Input, TextureDataFloat& LUT_Tinv, int channel);


float calculateMean( TextureDataFloat& image);
void calculateAutocovariance(TextureDataFloat& image, TextureDataFloat& acf, std::vector<float>& acf_weight);
void updateSample(std::vector<float>& acf_weight, std::vector<float>& acf_pdf, std::vector<float>& sample_uv_list, uint dim, ACFCurve curve);
void updateSample(std::vector<float>& acf_weight, std::vector<float>& acf_pdf, std::vector<float>& sample_uv_list, uint dim);
void updateSample(
    std::vector<float>& acf_weight,
    std::vector<float>& acf_pdf,
    std::vector<float>& sample_uv_list,
    uint dim,
    float2* ctrl_point
);


/*********************************************************************/
/*********************************************************************/
/*************************** Main Function ***************************/
/*********************************************************************/
/*********************************************************************/

void Precomputations(
    TextureDataFloat& input,  // input: example image
    TextureDataFloat& Tinput, // output: T(input) image
    TextureDataFloat& Tinv   // output: T^{-1} look-up table
);



}
