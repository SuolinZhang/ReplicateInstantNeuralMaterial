// g++ -O2 -std=c++17 -mavx -mf16c btf_dump_height.cpp -o btf_dump_height
// Usage: ./btf_dump_height material_resampled.btf height.pgm [--normalize] [--invert] [--flipY]

#define BTF_IMPLEMENTATION
#include "btf.hh"            // your Free BTF Library header/impl in the same folder
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>

static bool save_pgm16_be(const std::string& path, uint16_t W, uint16_t H, const uint16_t* data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P5\n" << W << " " << H << "\n65535\n";
    // PGM expects big-endian bytes
    for (size_t i = 0, n = size_t(W) * H; i < n; ++i) {
        uint16_t v = data[i];
        unsigned char hi = (unsigned char)(v >> 8);
        unsigned char lo = (unsigned char)(v & 0xFF);
        f.put(hi); f.put(lo);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s input_resampled.btf out.{pgm|exr} [--normalize] [--invert] [--flipY]\n", argv[0]);
        return 1;
    }
    const std::string in = argv[1];
    const std::string out = argv[2];
    bool normalize = false, invert = false, flipY = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--normalize") normalize = true;
        else if (a == "--invert") invert = true;
        else if (a == "--flipY")  flipY = true;
    }

    BTFExtra extra;
    BTF* btf = LoadBTF(in.c_str(), &extra);
    if (!btf) {
        std::fprintf(stderr, "Failed to load: %s\n", in.c_str());
        return 2;
    }
    if (!btf->HeightMap || btf->HeightMapSize.Width == 0 || btf->HeightMapSize.Height == 0) {
        std::fprintf(stderr, "This BTF has no embedded height map (only FMF1/resampled variants do).\n");
        DestroyBTF(btf);
        return 3;
    }

    const uint32_t W = btf->HeightMapSize.Width;
    const uint32_t H = btf->HeightMapSize.Height;
    const size_t N = size_t(W) * H;

    // Copy into a working buffer
    std::vector<uint16_t> Hraw(btf->HeightMap, btf->HeightMap + N);

    // Optional flip (the loader flips reflectance SxV planes; if your height looks upside-down, try --flipY)
    if (flipY) {
        for (uint32_t y = 0; y < H / 2; ++y) {
            uint16_t* rowA = Hraw.data() + size_t(y) * W;
            uint16_t* rowB = Hraw.data() + size_t(H - 1 - y) * W;
            std::swap_ranges(rowA, rowA + W, rowB);
        }
    }

    // Stats
    auto [mn_it, mx_it] = std::minmax_element(Hraw.begin(), Hraw.end());
    uint16_t mn = *mn_it, mx = *mx_it;
    std::fprintf(stderr, "height: %ux%u  min=%u  max=%u\n", W, H, mn, mx);

    std::vector<uint16_t> Hout(N);

    if (normalize) {
        // Stretch to full 0..65535 for viewing
        const float denom = std::max(1u, uint32_t(mx - mn));
        for (size_t i = 0; i < N; ++i) {
            float t = (Hraw[i] - mn) / denom;
            if (invert) t = 1.0f - t;
            Hout[i] = (uint16_t)std::lround(t * 65535.0f);
        }
    } else {
        // Keep raw units; optionally invert for white=high look
        if (invert) {
            for (size_t i = 0; i < N; ++i) Hout[i] = 65535u - Hraw[i];
        } else {
            Hout = std::move(Hraw);
        }
    }

    // If request is .exr, save a temporary .pgm and convert via ImageMagick if available
    auto toLower = [](std::string s){ for (auto &c : s) c = (char)std::tolower((unsigned char)c); return s; };
    std::string outLower = toLower(out);
    if (outLower.size() >= 4 && outLower.substr(outLower.size()-4) == ".exr") {
        std::string tmpPGM = out + ".tmp.pgm";
        if (!save_pgm16_be(tmpPGM, (uint16_t)W, (uint16_t)H, Hout.data())) {
            std::fprintf(stderr, "Failed to save intermediate: %s\n", tmpPGM.c_str());
            DestroyBTF(btf);
            return 4;
        }
        // Try using ImageMagick's convert to produce EXR
        std::string cmd = "convert \"" + tmpPGM + "\" \"" + out + "\"";
        int rc = std::system(cmd.c_str());
        std::remove(tmpPGM.c_str());
        if (rc != 0) {
            std::fprintf(stderr, "ImageMagick 'convert' failed (rc=%d). EXR not written. Intermediate PGM removed.\n", rc);
            DestroyBTF(btf);
            return 5;
        }
        std::fprintf(stderr, "Wrote %s\n", out.c_str());
    } else {
        if (!save_pgm16_be(out, (uint16_t)W, (uint16_t)H, Hout.data())) {
            std::fprintf(stderr, "Failed to save: %s\n", out.c_str());
            DestroyBTF(btf);
            return 4;
        }
        std::fprintf(stderr, "Wrote %s\n", out.c_str());
    }
    DestroyBTF(btf);
    return 0;
}
