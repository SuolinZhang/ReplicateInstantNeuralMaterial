#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include <fstream>
namespace Falcor
{
std::vector<float> readBinaryFile(const char* filename);
void writeToBinaryFile(const std::vector<float>& vec, const std::string& filename);
}

