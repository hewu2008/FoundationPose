#!/bin/bash
FILE="csrc/common/cudaraster/impl/RasterImpl.cpp"

# 在文件开头添加头文件
sed -i '1i#include <cstring>\n#include <algorithm>\n#include <cuda_runtime.h>\nusing namespace std;' $FILE

# 添加 NVDR_CHECK 宏定义（如果未定义）
sed -i '/#include "RasterImpl.hpp"/a #ifndef NVDR_CHECK_CUDA_ERROR\n#define NVDR_CHECK_CUDA_ERROR(x) x\n#endif\n#ifndef NVDR_CHECK\n#define NVDR_CHECK(x, msg) if(!(x)) { throw std::runtime_error(msg); }\n#endif' $FILE

# 添加 LOG 宏（如果未定义）
sed -i '/#include "RasterImpl.hpp"/a #ifndef LOG\n#define LOG(level) std::cout\n#endif' $FILE
