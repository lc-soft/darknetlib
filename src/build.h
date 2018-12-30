#ifndef DARKNET_BUILD_H
#define DARKNET_BUILD_H

#ifdef _WIN32
#define WIN32 1
#endif

#include <stdlib.h>

#define getenv

#define DEBUG_MSG(format, ...)                                        \
	printf(__FILE__ " %d: %s(): " format, __LINE__, __FUNCTION__, \
	       ##__VA_ARGS__)

// Define to 1 if you have GPU.
// Check the NVIDIA GPU Computing Toolkit are installed, if aren't, please
// downloat it: https://developer.nvidia.com/cuda-toolkit
//#define GPU 1

// Define to 1 if you want to build with cuDNN (speedup neural network).
// To use cuDNN, you need download and install cuDNN v7.4.1 for CUDA 10.0:
// https://developer.nvidia.com/cudnn
//#define CUDNN 1

// Define to 1 If you have GPU with Tensor Cores (nVidia Titan V / Tesla V100 /
// DGX-2 and later) speedup Detection 3x, Training 2x
//#define CUDNN_HALF 1

#endif
