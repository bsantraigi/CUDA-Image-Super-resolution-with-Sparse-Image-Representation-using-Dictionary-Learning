#pragma once

// Standard and CUDA Includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <ctime>

// User defined includes
#include "gpuMat.cu"
#include "gpuOpsAPI.cu"
#include "Timer.cu.h"
#include "DLLayer.cu"
#include "ImLoader.cu.h"

#define min(x, y) ((x)<(y)?(x):(y))

using namespace std;

int calcN(int imsize, int patchsize, int imcount);

void testRand();
int testPerformance();

int calcN();
void DLCode();
void DL_encapsulated();