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
#include "DLLayer.cu.h"
#include "ImLoader.cu.h"
#include "DLConfig.cu.h"
#include "ModelParams.cu.h"
#include "Random_kernels.cu.h"

#define min(x, y) ((x)<(y)?(x):(y))

using namespace std;

int calcN(int imsize, int patchsize, int imcount);


__global__ void uniformTest_kernel(double* d_samples, curandState_t* d_localstates);
__global__ void gammaTest_kernel(curandState_t* d_localstates, double2* params, double* d_samples, int length);
__global__ void betaTest_kernel(curandState_t* d_localstates, double2* params, double* d_samples, int length);

void testRand();
int testPerformance();

int calcN();
void DLCode();
void DL_encapsulated();