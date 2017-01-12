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
#include "gpuMat.cuh"
#include "gpuOpsAPI.cuh"
#include "Timer.cuh"
#include "DLLayer.cu"
#include "ImLoader.cuh"
#include "DLConfig.cuh"
#include "ModelParams.cu"
#include "Random_kernels.cuh"

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