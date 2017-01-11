#pragma once
// Standard/CUDA Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// User Defined Includes
#include "Random_kernels.h"
#include "DLConfig.h"

typedef struct {
	double gam_d, gam_s, gam_n, gam_bias;
} _modelParams;

__global__ void initGibbsParams_kernel(_modelParams* modelParams, _dlConfig* dlConfig, curandState_t* d_localstates)
{
	double gam_d, gam_s, gam_n, gam_bias;
}