#pragma once

// Standard/CUDA Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <ctime>

// User Defined Includes
#include "Random_kernels.cu.h"
#include "DLConfig.cu.h"

typedef struct {
	double gam_d, gam_s, gam_n, gam_bias;
} _modelParams;

__global__ void initGibbsParams_kernel(_modelParams* modelParams, _dlConfig* dlConfig, curandState_t* d_localstates)
{
	double2 hyperParams_d{ dlConfig->a_d, dlConfig->b_d };
	double gam_d, gam_s, gam_n, gam_bias;
	// Sample from gamrnd_d here
	gamrnd_d(&gam_d, &hyperParams_d, d_localstates);

	// Copy the sampled values to modelParams
	modelParams->gam_d = gam_d;
	/*modelParams->gam_s = gam_s;
	modelParams->gam_n = gam_n;
	modelParams->gam_bias = gam_bias;*/
}
