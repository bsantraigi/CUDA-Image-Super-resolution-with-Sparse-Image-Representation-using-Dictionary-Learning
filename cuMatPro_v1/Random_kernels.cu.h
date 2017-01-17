#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

/*
DECLARATIONS
*/
//__global__ void setup_kernel(curandState_t* d_localstates, unsigned int seed);
//__device__ void gamrnd_d(double* x, double2* params, curandState_t* d_localstates);
//__device__ void betarnd_d(double* x, double2* params, curandState_t* d_localstates);

void gamrnd_dwrap(double* x, double2* params, curandState_t* d_localstates);

