#pragma once

// Standard/CUDA Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// User defined Libraries

using namespace std;

/*
DATA STRUCTURES
*/
typedef struct {
	double gam_d, gam_s, gam_n, gam_bias;
} _modelParams;

typedef struct{
	int K;
	double a_d, a_s, a_bias, a_n;
	double b_d, b_s, b_bias, b_n;
	double a_pi, b_pi;
} _dlConfig;

/*
CLASS DLLayer
*/
class DLLayer
{
public:
	// Model params
	_modelParams *h_params, *d_params;
	curandState_t *localState;

	// DLConfig: Layer cofigurations
	_dlConfig *h_dlConfig;
	_dlConfig *d_dlConfig;

	void Init();
	void reflect();

	DLLayer();
	~DLLayer();
};
//#include "DLLayer.cu.h"

// GPU Kernels

/*
DEFINITIONS
*/
__global__ void setup_kernel(curandState_t* d_localstates, unsigned int seed)
{
	/*QUALIFIERS void curand_init(unsigned long long seed,
	unsigned long long subsequence,
	unsigned long long offset,
	curandStateXORWOW_t *state)*/
	int id = threadIdx.x;
	curand_init(seed, id, 0, &d_localstates[id]);
}

/*
Gamma Random Variable generator
Marsaglia and Tsang’s Method
*/
__device__ void gamrnd_d(double* x, double2* params, curandState_t* d_localstates)
{
	double alpha = params->x;
	double beta = params->y;

	if (alpha >= 1){
		curandState_t localState = *d_localstates; // Be careful the change in localState variable needs to be reflected back to d_localStates
		double d = alpha - 1 / 3.0, c = 1 / sqrt(9 * d);
		do{
			double z = curand_normal(&localState);
			double u = curand_uniform(&localState);
			double v = pow((double) 1.0f + c*z, (double) 3.0f);
			double extra = 0;
			if (z > -1 / c && log(u) < (z*z / 2 + d - d*v + d*log(v))){
				*x = d*v / beta;
				*d_localstates = localState;
				printf("GRND: a = %f, b = %f, x = %f\n", alpha, beta, *x);
				return;
			}
		} while (true);
	}
	else{
		double r;
		params->x += 1;
		gamrnd_d(&r, params, d_localstates);

		curandState_t localState = *d_localstates;
		double u = curand_uniform(&localState);
		*x = r*pow((double)u, (double)1 / alpha);
		params->x -= 1;
		return;
	}
}

/*
Algorithm as mentioned in Wikipedia:
x ~ Gamma(a, 1)
y ~ Gamma(b, 1)
then,
z = x/(x+y) ~ Beta(a, b)
*/
__device__ void betarnd_d(double* x, double2* params, curandState_t* d_localstates)
{
	double alpha = params->x;
	double beta = params->y;

	double2 params1{ params->x, 1 };
	double x1;
	gamrnd_d(&x1, &params1, d_localstates);

	double2 params2{ params->y, 1 };
	double x2;
	gamrnd_d(&x2, &params2, d_localstates);

	*x = x1 / (x1 + x2);
}

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

/*
DLLayer Class method definitions
*/
DLLayer::DLLayer()
{
	// Model Params ctor
	h_params = new _modelParams();
	cudaMalloc(&d_params, sizeof(_modelParams));
	cudaMalloc(&localState, sizeof(curandState_t));
	// Need only a single state variable
	setup_kernel << <1, 1 >> >(localState, time(NULL));

	// DLConfig.cu
	h_dlConfig = new _dlConfig();
	h_dlConfig->K = 80;
	h_dlConfig->a_d = h_dlConfig->a_s = h_dlConfig->a_bias = 1;
	h_dlConfig->b_d = h_dlConfig->b_s = h_dlConfig->b_bias = 1;
	h_dlConfig->a_n = h_dlConfig->b_n = 1e-1;

	h_dlConfig->a_pi = 1;
	h_dlConfig->b_pi = 1200;

	cudaMalloc(&d_dlConfig, sizeof(_dlConfig));
	cudaMemcpy(d_dlConfig, h_dlConfig, sizeof(_dlConfig), cudaMemcpyHostToDevice);
	
	this->Init();
}

DLLayer::~DLLayer()
{
	//ModelParams.cu
	cout << "Destroying Model Params" << endl;
	delete[] h_params;
	cudaFree(d_params);
	cudaFree(localState);

	// DLConfig.cu
	free(h_dlConfig);
	cudaFree(d_dlConfig);

	
}

void DLLayer::Init()
{
	// Initialize Model hyperparameters
	initGibbsParams_kernel << <1, 1 >> >(d_params, d_dlConfig, localState);
	this->reflect();
	cout << "Initial Sample gam_d: " << h_params->gam_d << endl;
}

void DLLayer::reflect()
{
	// reflect current state of hyperparams
	cudaMemcpy(h_params, d_params, sizeof(_modelParams), cudaMemcpyDeviceToHost);
}

