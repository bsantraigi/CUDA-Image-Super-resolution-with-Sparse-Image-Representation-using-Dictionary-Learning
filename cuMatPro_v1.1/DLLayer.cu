#pragma once

// Standard/CUDA Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// User defined Libraries
#include "Utilities.cu.h"
#include "gpuMat.cu"

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

	// Actual YDSB matrices
	int N, M, K;
	gpuMat<double> D;
	gpuMat<double> S;
	gpuMat<bool> B;
	gpuMat<double> PI;
	gpuMat<double> post_PI;

	void Init();
	void reflect();

	DLLayer(int propImSize = 256, int propPatchSize = 8, int propImCount = 5);
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
Marsaglia and Tsang�s Method
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
	double2 hyperParams_s{ dlConfig->a_s, dlConfig->b_s };
	double2 hyperParams_n{ dlConfig->a_n, dlConfig->b_n };
	double2 hyperParams_bias{ dlConfig->a_bias, dlConfig->b_bias };
	double gam_d, gam_s, gam_n, gam_bias;
	// Sample from gamrnd_d here
	gamrnd_d(&gam_d, &hyperParams_d, d_localstates);
	gamrnd_d(&gam_s, &hyperParams_s, d_localstates);
	gamrnd_d(&gam_n, &hyperParams_n, d_localstates);
	gamrnd_d(&gam_bias, &hyperParams_bias, d_localstates);

	// Copy the sampled values to modelParams
	modelParams->gam_d = gam_d;
	modelParams->gam_s = gam_s;
	modelParams->gam_n = gam_n;
	modelParams->gam_bias = gam_bias;
}

/*
Just fill in the D[:, col] with samples from a standard normal Dist.
Just launch with 1D launch configuration - split into blocks s.t.
each block has close to permissible amount of concurrent threads
*/
__global__ void DPointSample_kernel(double* D, int2* _size, int col, curandState_t* d_localstates)
{
	int Rows = _size->y;
	int Cols = _size->x;

	int row = blockIdx.x*blockDim.x + threadIdx.x;
	D[col*Rows + row] = curand_normal(&d_localstates[row]);
}

/*
Using Cholesky decomposition transform the already sampled
column D[:, col]
*/
__global__ void DSetVar_kernel(double* D, double* muD, double* covar, int2* _size, int col)
{
	int Rows = _size->y;
	int Cols = _size->x;

	// Do cholskey decomp
	// Multiply
}

__global__ void DSetMean_kernel(double* D, double* muD, int2* _size, int col)
{
	int Rows = _size->y;
	int Cols = _size->x;

	int row = blockIdx.x*blockDim.x + threadIdx.x;
	D[col*Rows + row] = D[col*Rows + row] + muD[row];
}

void mvnrnd_h(gpuMat<double> &D, gpuMat<double> &muD, gpuMat<double> &covar, gpuMat<double> &Ld, int col)
{
	// Launch DPointSample_kernel

	// Cholesky Decomp of covar and find Ld

	// Apply covar transformation - Multiply with Ld

	// Add mu - Launch DSetMean_kernel

}

__global__ void initDMatrix_kernel(double* D, int2* _size, int col, _modelParams* params, curandState_t* d_localstates)
{
	int rows = _size->y;
	int cols = _size->x;
}
/*
Calculate value of N from imsize, patchsize & imcount
*/
int calcN(int imsize, int patchsize, int imcount)
{
	return (imsize - patchsize + 1)*(imsize - patchsize + 1)*imcount;
}

/*
DLLayer Class method definitions
Must use copy constructor for objects (no need for pointers as temporary
objects aren't the issue there)
*/
DLLayer::DLLayer(int propImSize, int propPatchSize, int propImCount) : 
M(propPatchSize*propPatchSize), N(calcN(propImSize, propPatchSize, propImCount)), K(100),
D(gpuMat<double>(M, K)), S(gpuMat<double>(K, N)), B(gpuMat<bool>(K, N)), PI(gpuMat<double>(K, 1)), post_PI(gpuMat<double>(K, N))
{
	Utilities::prettyStart("Constructing LAYER");
	// DLLayer Matrices YDSB	
	cout << "M: " << M << ", N: " << N << ", K: " << K << endl;

	/*D = gpuMat<double>(M, K);
	S = gpuMat<double>(K, N);
	B = gpuMat<bool>(K, N);
	PI = gpuMat<double>(K, 1);
	post_PI = gpuMat<double>(K, N);*/

	// Model Params ctor
	h_params = new _modelParams();
	cudaMalloc(&d_params, sizeof(_modelParams));
	cudaMalloc(&localState, sizeof(curandState_t));
	// Need only a single state variable
	setup_kernel << <1, 1 >> >(localState, (unsigned int)time(NULL));

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
	
	Utilities::prettyEnd("LAYER Constructed");	

	Utilities::prettyStart("Layer Initialization STARTING");
	this->Init();
	Utilities::prettyStart("Layer Initialization Complete");
}


DLLayer::~DLLayer()
{
	cout << "DLLayer destructor!!!" << endl;

	//ModelParams.cu
	cout << "Destroying Model Params" << endl;
	delete[] h_params;
	cudaFree(d_params);
	cudaFree(localState);

	// DLConfig.cu
	cout << "Destroying dlConfig object" << endl;
	free(h_dlConfig);
	cudaFree(d_dlConfig);

	
}

void DLLayer::Init()
{
	// Initialize Model hyperparameters
	initGibbsParams_kernel << <1, 1 >> >(d_params, d_dlConfig, localState);
	this->reflect();
	cout << "Initial Sample gam_d: " << h_params->gam_d << endl;
	cout << "Initial Sample gam_s: " << h_params->gam_s << endl;
	cout << "Initial Sample gam_n: " << h_params->gam_n << endl;
	cout << "Initial Sample gam_bias: " << h_params->gam_bias << endl;

	//
}

void DLLayer::reflect()
{
	// reflect current state of hyperparams
	cudaMemcpy(h_params, d_params, sizeof(_modelParams), cudaMemcpyDeviceToHost);
}

