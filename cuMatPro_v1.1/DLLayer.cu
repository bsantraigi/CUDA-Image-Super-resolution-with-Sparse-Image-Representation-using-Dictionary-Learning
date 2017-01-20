#pragma once

// Macro for timing kernel runs
#define START_METER {\
	cudaEvent_t start, stop;\
	float elapsedTime;\
	cudaEventCreate(&start);\
	cudaEventRecord(start, 0);
#define STOP_METER cudaEventCreate(&stop);\
	cudaEventRecord(stop, 0);\
	cudaEventSynchronize(stop);\
	cudaEventElapsedTime(&elapsedTime, start, stop);\
	printf("Elapsed time : %f ms\n", elapsedTime);\
	}

//Do kernel activity here
// Standard/CUDA Includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cusolverDn.h>

// User defined Libraries
#include "Utilities.cu.h"
#include "gpuMat.cu"
#include "gpuOpsAPI.cu"

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
* DATA STRUCT: _cusolverStruct
*	This data structure contains matrices and helper cuda variables
*	required for sampling of D.col or S.col from a Multivariate Random distribution
*/
typedef struct{
	cusolverDnHandle_t handle; //Host
	int Lwork; // Host
	double* workspace; //Device
	int *devInfo; // Device

	// Host-Device paired
	gpuMat<double> mu; // Mu - Mean vector
	gpuMat<double> covar; // Covariance Matrix

	gpuMat<double> eigenVals; // Vector of eigen values
	gpuMat<double> diagonally; // Diagonal matrix with eigen value
	gpuMat<double> L; // L matrix from LLT decomposition of covar
} _cusolverStruct;

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
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &d_localstates[id]);
	//printf("[setup_kernel] id: %d, seed: %u\n", id, seed);
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
			if (z > -1 / c && log(u) < (z*z / 2 + d - d*v + d*log(v))){
				*x = d*v / beta;
				*d_localstates = localState;
				//printf("GRND: a = %f, b = %f, x = %f\n", alpha, beta, *x);
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

__global__ void InitGammaParams_kernel(_modelParams* modelParams, _dlConfig* dlConfig, curandState_t* d_localstates)
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

__global__ void copyDiag_sqrt(double* srcVec, double* destMat, int m)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	destMat[i*m + i] = sqrt(srcVec[i]);
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
	//printf("[%d]: %f\n", col*Rows + row, D[col*Rows + row]);
}

/*
Using LLT decomposition transform the already sampled
column D[:, col]

Launch Rows number of threads BUT only ONE BLOCK
Each thread compute a value from final output @D
*/
__global__ void DSetVar_kernel(double* D, double* L, int2* _size, int col)
{
	// Dimension of D
	int Rows = _size->y;
	int DCols = _size->x;
	
	// covar = L'*L
	// Multiply L' * D[:, col]
	// While multiplying use only the lower part of the covar matrix
	//int k = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Choose row of D to fill in
	int k = threadIdx.x;
	__shared__ double d_final[1024];

	double cvalue = 0;
	for (int i = 0; i < Rows; i++)
	{
		cvalue += L[i*Rows + k] * D[col*Rows + i];
	}
	d_final[k] = cvalue;
	
	// Wait for all threads finish calculating the rows
	__syncthreads();

	D[col*Rows + k] = d_final[k];
}

__global__ void DSetMean_kernel(double* D, double* muD, int2* _size, int col)
{
	int Rows = _size->y;
	int Cols = _size->x;

	int row = blockIdx.x*blockDim.x + threadIdx.x;
	D[col*Rows + row] = D[col*Rows + row] + muD[row];
}

/*
MakeIdentity_d:
Convert a square matrix to identity matrix
*/
__global__ void MakeIdentity_d(double* D, int2 *size)
{
	int Rows = size->y;
	int Cols = size->x;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (row < Rows && col < Cols){
		D[col*Rows + row] = (row == col) ? 1 : 0;
		//printf("[%d, %d] -> %d\n", row, col, col*Rows + row);
	}
}

void MakeIdentity_H(gpuMat<double> &D)
{
	//cout << "Make Identity: " << D.rows << endl;
	int K = D.rows;
	const int L = std::min(K, 32);
	dim3 threadsPerBlock(L, L);
	uint nbx = (unsigned int)ceil((double)K / L);
	dim3 numBlocks(nbx, nbx);

	MakeIdentity_d <<<numBlocks, threadsPerBlock >>>(D.d_elems, D.d_size);
}

/*
Calculate value of N from imsize, patchsize & imcount
*/
int calcN(int imsize, int patchsize, int imcount)
{
	return (imsize - patchsize + 1)*(imsize - patchsize + 1)*imcount;
}

void InitSolverKit_evd(_cusolverStruct &solverKit, gpuMat<double> &D)
{
	int Rows = D.rows;
	solverKit.eigenVals.create(D.rows, 1);
	solverKit.L.create(D.rows, D.rows);

	// Initialize covariance matrix to I
	solverKit.covar.create(D.rows, D.rows);
	MakeIdentity_H(solverKit.covar);

	// Initialize diagonally matrix
	solverKit.diagonally.create(D.rows, D.rows);
	MakeIdentity_H(solverKit.diagonally);

	// Initialize mu vector
	solverKit.mu.create(Rows, 1);
	gpuMat<double> &mu = solverKit.mu;
	for (int i = 0; i < D.rows; i++)
	{
		mu(i, 0) = 0;
	}
	mu.copy2Device();

	cublasFillMode_t MODE = CUBLAS_FILL_MODE_UPPER;
	cusolverDnCreate(&solverKit.handle);
	cudaMalloc(&solverKit.devInfo, sizeof(int));

	cusolverDnDsyevd_bufferSize(solverKit.handle, CUSOLVER_EIG_MODE_VECTOR,
		MODE, Rows, solverKit.covar.d_elems, Rows, solverKit.eigenVals.d_elems, &solverKit.Lwork);

	cudaMalloc(&solverKit.workspace, solverKit.Lwork*sizeof(double));
}

/*
M = LL^T decomposition for covariance matrix
*/
void LLT_d(_cusolverStruct &solverKit)
{
	gpuMat<double> &covar = solverKit.covar;
	cublasFillMode_t MODE = CUBLAS_FILL_MODE_LOWER;
	cusolverStatus_t status;
	gpuMat<double> &W = solverKit.eigenVals;
	int Rows = covar.rows;
	// cusolverDn - for dense mat, D - Double, potrf - Cholesky solver
	// sytrf - LDLT decomposition
	// syevd - EigenValue decomp
	// Taking Long time ~ 20 ms
	status = cusolverDnDsyevd(solverKit.handle, CUSOLVER_EIG_MODE_VECTOR, MODE,
		covar.rows, covar.d_elems, covar.rows, W.d_elems, solverKit.workspace, solverKit.Lwork, solverKit.devInfo);

	const int L = std::min(Rows, 32);
	dim3 threadsPerBlock(L);
	dim3 numBlocks((unsigned int)ceil((double)Rows / L));
	
	copyDiag_sqrt<<<numBlocks, threadsPerBlock>>>(W.d_elems, solverKit.diagonally.d_elems, Rows);
	MatMul<double, double, double>(covar.d_elems, solverKit.diagonally.d_elems, solverKit.L.d_elems, Rows, Rows, Rows);
	
	if (status != cudaSuccess){
		cout << "LLT Decomp. Failed Badly !!!" << endl;
	}
}

/*
CLASS DLLayer
*/
class DLLayer
{
public:
	// Model params
	_modelParams *h_params, *d_params;
	// Keep the size of d_localstates M*M or K
	curandState_t *d_localstates; // Device Allocated
	int statesCount;

	// DLConfig: Layer cofigurations
	_dlConfig *h_dlConfig;
	_dlConfig *d_dlConfig;

	// Actual YDSB matrices
	int N, M, K;
	gpuMat<double> D;
	gpuMat<double> S;
	gpuMat<bool> B;
	gpuMat<double> PI, Bias;
	gpuMat<double> post_PI;

	// Solverkits for cholesky decomposition while sampling D[:, i] and S[:, j]
	_cusolverStruct solverKitD;
	_cusolverStruct solverKitS;
	_cusolverStruct solverKitBias;

	DLLayer(int propImSize = 256, int propPatchSize = 8, int propImCount = 5);
	~DLLayer();

	void Init();
	void reflect();
	void mvnrnd_d(gpuMat<double> &holder, _cusolverStruct &solverKit, int col);
};

/*
DLLayer Class method definitions
Must use copy constructor for objects (no need for pointers as temporary
objects aren't the issue there)
*/
DLLayer::DLLayer(int propImSize, int propPatchSize, int propImCount) : 
M(propPatchSize*propPatchSize), N(calcN(propImSize, propPatchSize, propImCount)), K(100),
D(gpuMat<double>(M, K)), S(gpuMat<double>(K, N)), B(gpuMat<bool>(K, N)), PI(gpuMat<double>(K, 1)), post_PI(gpuMat<double>(K, N)),
Bias(gpuMat<double>(M, 1))
{
	Utilities::prettyStart("Constructing LAYER");
	// DLLayer Matrices YDSB	
	cout << "M: " << M << ", N: " << N << ", K: " << K << endl;
	statesCount = max(K, M);

	// Model Params ctor
	h_params = new _modelParams();
	cudaMalloc(&d_params, sizeof(_modelParams));
	
	cudaMalloc(&d_localstates, statesCount*sizeof(curandState_t));
	
	// Need only a M or K state variables - sampling rows of D in parallel
	const int L = std::min(statesCount, 32);
	dim3 threadsPerBlock(L);
	dim3 numBlocks((unsigned int)ceil((double)statesCount / L));
	setup_kernel <<<(uint)ceil(statesCount/L), L >>>(d_localstates, (unsigned int)time(NULL));

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
	cudaFree(d_localstates);

	// DLConfig.cu
	cout << "Destroying dlConfig object" << endl;
	free(h_dlConfig);
	cudaFree(d_dlConfig);

	
}

void DLLayer::Init()
{
	// Initialize Model hyperparameters - Generating initial samples of gammas
	InitGammaParams_kernel <<<1, 1 >>>(d_params, d_dlConfig, d_localstates);
	//this->reflect(); // Required if printing the values

	// Initialize LLT transformation sovler Kit
	InitSolverKit_evd(solverKitD, D);
	InitSolverKit_evd(solverKitS, S);
	InitSolverKit_evd(solverKitBias, Bias);

	{
		// Sample columns of D
		cout << "** SAMPLING COLUMNS of D **" << endl;

		START_METER
		for (int k = 0; k < K; k++)
		{
			mvnrnd_d(D, solverKitD, k);
			MakeIdentity_H(solverKitD.covar);
		}
		STOP_METER

		D.ToFile("outputs/d.csv");
		//solverKitD.L.ToFile("outputs/Ld.csv");
		//solverKitD.covar.ToFile("outputs/covarD.csv");
	}

	{
		// Sample columns of S
		cout << "** SAMPLING COLUMNS of S **" << endl;	

		//START_METER
		for (int n = 0; n < N; n++)
		{
			//cout << "S col: " << n << endl;
			mvnrnd_d(S, solverKitS, n);
			MakeIdentity_H(solverKitS.covar);
		}
		//STOP_METER
		S.ToFile("outputs/s.csv");
		//solverKitS.L.ToFile("outputs/Ls.csv");
		//solverKitS.covar.ToFile("outputs/covarS.csv");
	}

	{
		// Sample columns of S
		cout << "** SAMPLING Bias*" << endl;
		START_METER
		mvnrnd_d(Bias, solverKitBias, 0);
		STOP_METER
	}
}

void DLLayer::reflect()
{
	// reflect current state of hyperparams
	cudaMemcpy(h_params, d_params, sizeof(_modelParams), cudaMemcpyDeviceToHost);
}

void DLLayer::mvnrnd_d(gpuMat<double> &holder, _cusolverStruct &solverKit, int col)
{
	gpuMat<double> &covar = solverKit.covar;
	int Rows = holder.rows;
	int Cols = holder.cols;

	const int L = std::min(Rows, 32);
	dim3 threadsPerBlock(L);
	dim3 numBlocks((unsigned int)ceil((double)Rows / L));
	
	// Launch DPointSample_kernel
	DPointSample_kernel <<<numBlocks, threadsPerBlock>>> (holder.d_elems, holder.d_size , col, d_localstates);

	// LLT Decomp of covar and find Ld	
	// Takes most time - 21 ms for a single call
	LLT_d(solverKit);

	// Apply covar transformation - Multiply with Ld
	// Taking SECOND MOST time
	DSetVar_kernel<<<1, Rows>>>(holder.d_elems, solverKit.L.d_elems, holder.d_size, col);

	// Add mu - Launch DSetMean_kernel
	DSetMean_kernel<<<numBlocks, threadsPerBlock>>>(holder.d_elems, solverKit.mu.d_elems, holder.d_size, col);
}

