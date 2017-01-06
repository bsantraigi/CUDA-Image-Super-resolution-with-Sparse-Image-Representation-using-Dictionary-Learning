#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <iostream>
using namespace std;

/* GPU HADAMARD PRODUCT FUNCTION
*	Following are the gpu kernel and the host-side wrapper function for two matrices 
*	of same dimensions.
*/
template <typename T1, typename T2, typename T3>
__global__ void hadamard_d(T1* A, T2* B, T3* C, int m, int n)
{
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int c = blockIdx.x*blockDim.x + threadIdx.x;

	if (r < m && c < n){
		C[c*m + r] = (T3)A[c*m + r] * B[c*m + r];
	}
}

/*
Call the following function like
// Hadamard<float, float, float>(Y.d_elems, B.d_elems, C.d_elems, M, N);
*/
template <typename T1, typename T2, typename T3>
void Hadamard(T1* A, T2* B, T3* C, int m, int n)
{
	const int L = max(m, n) > 16?16:max(m, n);
	dim3 threadsPerBlock(L, L);
	dim3 numBlocks(ceil((double)m / L), ceil((double)n / L));

	hadamard_d<T1, T2, T3> <<<numBlocks, threadsPerBlock >>>(A, B, C, m, n);
}

/* GPU ADD FUNCTION
*	Following are the gpu kernel and the host side wrapper function for matrix 
*	addition operation. 
*/

template <typename T1, typename T2, typename T3>
__global__ void add_d(T1* A, T2* B, T3* C, int m, int n)
{
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int c = blockIdx.x*blockDim.x + threadIdx.x;

	if (r < m && c < n){
		C[c*m + r] = A[c*m + r] + B[c*m + r];
	}
}

/*
Call the following function like
// Add<float, float, float>(Y.d_elems, B.d_elems, C.d_elems, M, N);
*/
template <typename T1, typename T2, typename T3>
void Add(T1* A, T2* B, T3* C, int m, int n)
{
	const int L = max(m, n) > 16 ? 16 : max(m, n);
	dim3 threadsPerBlock(L, L);
	dim3 numBlocks(ceil((double)m / L), ceil((double)n / L));

	cout << threadsPerBlock.y <<" " << threadsPerBlock.x << endl;
	cout << numBlocks.y << " " << numBlocks.x << endl;

	add_d<T1, T2, T3> <<<numBlocks, threadsPerBlock >>>(A, B, C, m, n);
}

/* GPU MATRIX MULTIPLY FUNCTION
*	Following are the gpu kernel and the host side wrapper function for matrix
*	multiplication of two matrices A and B where A is m x k matrix and B is a 
*	k x n matrix.
*/

template <typename T1, typename T2, typename T3>
__global__ void matmul_d(T1* A, T2* B, T3* C, int m, int n, int k)
{
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	T3 cvalue = 0;
	if (r < m && c < n){
		for (int i = 0; i < k; i++)
		{
			cvalue += A[k*m + r] + B[c*m + k];
		}
		C[c*m + r] = cvalue;
	}
}

/*
Call the following function like
// MatMul<float, float, float>(Y.d_elems, B.d_elems, C.d_elems, M, N, K);
*/
template <typename T1, typename T2, typename T3>
void MatMul(T1* A, T2* B, T3* C, int m, int n, int k)
{
	const int L = max(m, n) > 16 ? 16 : max(m, n);
	dim3 threadsPerBlock(L, L);
	dim3 numBlocks(ceil((double)m / L), ceil((double)n / L));

	cout << threadsPerBlock.y << " " << threadsPerBlock.x << endl;
	cout << numBlocks.y << " " << numBlocks.x << endl;

	matmul_d<T1, T2, T3><<<numBlocks, threadsPerBlock >>>(A, B, C, m, n, k);
}