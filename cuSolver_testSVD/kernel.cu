#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
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

int main()
{
	const int m = 64;
	const int rows = m;
	const int cols = m;
	/*       | 3.5 0.5 0 |
	*   A = | 0.5 3.5 0 |
	*       | 0   0   2 |
	*
	*/
	double A[rows*m];
	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			A[i*rows + j] = (double)rand() / RAND_MAX;
			if (i == j){
				A[i*rows + j] += 1;
			}
		}
	}

	cusolverDnHandle_t handle;
	cusolverDnCreate(&handle);
	int lwork;

	cusolverDnDgesvd_bufferSize(
		handle,
		rows,
		cols,
		&lwork);

	double *d_A;
	cudaMalloc(&d_A, sizeof(double)*rows*cols);
	cudaMemcpy(d_A, A, sizeof(double)*rows*cols, cudaMemcpyHostToDevice);

	double *d_S;
	cudaMalloc(&d_S, sizeof(double)*rows);

	double *d_U;
	cudaMalloc(&d_U, sizeof(double)*rows*rows);

	double *d_VT;
	cudaMalloc(&d_VT, sizeof(double)*rows*rows);

	double *d_work;
	cudaMalloc(&d_work, sizeof(double)*lwork);

	double *d_rwork;
	cudaMalloc(&d_rwork, sizeof(double)*(rows - 1));

	int *devInfo;
	cudaMalloc(&devInfo, sizeof(int));

	signed char jobu = 'A';
	signed char jobvt = 'A';
	START_METER
	cusolverDnDgesvd(
		handle,
		jobu,
		jobvt,
		rows,
		cols,
		d_A,
		rows,
		d_S,
		d_U,
		rows,
		d_VT,
		rows,
		d_work,
		lwork,
		d_rwork,
		devInfo);
	STOP_METER
	cudaFree(d_A);
	cudaFree(d_rwork);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_VT);
	cudaFree(d_work);

}