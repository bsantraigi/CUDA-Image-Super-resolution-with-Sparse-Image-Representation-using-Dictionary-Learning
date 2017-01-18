/*
* How to compile (assume cuda is installed at /usr/local/cuda/)
*   nvcc -c -I/usr/local/cuda/include syevd_example.cpp
*   g++ -o -fopenmp a.out syevd_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
*
*/

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
void printMatrix(int m, int n, const double*A, int rows, const char* name)
{
	for (int row = 0; row < m; row++){
		for (int col = 0; col < n; col++){
			double Areg = A[row + col*rows];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
	}
}

int main(int argc, char*argv[])
{
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
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
	//double lambda[m] = { 2.0, 3.0, 4.0 };

	double V[rows*m]; // eigenvectors
	double W[m]; // eigenvalues

	double *d_A = NULL;
	double *d_W = NULL;
	int *devInfo = NULL;
	double *d_work = NULL;
	int  lwork = 0;

	int info_gpu = 0;

	/*printf("A = (matlab base-1)\n");
	printMatrix(m, m, A, rows, "A");
	printf("=====\n");*/
		// step 1: create cusolver/cublas handle
		cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * rows * m);
	cudaStat2 = cudaMalloc((void**)&d_W, sizeof(double) * m);
	cudaStat3 = cudaMalloc((void**)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * rows * m, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);

	// step 3: query working space of syevd
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cusolver_status = cusolverDnDsyevd_bufferSize(
		cusolverH,
		jobz,
		uplo,
		m,
		d_A,
		rows,
		d_W,
		&lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	assert(cudaSuccess == cudaStat1);

	// step 4: compute spectrum
	START_METER
	cusolver_status = cusolverDnDsyevd(
		cusolverH,
		jobz,
		uplo,
		m,
		d_A,
		rows,
		d_W,
		d_work,
		lwork,
		devInfo);
	STOP_METER
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*rows*m, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	printf("after syevd: info_gpu = %d\n", info_gpu);
	assert(0 == info_gpu);

	/*printf("eigenvalue = (matlab base-1), ascending order\n");
	for (int i = 0; i < m; i++){
		printf("W[%d] = %E\n", i + 1, W[i]);
	}*/

	/*printf("V = (matlab base-1)\n");
	printMatrix(m, m, V, rows, "V");
	printf("=====\n");*/

	// step 4: check eigenvalues
	/*double lambda_sup = 0;
	for (int i = 0; i < m; i++){
		double error = fabs(lambda[i] - W[i]);
		lambda_sup = (lambda_sup > error) ? lambda_sup : error;
	}
	printf("|lambda - W| = %E\n", lambda_sup);*/

	// free resources
	if (d_A) cudaFree(d_A);
	if (d_W) cudaFree(d_W);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);

	if (cusolverH) cusolverDnDestroy(cusolverH);

	cudaDeviceReset();

	return 0;
}