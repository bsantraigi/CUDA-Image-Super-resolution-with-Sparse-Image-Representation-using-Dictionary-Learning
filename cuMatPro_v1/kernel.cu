#pragma once
#include "kernel.cu.h"

Timer timer1;

int main(){
	//test();
	//DLCode();
	//testRand();
	DL_encapsulated();
	cudaDeviceReset();
}

__global__ void uniformTest_kernel(double* d_samples, curandState_t* d_localstates)
{
	int length = 16;
	int sid = threadIdx.x * length;
	curandState_t localState = d_localstates[threadIdx.x];
	for (int i = 0; i < length; i++)
	{
		d_samples[sid + i] = curand_uniform(&localState);
	}
}

__global__ void gammaTest_kernel(curandState_t* d_localstates, double2* params, double* d_samples, int length)
{
	int sid = threadIdx.x * length;
	curandState_t localState = d_localstates[threadIdx.x];
	for (int i = 0; i < length; i++)
	{
		gamrnd_d(d_samples + sid + i, params, &localState);
	}
}

__global__ void betaTest_kernel(curandState_t* d_localstates, double2* params, double* d_samples, int length)
{
	int sid = threadIdx.x * length;
	curandState_t localState = d_localstates[threadIdx.x];
	for (int i = 0; i < length; i++)
	{
		betarnd_d(d_samples + sid + i, params, &localState);
	}
}

void testRand()
{
	curandState_t* d_states;
	int seqs = 200;
	int length = 32;
	cudaMalloc(&d_states, sizeof(curandState_t) * seqs);
	setup_kernel<<<1, seqs>>>(d_states, time(NULL));
	double* samples = new double[seqs * length];
	double* d_samples;
	size_t bytes = sizeof(double)*seqs * length;
	cudaMalloc(&d_samples, bytes);
	cudaMemset(d_samples, 0, bytes);
	/*uniformTest_kernel<<<1, seqs>>>(d_samples, d_states);
	cudaMemcpy(samples, d_samples, bytes, cudaMemcpyDeviceToHost);

	for (int s = 0; s < seqs; s++)
	{
	for (int i = 0; i < length; i++)
	{
	printf("%4.2f ", samples[s * length + i]);
	}
	cout << endl;
	}*/


	{
		cout << "Gamma Distro" << endl;
		double2 params{ 0.8, 6000 };
		double2* params_d;
		cudaMalloc(&params_d, sizeof(double2));
		cudaMemcpy(params_d, &params, sizeof(double2), cudaMemcpyHostToDevice);
		gammaTest_kernel << <1, seqs >> >(d_states, params_d, d_samples, length);

		cudaMemcpy(samples, d_samples, bytes, cudaMemcpyDeviceToHost);

		ofstream gcsv("gamout.csv");
		for (int s = 0; s < seqs; s++)
		{
			for (int i = 0; i < length; i++)
			{
				gcsv << samples[s * length + i] << endl;
			}
		}
		gcsv.close();
	}

	{
		cout << "Beta Distro" << endl;
		double2 params{ 0.8, 3000 };
		double2* params_d;
		cudaMalloc(&params_d, sizeof(double2));
		cudaMemcpy(params_d, &params, sizeof(double2), cudaMemcpyHostToDevice);
		betaTest_kernel << <1, seqs >> >(d_states, params_d, d_samples, length);

		cudaMemcpy(samples, d_samples, bytes, cudaMemcpyDeviceToHost);

		ofstream gcsv("betaout.csv");
		for (int s = 0; s < seqs; s++)
		{
			for (int i = 0; i < length; i++)
			{
				gcsv << samples[s * length + i] << endl;
			}
		}
		gcsv.close();
	}

	// Free mems
	delete[] samples;
	cudaFree(d_samples);
	cudaFree(d_states);
}

int testPerformance()
{
	int S = 20;
	gpuMat<float> Y(S, S);
	gpuMat<bool> B(S, S);
	gpuMat<double> C(S, S);
	cout << Y.cols << "by" << Y.rows << endl;

	for (int i = 0; i < S; i++)
	{
		for (int j = 0; j < S; j++)
		{
			Y(i, j) = i*Y.cols + j;
			B(i, j) = (i >= j);
		}
	}
	Y.copy2Device();
	B.copy2Device();

	Y.print();
	B.print();

	// CUBLAS TEST
	/*float al = 1;
	float bet = 0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &al, Y.d_elems, S, B.d_elems, S, &bet, C.d_elems, S);*/

	MatMul<float, bool, double>(Y.d_elems, B.d_elems, C.d_elems, S, S, S);

	C.copy2Host();
	C.print();


	// Test functions for rectangular matrices
	int m = 682768, n = 256, k = 128;
	gpuMat<float> mat1(m, k);
	gpuMat<float> vec1(k, n);
	gpuMat<float> result(m, n);

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < k; j++)
		{
			//mat1(i, j) = ((i + 1)%(j + 1));
			mat1(i, j) = (float)rand() / RAND_MAX - 0.5;
		}
	}

	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < n; j++)
		{
			vec1(i, j) = (float)rand() / RAND_MAX - 0.5;
		}
	}

	mat1.print();
	vec1.print();

	mat1.copy2Device();
	vec1.copy2Device();

	cout << "Using my API." << endl;
	{
		cudaEvent_t start, stop;
		float elapsedTime;

		cudaEventCreate(&start);
		cudaEventRecord(start, 0);

		//Do kernel activity here
		MatMul<float, float, float>(mat1.d_elems, vec1.d_elems, result.d_elems, m, n, k);

		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("Elapsed time : %f ms\n", elapsedTime);
	}

	result.copy2Host();

	result.print();

	cout << "Using CUBLAS" << endl;
	float al = 1;
	float bet = 0;
	cublasHandle_t handle;
	cublasCreate(&handle);

	{
		cudaEvent_t start, stop;
		float elapsedTime;

		cudaEventCreate(&start);
		cudaEventRecord(start, 0);

		//Do kernel activity here
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, mat1.d_elems, m, vec1.d_elems, k, &bet, result.d_elems, m);

		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("Elapsed time : %f ms\n", elapsedTime);
	}
	result.copy2Host();

	result.print();

	/*cout << "Calculating in host CPU | Single thread" << endl;
	{
	timer1.start();
	for (int i = 0; i < m; i++)
	{
	for (int j = 0; j < n; j++)
	{
	double cvalue = 0;
	for (int l = 0; l < k; l++)
	{
	cvalue += mat1(i, l)*vec1(l, j);
	}
	result(i, j) = cvalue;
	}
	}
	timer1.stop();
	}

	result.print();*/

	return 0;
}

int calcN(int imsize, int patchsize, int imcount)
{
	return (imsize - patchsize + 1)*(imsize - patchsize + 1)*imcount;
}

void DLCode()
{
	int propImSize = 256;
	int propPatchSize = 8;
	int propImCount = 5;
	int N = calcN(propImSize, propPatchSize, propImCount);
	int M = propPatchSize*propPatchSize;
	int K = 100;
	cout << "M: " << M << ", N: " << N << ", K: " << K << endl;

	ImLoader imloader(propImSize, propPatchSize);
	gpuMat<double> Y(M, N);
	imloader.GetDataMatrix(Y, propImCount);

	DLConfig config1;
	DLConfig *config1_d;
	cudaMalloc(&config1_d, sizeof(DLConfig));
	cudaMemcpy(config1_d, &config1, sizeof(DLConfig), cudaMemcpyHostToDevice);

	gpuMat<double> D(M, K);
	gpuMat<double> S(K, N);
	gpuMat<bool> B(K, N);
	gpuMat<double> PI(K, 1);
	gpuMat<double> post_PI(K, N);

	ModelParams modelParams1;
}

void DL_encapsulated(){
	DLLayer layer1;
}