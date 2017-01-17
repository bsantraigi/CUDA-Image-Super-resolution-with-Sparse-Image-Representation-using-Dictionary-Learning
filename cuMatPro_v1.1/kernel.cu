#pragma once
#include "kernel.cu.h"

Timer timer1;

int main(){
	//test();
	//DLCode();
	//testRand();
	DL_encapsulated();
	//testPerformance();
	cudaDeviceReset();
}

//int testPerformance()
//{
//	int S = 20;
//	gpuMat<float> Y(S, S);
//	gpuMat<bool> B(S, S);
//	gpuMat<double> C(S, S);
//	cout << Y.cols << "by" << Y.rows << endl;
//
//	for (int i = 0; i < S; i++)
//	{
//		for (int j = 0; j < S; j++)
//		{
//			Y(i, j) = i*Y.cols + j;
//			B(i, j) = (i >= j);
//		}
//	}
//	Y.copy2Device();
//	B.copy2Device();
//
//	Y.print();
//	B.print();
//
//	// CUBLAS TEST
//	/*float al = 1;
//	float bet = 0;
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &al, Y.d_elems, S, B.d_elems, S, &bet, C.d_elems, S);*/
//
//	MatMul<float, bool, double>(Y.d_elems, B.d_elems, C.d_elems, S, S, S);
//
//	C.copy2Host();
//	C.print();
//
//
//	// Test functions for rectangular matrices
//	int m = 682768, n = 256, k = 128;
//	gpuMat<float> mat1(m, k);
//	gpuMat<float> vec1(k, n);
//	gpuMat<float> result(m, n);
//
//	for (int i = 0; i < m; i++)
//	{
//		for (int j = 0; j < k; j++)
//		{
//			//mat1(i, j) = ((i + 1)%(j + 1));
//			mat1(i, j) = (float)rand() / RAND_MAX - 0.5;
//		}
//	}
//
//	for (int i = 0; i < k; i++)
//	{
//		for (int j = 0; j < n; j++)
//		{
//			vec1(i, j) = (float)rand() / RAND_MAX - 0.5;
//		}
//	}
//
//	mat1.print();
//	vec1.print();
//
//	mat1.copy2Device();
//	vec1.copy2Device();
//
//	cout << "Using my API." << endl;
//	{
//		cudaEvent_t start, stop;
//		float elapsedTime;
//
//		cudaEventCreate(&start);
//		cudaEventRecord(start, 0);
//
//		//Do kernel activity here
//		MatMul<float, float, float>(mat1.d_elems, vec1.d_elems, result.d_elems, m, n, k);
//
//		cudaEventCreate(&stop);
//		cudaEventRecord(stop, 0);
//		cudaEventSynchronize(stop);
//
//		cudaEventElapsedTime(&elapsedTime, start, stop);
//		printf("Elapsed time : %f ms\n", elapsedTime);
//	}
//
//	result.copy2Host();
//
//	result.print();
//
//	cout << "Using CUBLAS" << endl;
//	float al = 1;
//	float bet = 0;
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//
//	{
//		cudaEvent_t start, stop;
//		float elapsedTime;
//
//		cudaEventCreate(&start);
//		cudaEventRecord(start, 0);
//
//		//Do kernel activity here
//		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, mat1.d_elems, m, vec1.d_elems, k, &bet, result.d_elems, m);
//
//		cudaEventCreate(&stop);
//		cudaEventRecord(stop, 0);
//		cudaEventSynchronize(stop);
//
//		cudaEventElapsedTime(&elapsedTime, start, stop);
//		printf("Elapsed time : %f ms\n", elapsedTime);
//	}
//	result.copy2Host();
//
//	result.print();
//
//	cout << "Calculating in host CPU | Single thread" << endl;
//	{
//		timer1.start();
//		for (int i = 0; i < m; i++)
//		{
//			for (int j = 0; j < n; j++)
//			{
//				double cvalue = 0;
//				for (int l = 0; l < k; l++)
//				{
//					cvalue += mat1(i, l)*vec1(l, j);
//				}
//				result(i, j) = cvalue;
//			}
//		}
//		timer1.stop();
//	}
//
//	result.print();
//
//	return 0;
//}

//int calcN(int imsize, int patchsize, int imcount)
//{
//	return (imsize - patchsize + 1)*(imsize - patchsize + 1)*imcount;
//}
//
//void DLCode()
//{
//	int propImSize = 256;
//	int propPatchSize = 8;
//	int propImCount = 5;
//	int N = calcN(propImSize, propPatchSize, propImCount);
//	int M = propPatchSize*propPatchSize;
//	int K = 100;
//	cout << "M: " << M << ", N: " << N << ", K: " << K << endl;
//
//	ImLoader imloader(propImSize, propPatchSize);
//	gpuMat<double> Y(M, N);
//	imloader.GetDataMatrix(Y, propImCount);
//
//	gpuMat<double> D(M, K);
//	gpuMat<double> S(K, N);
//	gpuMat<bool> B(K, N);
//	gpuMat<double> PI(K, 1);
//	gpuMat<double> post_PI(K, N);
//
//	
//}

void DL_encapsulated(){
	int propImSize = 256;
	int propPatchSize = 8;
	int propImCount = 5;
	DLLayer layer1 = DLLayer(propImSize, propPatchSize, propImCount);
	
}