#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <iostream>
#include "gpuMat.h"
#include "gpuOpsAPI.h"

#define pn(x) printf("%5.0f", (double)x)
#define min(x, y) ((x)<(y)?(x):(y))

using namespace std;

int main()
{
	int S = 4;
	gpuMat<float> Y(S, S);
	gpuMat<bool> B(S, S);
	gpuMat<double> C(S, S);
	cout << Y.cols << "by" << Y.rows << endl;

	for (int i = 0; i < S; i++)
	{
		for (int j = 0; j < S; j++)
		{
			Y(i, j) = i*Y.cols + j;
			B(i, j) = (i>=j);
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
	gpuMat<double> mat1(4, 5);
	gpuMat<double> vec1(5, 1);
	gpuMat<double> result(4, 1);

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			mat1(i, j) = ((i + 1)%(j + 1));
		}
	}

	for (int i = 0; i < 5; i++)
	{
		vec1(i) = 1;
	}
	
	mat1.print();
	vec1.print();

	mat1.copy2Device();
	vec1.copy2Device();

	MatMul<double, double, double>(mat1.d_elems, vec1.d_elems, result.d_elems, 4, 1, 5);
	result.copy2Host();
	
	result.print();
}