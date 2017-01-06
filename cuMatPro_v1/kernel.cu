#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <iostream>
#include "gpuMat.h"
#include "gpuOpsAPI.h"

#define pn(x) printf("%5.0f", (double)x)

using namespace std;

int main()
{
	int S = 1024;
	gpuMat<float> Y(S, S);
	gpuMat<int> B(S, S);
	gpuMat<double> C(S, S);
	cout << Y.cols << "by" << Y.rows << endl;

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			Y(i, j) = i*Y.cols + j;
			B(i, j) = i>=j;
		}
	}
	Y.copy2Device();
	B.copy2Device();

	

	cout << endl;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			pn(Y(i, j));
		}
		cout << endl;
	}

	cout << endl;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			pn(B(i, j));

		}
		cout << endl;
	}

	// CUBLAS TEST
	/*float al = 1;
	float bet = 0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &al, Y.d_elems, S, B.d_elems, S, &bet, C.d_elems, S);*/

	Hadamard<float, int, double>(Y.d_elems, B.d_elems, C.d_elems, S, S);

	C.copy2Host();

	cout << endl;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			pn(C(i, j));
		}
		cout << endl;
	}
}