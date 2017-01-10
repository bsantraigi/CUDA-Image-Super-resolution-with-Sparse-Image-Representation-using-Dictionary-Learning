#include <cuda_runtime.h>
#include <iostream>

using namespace std;

typedef struct {
	double gam_d, gam_s, gam_n, gam_bias;
} _modelParams;

class ModelParams{
public:
	_modelParams *h;
	_modelParams *d;

	ModelParams();
	~ModelParams();
};

ModelParams::ModelParams()
{
	h = new _modelParams();
	cudaMalloc(&d, sizeof(_modelParams));
}

ModelParams::~ModelParams()
{
	cout << "Destroying Model Params" << endl;
	delete[] h;
	cudaFree(d);
}