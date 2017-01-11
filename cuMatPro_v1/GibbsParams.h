#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "ModelParams_kernels.h"

using namespace std;

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