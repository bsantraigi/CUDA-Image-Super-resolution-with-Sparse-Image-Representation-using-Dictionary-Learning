#pragma once

#include "ModelParams_kernels.cu"

using namespace std;

class ModelParams{
public:
	_modelParams *h;
	_modelParams *d;
	curandState_t *localState;

	ModelParams();
	~ModelParams();
	void Init(DLConfig &dlConfig);
	void reflect();
};

ModelParams::ModelParams()
{
	h = new _modelParams();
	cudaMalloc(&d, sizeof(_modelParams));
	cudaMalloc(&localState, sizeof(curandState_t));
	// Need only a single state variable
	setup_kernel << <1, 1 >> >(localState, time(NULL));
}

ModelParams::~ModelParams()
{
	cout << "Destroying Model Params" << endl;
	delete[] h;
	cudaFree(d);
	cudaFree(localState);
}

void ModelParams::Init(DLConfig &dlConfig)
{
	initGibbsParams(this->d, dlConfig.d, localState);
	this->reflect();
	cout << "Initial Sample gam_d: " << h->gam_d << endl;
}

void ModelParams::reflect()
{
	cudaMemcpy(h, d, sizeof(_modelParams), cudaMemcpyDeviceToHost);
}