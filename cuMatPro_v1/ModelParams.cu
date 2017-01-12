#include "ModelParams.cu.h"

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
	initGibbsParams_kernel << <1, 2 >> >(this->d, dlConfig.d, localState);
	this->reflect();
	cout << "Initial Sample gam_d: " << h->gam_d << endl;
}

void ModelParams::reflect()
{
	cudaMemcpy(h, d, sizeof(_modelParams), cudaMemcpyDeviceToHost);
}