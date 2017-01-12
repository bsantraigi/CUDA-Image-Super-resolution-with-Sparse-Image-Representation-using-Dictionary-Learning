#pragma once

#include "ModelParams_kernels.cu.h"

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