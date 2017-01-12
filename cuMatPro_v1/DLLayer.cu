#pragma once
#include "DLLayer_kernels.cu"

class DLLayer
{
	DLConfig dlConfig;
	ModelParams modelParams;
public:
	DLLayer();
	~DLLayer();
};

DLLayer::DLLayer()
{

	// Initialize hyperprior and prior parameters e.g. gam_a, gam_d, etc.
	modelParams.Init(dlConfig);
}

DLLayer::~DLLayer()
{
}


