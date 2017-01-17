#include "DLLayer.cu.h"

DLLayer::DLLayer()
{

	// Initialize hyperprior and prior parameters e.g. gam_a, gam_d, etc.
	modelParams.Init(dlConfig);
}

DLLayer::~DLLayer()
{
}


