#pragma once

#include "DLLayer_kernels.cu.h"

class DLLayer
{
	DLConfig dlConfig;
	ModelParams modelParams;
public:
	DLLayer();
	~DLLayer();
};