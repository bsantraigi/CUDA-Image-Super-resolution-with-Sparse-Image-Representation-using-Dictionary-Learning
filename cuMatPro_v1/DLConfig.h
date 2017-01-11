#pragma once
#include <curand_kernel.h>

typedef struct{
	int K;
	double a_d, a_s, a_bias, a_n;
	double b_d, b_s, b_bias, b_n;
	double a_pi, b_pi;
	
} _dlConfig;

class DLConfig
{
public:
	_dlConfig *h;
	_dlConfig *d;

	DLConfig();
	~DLConfig();
	void apply();
};


DLConfig::DLConfig(){
	h = new _dlConfig();
	h->K = 80;
	h->a_d = h->a_s = h->a_bias = 1;
	h->b_d = h->b_s = h->b_bias = 1;
	h->a_n = h->b_n = 1e-1;

	h->a_pi = 1;
	h->b_pi = 1200;
	cudaMalloc(&d, sizeof(_dlConfig));
}

void DLConfig::apply()
{
	cudaMemcpy(d, h, sizeof(_dlConfig), cudaMemcpyDeviceToHost);
}

DLConfig::DLConfig(){
	free(h);
	cudaFree(d);
}