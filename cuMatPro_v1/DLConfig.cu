#include "DLConfig.cu.h"

DLConfig::DLConfig(){
	h = new _dlConfig();
	h->K = 80;
	h->a_d = h->a_s = h->a_bias = 1;
	h->b_d = h->b_s = h->b_bias = 1;
	h->a_n = h->b_n = 1e-1;

	h->a_pi = 1;
	h->b_pi = 1200;

	cudaMalloc(&d, sizeof(_dlConfig));
	apply();
}

DLConfig::~DLConfig(){
	free(h);
	cudaFree(d);
}

void DLConfig::apply()
{
	cudaMemcpy(d, h, sizeof(_dlConfig), cudaMemcpyDeviceToHost);
}

//void DLConfig::reflect()
//{
//	cudaMemcpy(h, d, sizeof(_dlConfig), cudaMemcpyDeviceToHost);
//}

