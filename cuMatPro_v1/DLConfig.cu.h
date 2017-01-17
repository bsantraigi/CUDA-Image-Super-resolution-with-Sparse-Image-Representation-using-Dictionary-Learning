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