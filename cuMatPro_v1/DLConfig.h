#pragma once
class DLConfig
{
public:
	int K;
	double a_d, a_s, a_bias, a_n;
	double b_d, b_s, b_bias, b_n;
	double a_pi, b_pi;

	DLConfig();
	~DLConfig();
};

