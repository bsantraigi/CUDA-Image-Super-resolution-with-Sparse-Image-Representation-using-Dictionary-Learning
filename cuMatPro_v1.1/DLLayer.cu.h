//#pragma once
//
//// Standard/CUDA Includes
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <curand.h>
//#include <curand_kernel.h>
//#include <iostream>
//
//// User defined Libraries
//
//using namespace std;
//
///*
//DATA STRUCTURES
//*/
//typedef struct {
//	double gam_d, gam_s, gam_n, gam_bias;
//} _modelParams;
//
//typedef struct{
//	int K;
//	double a_d, a_s, a_bias, a_n;
//	double b_d, b_s, b_bias, b_n;
//	double a_pi, b_pi;
//} _dlConfig;
//
///*
//CLASS DLLayer
//*/
//class DLLayer
//{
//public:
//	// Model params
//	_modelParams *h_params, *d_params;
//	curandState_t *localState;
//
//	// DLConfig: Layer cofigurations
//	_dlConfig *h_dlConfig;
//	_dlConfig *d_dlConfig;
//
//	void Init();
//	void reflect();
//
//	DLLayer();
//	~DLLayer();
//};