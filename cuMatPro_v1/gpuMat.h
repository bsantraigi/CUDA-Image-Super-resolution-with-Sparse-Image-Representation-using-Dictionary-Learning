#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

template <typename T>
class gpuMat
{
public:
	T* h_elems = nullptr;
	T* d_elems = nullptr;
	int rows, cols;

	gpuMat(int rows, int cols);
	~gpuMat();
	T& operator()(int row, int col);
	void copy2Device();
	void copy2Host();
};

template <typename T>
gpuMat<T>::gpuMat(int rows, int cols)
{
	this->rows = rows;
	this->cols = cols;
	h_elems = new T[rows*cols];
	cudaMalloc(&d_elems, rows*cols*sizeof(double));
}

template <typename T>
gpuMat<T>::~gpuMat()
{
}

template <typename T>
T& gpuMat<T>::operator()(int row, int col)
{
	return h_elems[col*rows + row];
}

template <typename T>
void gpuMat<T>::copy2Device()
{
	cudaMemcpy(d_elems, h_elems, rows*cols*sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void gpuMat<T>::copy2Host()
{
	cudaMemcpy(h_elems, d_elems, rows*cols*sizeof(T), cudaMemcpyDeviceToHost);
}