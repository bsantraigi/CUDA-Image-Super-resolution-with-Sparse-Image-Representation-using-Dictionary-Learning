#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define pn(x) printf("%5.2f ", (double)x)

using namespace std;

template <typename T>
class gpuMat
{
	bool blank = true;
public:
	T* h_elems = nullptr;
	T* d_elems = nullptr;
	int rows, cols;

	gpuMat();
	gpuMat(int rows, int cols);
	void create(int rows, int cols);
	~gpuMat();
	T& operator()(int row, int col = 0);
	void print(bool start = true);
	void copy2Device();
	void copy2Host();
};

template <typename T>
gpuMat<T>::gpuMat()
{
	blank = true;
}

template <typename T>
gpuMat<T>::gpuMat(int rows, int cols)
{
	if (!blank){
		delete[] h_elems;
		cudaFree(d_elems);
	}
	blank = false;	
	this->rows = rows;
	this->cols = cols;
	h_elems = new T[rows*cols];
	cudaError_t err = cudaMalloc(&d_elems, rows*cols*sizeof(double));
	if (err != cudaSuccess){
		cout << "[gpuMat::ctor]Memory allocation on GPU failed." << endl;
	}
}

template <typename T>
void gpuMat<T>::create(int rows, int cols)
{
	if (!blank){
		delete[] h_elems;
		cudaFree(d_elems);
	}
	blank = false;
	this->rows = rows;
	this->cols = cols;
	h_elems = new T[rows*cols];
	cudaError_t err = cudaMalloc(&d_elems, rows*cols*sizeof(double));
	if (err != cudaSuccess){
		cout << "[gpuMat::ctor]Memory allocation on GPU failed." << endl;
	}
}

template <typename T>
gpuMat<T>::~gpuMat()
{
	if (!blank){
		cout << "[gpuMat::dtor]Destroying gpuMat[auto]" << endl;
		delete[] h_elems;
		cudaFree(d_elems);
	}
	else{
		cout << "[gpuMat::dtor] object was blank" << endl;
	}

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

template <typename T>
void gpuMat<T>::print(bool start)
{
	cout << endl;
	cout << start << " <- start" << endl;
	if (start){
		for (int i = 0; i < min(10, rows); i++)
		{
			for (int j = 0; j < min(10, cols); j++)
			{
				pn((*this)(i, j));
			}
			cout << endl;
		}
	}
	else{
		for (int i = max(0, rows - 10); i < rows; i++)
		{
			for (int j = max(10, cols - 10); j < cols; j++)
			{
				pn((*this)(i, j));
			}
			cout << endl;
		}
	}
}