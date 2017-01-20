#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define pn(x) printf("%6.3f ", (double)x)

using namespace std;

template <typename T>
class gpuMat
{
	bool blank = true;
public:
	T* h_elems = nullptr;
	T* d_elems = nullptr;
	int rows, cols;
	int2 *d_size;

	gpuMat();
	gpuMat(int rows, int cols);
	void create(int rows, int cols);
	~gpuMat();
	T& operator()(int row, int col = 0);
	void print(bool start = true);
	void copy2Device();
	void copy2Host();
	void ToFile(string fileName);
private:
	void SetSize();
};

template <typename T>
gpuMat<T>::gpuMat()
{
	blank = true;
}

template <typename T>
void gpuMat<T>::SetSize()
{
	// Initialize d_size
	int2 h_size;
	h_size.x = cols;
	h_size.y = rows;
	cudaMalloc(&d_size, sizeof(int2));
	cudaMemcpy(d_size, &h_size, sizeof(int2), cudaMemcpyHostToDevice);
}
template <typename T>
gpuMat<T>::gpuMat(int rows, int cols)
{
	create(rows, cols);
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

	SetSize();
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
	cudaFree(d_size);
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
	// set start = false if you want to print the bottom-right corner
	// of a large matrix
	cout << endl;
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

template <typename T>
void gpuMat<T>::ToFile(string filename)
{
	this->copy2Host();
	FILE *fh;
	fh = fopen(filename.c_str(), "w");
	if (fh != NULL){
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				fprintf(fh, "%6.2f", (*this)(i, j));
				//os << (*this)(i, j) << ",";
				//os << j*rows + i << ",";
				//os << "[" << j*rows + i  << "]" << h_elems[j*rows + i] << ",";
			}
			//os << endl;
			fprintf(fh, "\n");
		}
		fclose(fh);
	}
	else{
		cout << filename << " failed to open..." << endl;
	}
}