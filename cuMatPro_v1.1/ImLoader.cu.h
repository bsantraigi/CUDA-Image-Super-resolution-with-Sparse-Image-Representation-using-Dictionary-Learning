#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Utilities.cu.h"
#include "gpuMat.cu"

using namespace std;
using namespace cv;

#define del2D(mat,n) for(int i=0;i<n;i++){delete [] mat[i];}

class ImLoader
{
public:
	string folder, path;
	Size redSize;
	int patchSize, reduceTo;
	vector<int> imLocations;
	vector<int> rowList;
	vector<int> colList;

	int unpId = 0;
public:
	ImLoader();
	ImLoader(int reduceTo = 16, int patchSize = 8, string folder = "Faces_easy",
		string path = "D:/Users/Bishal Santra/Documents/MATLAB/MTP/neural_generative/caltech101/101_ObjectCategories/");
	~ImLoader();
	void GetDataMatrix(gpuMat<double>& dataMat, int totalImage2Data = 2);
	Mat LoadImage(string fpath, int reduceTo);
	int PatchImage(gpuMat<double> &dataMatrix, int from, Mat& image);
};

