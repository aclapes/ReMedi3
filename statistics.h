//
//  StatTools.h
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#ifndef __remedi2__statistics__
#define __remedi2__statistics__

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#include "GridMat.h"

using namespace std;

// Builds an histogram of the values contained in a vector (or matrix)
void histogram(cv::Mat mat, int nbins, cv::Mat & hist);

// Create a column vector containing the numbers in the interval [a,b] shuffled randomly
cv::Mat shuffledVector(int a, int b, cv::RNG randGen);
cv::Mat shuffledVector(int a, int b);

// Create a column vector containing the numbers in the interval [0,n) shuffled randomly
cv::Mat shuffledVector(int n, cv::RNG randGen);
cv::Mat shuffledVector(int n);

// Create a vector of labels representing the k folds of n elements
void cvpartition(int n, int k, int seed, cv::Mat& partitions);
void cvpartition(cv::Mat labels, int k, int seed, cv::Mat& partitions); // stratified
//void cvpartition(GridMat labels, int k, int seed, GridMat& partitions); // stratified

// Mathematical function approximating a Gaussian function
double phi(double x);

// Sort a vector by unique values
void uniqueSortValues(vector<int> & values);

// Find unique values of a Mat and returns them sorted
void findUniqueValues(cv::Mat image, vector<int> & values);

// Find unique values of a vector and returns them sorted
void findUniqueValues(vector<int> v, vector<int> & values);

// Generate variations with repetition
template<typename T>
void variate(vector<vector<T > > list, vector<vector<T > >& variations);
template<typename T>
void _variate(vector<vector<T > > list, int idx, vector<T> v, vector<vector<T > >& variations);

template<typename T>
void variate(vector<vector<T > > list, cv::Mat& variations);
template<typename T>
void _variate(vector<vector<T > > list, int idx, cv::Mat v, cv::Mat& variations);

template<typename T>
void expandParameters(vector<vector<T> > params, vector<vector<T> >& expandedParams);
template<typename T>
void expandParameters(vector<vector<T> > params, int ncells, vector<vector<T> >& expandedParams);

template<typename T>
void expandParameters(vector<vector<T> > params, cv::Mat& expandedParams);

template<typename T>
void selectParameterCombination(vector<vector<T> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams);

//template<typename T>
//void selectBestParameterCombination(vector<vector<T> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams);

//template<typename T>
//void selectBestParameterCombination(GridMat goodnesses, vector<cv::Mat>& selectedParams);

float accuracy(cv::Mat actuals, cv::Mat predictions);
//void accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies);
//void accuracy(cv::Mat actuals, GridMat predictions, cv::Mat& accuracies);
//float accuracy(GridMat actuals, GridMat predictions);
//float accuracy(cv::Mat actuals, GridMat predictions);

void accuracy(cv::Mat actuals, cv::Mat predictions, cv::Mat partitions, cv::Mat& accuracies);
//void accuracy(cv::Mat actuals, GridMat predictions, cv::Mat partitions, GridMat& accuracies);

template<typename T>
void narrow(cv::Mat coarse, cv::Mat goodnesses, int steps, int* discrete, cv::Mat& narrow);

void computeConfidenceInterval(cv::Mat values, float* mean, float* confidence, float alpha = 0.05);
//void computeConfidenceInterval(GridMat values, cv::Mat& mean, cv::Mat& confidence, float alpha = 0.05);

float computeOverlap(int tp, int fn, int fp);
float computeF1Score(int tp, int fn, int fp);
void computeF1Score(cv::Mat tp, cv::Mat fn, cv::Mat fp, cv::Mat& fscore);

void writeParametersToFile(string file, vector<string> names, vector<vector<double> > values, bool append = false);

#endif /* defined(__remedi2__statistics__) */
