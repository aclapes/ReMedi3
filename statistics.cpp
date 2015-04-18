//
//  StatTools.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#include "statistics.h"
#include "cvxtended.h"

#include <opencv2/opencv.hpp>

#include <math.h>
#include <fstream>
#include <set>

// Instantiation of template member functions
// -----------------------------------------------------------------------------
template void variate<int>(vector<vector<int > > list, vector<vector<int > >& variations);
template void variate<float>(vector<vector<float > > list, vector<vector<float > >& variations);
template void variate<double>(vector<vector<double > > list, vector<vector<double > >& variations);

template void expandParameters<int>(vector<vector<int> > params, vector<vector<int> >& expandedParams);
template void expandParameters<float>(vector<vector<float> > params, vector<vector<float> >& expandedParams);
template void expandParameters<double>(vector<vector<double> > params, vector<vector<double> >& expandedParams);

template void expandParameters<int>(vector<vector<int> > params, cv::Mat& expandedParams);
template void expandParameters<float>(vector<vector<float> > params, cv::Mat& expandedParams);
template void expandParameters<double>(vector<vector<double> > params, cv::Mat& expandedParams);

template void expandParameters<int>(vector<vector<int> > params, int ncells, vector<vector<int> >& expandedParams);
template void expandParameters<float>(vector<vector<float> > params, int ncells, vector<vector<float> >& expandedParams);
template void expandParameters<double>(vector<vector<double> > params, int ncells, vector<vector<double> >& expandedParams);

template void selectParameterCombination<int>(vector<vector<int> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams);
template void selectParameterCombination<float>(vector<vector<float> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams);
template void selectParameterCombination<double>(vector<vector<double> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams);

//template void selectBestParameterCombination<int>(vector<vector<int> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams);
//template void selectBestParameterCombination<float>(vector<vector<float> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams);
//template void selectBestParameterCombination<double>(vector<vector<double> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams);
//template void selectBestParameterCombination<double>(vector<vector<double> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams);

//template void selectBestParameterCombination<float>(GridMat expandedParams, vector<cv::Mat>& selectedParams);
//template void selectBestParameterCombination<double>(GridMat expandedParams, vector<cv::Mat>& selectedParams);

template void narrow<float>(cv::Mat coarse, cv::Mat goodnesses, int steps, int* discretes, cv::Mat& narrow);
template void narrow<double>(cv::Mat coarse, cv::Mat goodnesses, int steps, int* discretes, cv::Mat& narrow);

// -----------------------------------------------------------------------------


/**
 * Builds an histogram of the values contained in a vector (or matrix)
 */
void histogram(cv::Mat mat, int nbins, cv::Mat & hist)
{
    double minval, maxval;
    cv::minMaxIdx(mat, &minval, &maxval);
    
    if (nbins > (maxval-minval+1))
        return;
    
    // Create an histogram for the cell region of blurred intensity values
    int histSize[] = { (int) nbins };
    int channels[] = { 0 }; // 1 channel, number 0
    float tranges[] = { (float)minval, (float)maxval }; // thermal intensity values range: [0, 256)
    const float* ranges[] = { tranges };
    
    cv::calcHist(&mat, 1, channels, cv::noArray(), hist, 1, histSize, ranges, true, false);
}

/**
 * Create a column vector containing the numbers in the interval [a,b] shuffled randomly
 */
cv::Mat shuffledVector(int a, int b, cv::RNG randGen)
{
    cv::Mat vec (b-a+1, 1, cv::DataType<int>::type);
    for (int i = a; i <= b; i++)
    {
        vec.at<int>(i-a, 0) = i;
    }
    
    randShuffle(vec, 10*(b-a+1), &randGen);
    
    return vec;
}

/**
 * Create a column vector containing the numbers in the interval [a,b] shuffled randomly
 */
cv::Mat shuffledVector(int a, int b)
{
    cv::Mat vec (b-a+1, 1, cv::DataType<int>::type);
    for (int i = a; i <= b; i++)
    {
        vec.at<int>(i-a, 0) = i;
    }
    
    randShuffle(vec, 10*(b-a+1), NULL);
    
    return vec;
}

/**
 * Create a column vector containing the numbers in the interval [0,n) shuffled randomly
 */
cv::Mat shuffledVector(int n, cv::RNG randGen)
{
    return shuffledVector(0, n-1, randGen);
}

cv::Mat shuffledVector(int n)
{
    return shuffledVector(0, n-1);
}

/**
 * Create a vector of labels representing the k folds of n elements
 */
void cvpartition(int n, int k, int seed, cv::Mat& partitions)
{
    int foldElems = std::floorf(n/k);
    int extraElems = n - (k * std::floorf(n/k));

    cv::Mat indices = shuffledVector(n, cv::RNG(seed));
    
    partitions.release();
    partitions.create(indices.rows, indices.cols, cv::DataType<int>::type);

    cv::Mat tmp = shuffledVector(k);
    cv::Mat tmp2 = tmp(cv::Range(0, extraElems), cv::Range::all());

    unsigned int c = 0;
    unsigned int i, j;
    for (i = 0; i < k; i++)
    {
        int ifoldElems;
        if (cv::sum(tmp2 == i).val[0] > 0)
            ifoldElems = (foldElems + 1);
        else
            ifoldElems = foldElems;
        
        for (j = 0; j < ifoldElems; j++)
        {
            partitions.at<int>(indices.at<int>(c+j,0),0) = i;
        }
        
        c += ifoldElems;
    }
}


//void cvpartition(GridMat gclasses, int k, int seed, GridMat& gpartitions)
//{
//    gpartitions.release();
//    gpartitions.create(gclasses.crows(), gclasses.ccols());
//    for (int i = 0; i < gclasses.crows(); i++) for (int j = 0; j < gclasses.ccols(); j++)
//    {
//        cvpartition(gclasses.at(i,j), k, seed, gpartitions.at(i,j));
//    }
//}

/**
 * Create a vector of labels representing the k folds of n elements (stratified)
 */
void cvpartition(cv::Mat classes, int k, int seed, cv::Mat& partitions)
{
    double minVal, maxVal; // labels do not need to be between [0, #classes - 1]
    cv::minMaxIdx(classes, &minVal, &maxVal); // but, suppose continuous numeration of labels
    
    // separate the indices in a different vector for each class

    std::vector<std::vector<int> > classesIndices(maxVal - minVal + 1);
    int m = (classes.rows > 1) ? classes.rows : classes.cols;
    for (int i = 0; i < m; i++)
    {
        int l = (classes.rows > 1) ? classes.at<int>(i,0) : classes.at<int>(0,i);
        classesIndices[l - minVal].push_back(i);
    }
    
    // perform partitions separately in the classes' indices vectors
    // and then merge the separate partitions into one
    
    std::vector<cv::Mat> classesPartitions(maxVal - minVal + 1);
    
    partitions.release();
    partitions.create(classes.rows, classes.cols, cv::DataType<int>::type);
    
    for (int i = 0; i < classesIndices.size(); i++)
    {
        cvpartition(classesIndices[i].size(), k, seed, classesPartitions[i]);
        for (int j = 0; j < classesIndices[i].size(); j++)
        {
            (partitions.rows > 1) ?
                partitions.at<int>(classesIndices[i][j], 0) = classesPartitions[i].at<int>(j,0) :
                partitions.at<int>(0, classesIndices[i][j]) = classesPartitions[i].at<int>(j,0);
        }
    }
}


/**
 * Mathematical function approximating a Gaussian function
 */
double phi(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
    
    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);
    
    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
    
    return 0.5*(1.0 + sign*y);
}

/**
 * Converts Mat to Vector
 */
void matToVector(cv::Mat image, vector<int> & values)
{
    cv::Mat_<uchar>::iterator it_start = image.begin<uchar>();
    cv::Mat_<uchar>::iterator it_end = image.end<uchar>();
    
    for(; it_start != it_end; ++it_start) {
        values.push_back(*it_start);
    }
}

/**
 * Sort a vector by unique values
 */
void uniqueSortValues(vector<int> & values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

/**
 * Find unique values of a Mat and returns them sorted
 */
void findUniqueValues(cv::Mat image, vector<int> & values) {
    matToVector(image, values);
    uniqueSortValues(values);
}

/**
 * Find unique values of a vector and returns them sorted
 */
void findUniqueValues(vector<int> v, vector<int> & values) {
    uniqueSortValues(v);
    values = v;
}

template<typename T>
void variate(vector<vector<T > > list, vector<vector<T > >& variations)
{
    vector<T> v(list.size()); // empty
    _variate(list, 0, v, variations);
}

template<typename T>
void _variate(vector<vector<T > > list, int idx, vector<T> v, vector<vector<T > >& variations)
{
    if (idx == list.size())
    {
        return;
    }
    else
    {
        for (int i = 0; i < list[idx].size(); i++)
        {
            v[idx] = list[idx][i];
            _variate(list, idx+1, v, variations);
            if (idx == list.size() - 1)
            {
                variations.push_back(v);
                //v.erase(v.begin()+idx);
            }
        }
    }
    
}

template<typename T>
void variate(vector<vector<T > > list, cv::Mat& variations)
{
    cv::Mat v(1, list.size(), cv::DataType<T>::type);
    variations.create(0, list.size(), cv::DataType<T>::type);
    _variate(list, 0, v, variations);
}

template<typename T>
void _variate(vector<vector<T > > list, int idx, cv::Mat v, cv::Mat& variations)
{
    if (idx == list.size())
    {
        return;
    }
    else
    {
        for (int i = 0; i < list[idx].size(); i++)
        {
            v.at<T>(0,idx) = list[idx][i];
            _variate(list, idx+1, v, variations);
            if (idx == list.size() - 1)
            {
                variations.push_back(v);
            }
        }
    }
    
}



template<typename T>
void expandParameters(vector<vector<T> > params, vector<vector<T> >& expandedParams)
{
    expandedParams.clear();
    
    variate(params, expandedParams);
}

template<typename T>
void expandParameters(vector<vector<T> > params, cv::Mat& expandedParams)
{
    expandedParams.release();

    variate(params, expandedParams);
}


template<typename T>
void expandParameters(vector<vector<T> > params, int ncells, vector<vector<T> >& gridExpandedParams)
{
    gridExpandedParams.clear();

    vector<vector<T> > cellExpandedParams;
    variate(params, cellExpandedParams);
    
    // Create and expand a list of indices, used to index the cellExpandedParams
    
    vector<int> indices(cellExpandedParams.size());
    
    for (int i = 0; i < cellExpandedParams.size(); i++)
        indices[i] = i;
    
    vector<vector<int> > listsOfIndices(ncells);
    for (int i = 0; i < ncells; i++)
        listsOfIndices[i] = indices;
    
    vector<vector<int> > expandedIndices;
    variate(listsOfIndices, expandedIndices);
    
    //    // debug
    //    for (int i = 0; i < expandedIndices.size(); i++)
    //    {
    //        cv::Mat m (expandedIndices[i].size(), 1, cv::DataType<int>::type, expandedIndices[i].data());
    //        cout << m << endl;
    //    }
    //    //
    
    // Create the grid's combinations' list of parameters
    
    gridExpandedParams.clear();
    gridExpandedParams.resize(expandedIndices.size());
    
    for (int i = 0; i < expandedIndices.size(); i++)
    {
        for (int j = 0; j < expandedIndices[i].size(); j++)
        {
            vector<T> combination = cellExpandedParams[expandedIndices[i][j]];
            for (int k = 0; k < params.size(); k++)
            {
                gridExpandedParams[i].push_back(combination[k]);
            }
        }
    }
}


template<typename T>
void selectParameterCombination(vector<vector<T> > expandedParams, int hp, int wp,
                                int nparams, int idx, vector<cv::Mat>& selectedParams)
{
    selectedParams.clear();
    
    for (int k = 0; k < nparams; k++)
        selectedParams.push_back(cv::Mat(hp, wp, cv::DataType<T>::type));
    
    vector<T> lineParams = expandedParams[idx];
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        int l = i * wp + j;
        for (int k = 0; k < nparams; k++)
        {
            selectedParams[k].at<T>(i,j) = lineParams[l * nparams + k];
        }
    }
    
    // debug
    //    for (int k = 0; k < nparams; k++)
    //        cout << selectedParams[k] << endl;
}


//template<typename T>
//void selectBestParameterCombination(vector<vector<T> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams)
//{
//    selectedParams.clear();
//    
//    for (int k = 0; k < nparams; k++)
//        selectedParams.push_back(cv::Mat(hp, wp, cv::DataType<T>::type));
//    
//    GridMat gargmax;
//    goodnesses.argmax<T>(gargmax);
//    
//    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
//    {
//        int rowIdx = gargmax.at<T>(i,j,0,0); // maxrow index
//        
//        vector<T> lineParams = expandedParams[rowIdx];
//        
//        int l = i * wp + j;
//        for (int k = 0; k < nparams; k++)
//        {
//            selectedParams[k].at<T>(i,j) = lineParams[l * nparams + k];
//        }
//    }
//    
//    // debug
//    //    for (int k = 0; k < nparams; k++)
//    //        cout << selectedParams[k] << endl;
//}

//template<typename T>
//void selectBestParameterCombination(vector<vector<T> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams)
//{
//    selectedParams.clear();
//    
//    for (int k = 0; k < nparams; k++)
//        selectedParams.push_back(cv::Mat(hp, wp, cv::DataType<T>::type));
//    
//    GridMat gargmax;
//    goodnesses.argmax(gargmax);
//    
//    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
//    {
//        int rowIdx = gargmax.at(i,j).at<int>(0,0); // maxrow index
//        
//        vector<T> lineParams = expandedParams[rowIdx];
//        
//        for (int k = 0; k < lineParams.size(); k++)
//        {
//            selectedParams[k].at<T>(i,j) = lineParams[k];
//        }
//    }
//    
//    // debug
//    //    for (int k = 0; k < nparams; k++)
//    //        cout << selectedParams[k] << endl;
//}

//template<typename T>
//void selectBestParameterCombination(GridMat parameters, vector<cv::Mat>& selectedParams)
//{
//    int numOfParameters = parameters.at(0,0).cols - 1;
//    
//    selectedParams.clear();
//    for (int i = 0; i < numOfParameters; i++)
//        selectedParams.push_back(cv::Mat(parameters.crows(), parameters.ccols(), cv::DataType<T>::type));
//
//    // Find best parameters (using goodnesses)
//    double minVal, maxVal;
//    cv::Point worst, best;
//    
//    for (int i = 0; i < parameters.crows(); i++) for (int j = 0; j < parameters.ccols(); j++)
//    {
//        cv::minMaxLoc(parameters.at(i,j).col(parameters.at(i,j).cols - 1), &minVal, &maxVal, &worst, &best);
//
//        for (int p = 0; p < numOfParameters; p++)
//            selectedParams[p].at<T>(i,j) = parameters.at(i,j).row(best.y).at<T>(0,p);
//    }
//}

//template<typename T>
//void selectBestParameterCombination(GridMat goodnesses, vector<cv::Mat>& selectedParams)
//{
//    selectedParams.clear();
//    
//    for (int k = 0; k < goodnesses.at(0,0).cols - 1; k++)
//        selectedParams.push_back(cv::Mat(goodnesses.crows(), goodnesses.ccols(), cv::DataType<T>::type));
//    
//    for (int i = 0; i < goodnesses.crows(); i++) for (int j = 0; j < goodnesses.ccols(); j++)
//    {
//        double minVal, maxVal;
//        cv::Point min, max;
//        cv::minMaxLoc(goodnesses.at(i,j).col(goodnesses.at(i,j).cols - 1),
//                      &minVal, &maxVal, &min, &max);
//        
//        cv::Mat rowParams = goodnesses.at(i,j).row(max.y);
//        for (int k = 0; k < rowParams.cols - 1; k++)
//        {
//            selectedParams[k].at<T>(i,j) = rowParams.at<T>(0,k);
//        }
//    }
//}

float accuracy(cv::Mat actuals, cv::Mat predictions)
{
    actuals.convertTo(actuals, cv::DataType<int>::type);
    predictions.convertTo(predictions, cv::DataType<int>::type);
    
    cv::Mat negatives  = (actuals == 0 | actuals == -1);
    cv::Mat positives = (actuals == 1);
    cv::Mat hits = (actuals == predictions);
    
    int nneg = cv::sum(negatives).val[0];
    int npos = cv::sum(positives).val[0];
    
    float negAcc = (nneg > 0) ? (cv::sum(negatives & hits).val[0] / nneg) : 0;
    float posAcc = (npos > 0) ? (cv::sum(positives & hits).val[0] / npos) : 0;
    
    return (negAcc + posAcc) / 2.0;
}

//void accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies)
//{
//    accuracies.create(predictions.crows(), predictions.ccols(), cv::DataType<float>::type);
//    
//    for (int i = 0; i < predictions.crows(); i++) for (int j = 0; j < predictions.ccols(); j++)
//    {
//        cv::Mat cellActuals;
//        cv::Mat cellPredictions;
//        
//        actuals.at(i,j).convertTo(cellActuals, cv::DataType<int>::type);
//        predictions.at(i,j).convertTo(cellPredictions, cv::DataType<int>::type);
//        
//        cv::Mat objects  = (cellActuals == 0);
//        cv::Mat subjects = (cellActuals == 1);
//        cv::Mat hits = (cellActuals == cellPredictions);
//        
//        int nobj = cv::sum(objects).val[0];
//        int nsbj = cv::sum(subjects).val[0];
//        
//        int objHits = cv::sum(objects & hits).val[0];
//        int sbjHits = cv::sum(subjects & hits).val[0];
//        
//        accuracies.at<float>(i,j) = (float(objHits)/nobj + float(sbjHits)/nsbj) / 2;
//    }
//}

//void accuracy(cv::Mat actuals, GridMat predictions, cv::Mat& accuracies)
//{
//    GridMat gactuals;
//    gactuals.setTo(actuals);
//    
//    accuracy(gactuals, predictions, accuracies);
//}

//float accuracy(GridMat actuals, GridMat predictions)
//{
//    cv::Mat accuracies;
//    
//    accuracy(actuals, predictions, accuracies);
//    
//    return cv::mean(accuracies).val[0];
//}

//float accuracy(cv::Mat actuals, GridMat predictions)
//{
//    GridMat gactuals;
//    gactuals.setTo(actuals);
//    
//    return accuracy(gactuals, predictions);
//}

void accuracy(cv::Mat actuals, cv::Mat predictions, cv::Mat partitions, cv::Mat& accuracies)
{
    cv::Mat aux = partitions.t();
    std::set<int> set (aux.ptr<int>(0), aux.ptr<int>(0) + aux.cols);
    std::vector<int> labels (set.begin(), set.end());
    
    accuracies.create(labels.size(), 1, cv::DataType<float>::type);
    for (int k = 0; k < labels.size(); k++)
    {
        accuracies.at<float>(k,0) = accuracy(cvx::indexMat(actuals, partitions == k), cvx::indexMat(predictions, partitions == k));
    }
}

//void accuracy(cv::Mat actuals, GridMat predictions, cv::Mat partitions, GridMat& accuracies)
//{
//    cv::Mat aux = partitions.t();
//    std::set<int> set (aux.ptr<int>(0), aux.ptr<int>(0) + aux.cols);
//    std::vector<int> labels (set.begin(), set.end());
//    
//    accuracies.create(predictions.crows(), predictions.ccols());
//    for (int i = 0; i < predictions.crows(); i++) for (int j = 0; j < predictions.ccols(); j++)
//    {
//        cv::Mat cellAccuracies (labels.size(), 1, cv::DataType<float>::type);
//        for (int k = 0; k < labels.size(); k++)
//        {
//            cellAccuracies.at<float>(k,0) = accuracy(cvx::indexMat(actuals, partitions == k),
//                                                     cvx::indexMat(predictions.at(i,j), partitions == k));
//        }
//        accuracies.assign(cellAccuracies, i, j);
//    }
//}

template<typename T>
void narrow(cv::Mat coarse, cv::Mat goodnesses, int steps, int* discretes, cv::Mat& narrow)
{
    vector<vector<T> > parameters;
    for (int p = 0; p < coarse.cols; p++)
    {
        cv::Mat row = coarse.col(p).t();
        std::set<T> set (row.ptr<T>(0), row.ptr<T>(0) + row.cols);
        std::vector<T> parameter (set.begin(), set.end());
        parameters.push_back(parameter);
    }
    
    // Find best parameters (using goodnesses)
    double minVal, maxVal;
    cv::Point worst, best;
    cv::minMaxLoc(goodnesses.col(goodnesses.cols-1), &minVal, &maxVal, &worst, &best);
    
    // From linear index to point in the space of combinations
    int linIdxBest = best.y; // linear idx
    cv::Mat point (1, coarse.cols, cv::DataType<int>::type);
    
    float divisor = 1;
    for (int p = coarse.cols - 1; p >= 0; p--)
    {
        point.at<int>(0,p) = int(std::floor(linIdxBest/divisor)) % parameters[p].size();
        divisor *= parameters[p].size();
    }
    
    vector<vector<T> > nwparameters;
    for (int i = 0; i < parameters.size(); i++)
    {
        int coord = point.at<int>(0,i);
        
        std::vector<double> aux;
        
        if (coord == 0)
            cvx::linspace((double) parameters[i][coord], (double) parameters[i][coord+2], steps, aux);
        else if (coord == parameters[i].size() - 1)
            cvx::linspace((double) parameters[i][coord-2], (double) parameters[i][coord], steps, aux);
        else
            cvx::linspace((double) parameters[i][coord-1], (double) parameters[i][coord+1], steps, aux);
        
        if (discretes[i])
        {
            std::vector<int> iaux (aux.begin(), aux.end()); // keep unique values
            std::set<int> s (iaux.begin(), iaux.end());
            
            std::vector<T> v (s.begin(), s.end());
            nwparameters.push_back(v);
        }
        else
        {
            std::vector<T> v (aux.begin(), aux.end());
            nwparameters.push_back(v);
        }
    }
    
    expandParameters(nwparameters, narrow);
}

void computeConfidenceInterval(cv::Mat values, float* mean, float* confidence, float alpha)
{
    cv::Scalar _mean, stddev;
    cv::meanStdDev(values, _mean, stddev);
    
    float pval;
    if (alpha == 0.20) pval = 1.28;
    else if (alpha == 0.15) pval = 1.44;
    else if (alpha == 0.10) pval = 1.65;
    else if (alpha == 0.05) pval = 1.96;
    else if (alpha == 0.01) pval = 2.57;
    else pval = 1.96; // alpha == 0.05
    
    *mean = _mean.val[0];
    *confidence = pval * (stddev.val[0] / sqrtf(values.rows));
}

//void computeConfidenceInterval(GridMat values, cv::Mat& means, cv::Mat& confidences, float alpha)
//{
//    means.create(values.crows(), values.ccols(), cv::DataType<float>::type);
//    confidences.create(values.crows(), values.ccols(), cv::DataType<float>::type);
//
//    for (int i = 0; i < values.crows(); i++) for (int j = 0; j < values.ccols(); j++)
//    {
//        float mean, confidence;
//        computeConfidenceInterval(values.at(i,j), &mean, &confidence, alpha);
//        means.at<float>(i,j) = mean;
//        confidences.at<float>(i,j) = confidence;
//    }
//}

float computeOverlap(int tp, int fn, int fp)
{
    return ((float) tp) / (tp + fp + fn);
}

float computeF1Score(int tp, int fn, int fp)
{
    return (2.f * tp) / (2.f * tp + fp + fn);
}

void computeF1Score(cv::Mat tp, cv::Mat fn, cv::Mat fp, cv::Mat& fscore)
{
    cv::Mat tpF, fnF, fpF;
    tp.convertTo(tpF, cv::DataType<float>::type);
    fn.convertTo(fnF, cv::DataType<float>::type);
    fp.convertTo(fpF, cv::DataType<float>::type);

    fscore = (2.f * tpF) / ((2.f * tpF) + fpF + fnF);
}

void writeParametersToFile(string file, vector<string> names, vector<vector<double> > values, bool append)
{
    assert (names.size() == values.size());
    
    std::ofstream ofs;
    
    if (!append)
        ofs.open(file, std::ofstream::out);
    else
    {
        ofs.open(file, std::ofstream::app);
        ofs << endl;
    }
        
    for (int i = 0; i < names.size(); i++)
    {
        ofs << names[i] << " : [";
        for (int j = 0; j < values[i].size(); j++)
        {
            if (j > 0) ofs << ",";
            ofs << values[i][j];
        }
        
        ofs << "]" << endl;
    }
    
    ofs.close();
}