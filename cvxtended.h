//
//  cvxtended.cpp
//  remedi2
//
//  Created by Albert Clapés on 02/04/14.
//
//

#ifndef __remedi2__cvxtended__
#define __remedi2__cvxtended__

#include <iostream>
#include <string>
#include <vector>
#include <list>

#include <opencv2/opencv.hpp>

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b);

namespace cvx
{
    
    // There are not functions in OpenCV letting you select a set of arbitrary
    // spare rows or columns and create from them a new and smaller cv::Mat
    
    // src < dst, indices index dst
    void setMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    void setMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    void setMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    
    // src > dst, indices index src
    void indexMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    void indexMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    void indexMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    cv::Mat indexMat(cv::Mat src, cv::Mat indices, bool logical = true);
    //    cv::Mat indexMat(cv::Mat src, cv::Mat indices, bool logical = true); // alternative implementation to copyMat
    //    void indexMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    //    void indexMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    //    void indexMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    
    // Wrappers for cv::collapse(...)
    void hmean(cv::Mat src, cv::Mat& mean); // row-wise mean
    void vmean(cv::Mat src, cv::Mat& mean); // column-wise mean
    
    void hist(cv::Mat src, int nbins, float min, float max, cv::Mat& hist);
    void hist(cv::Mat src, cv::Mat msk, int nbins, float min, float max, cv::Mat& hist);
    
    void cumsum(cv::Mat src, cv::Mat& dst);
    
    void linspace(double start, double end, int n, cv::Mat& m);
    void linspace(double start, double end, int n, std::vector<double>& v);
    cv::Mat linspace(double start, double end, int n);
    
    void linspace(int start, int end, cv::Mat& m);
    void linspace(int start, int end, std::vector<int>& v);
    cv::Mat linspace(int start, int end);
    
    cv::Mat powspace(int start, int end, int base);
    
    // Data-to-disk and disk-to-data nice functions (hide cv::FileStorage declarations)
    void load(std::string file, cv::Mat& mat, int format = cv::FileStorage::FORMAT_YAML);
    void save(std::string file, cv::Mat mat, int format = cv::FileStorage::FORMAT_YAML);
    
//    template<typename T>
//    cv::Mat matlabread(std::string file);
//    template<typename T>
//    void matlabread(std::string file, cv::Mat& mat);
    
    void computePCA(cv::Mat src, cv::PCA& pca, cv::Mat& dst, int flags = CV_PCA_DATA_AS_ROW, double variance = 1);
    
    cv::Mat standardize(cv::Mat m, int dim = 0);
    
    void fillErrorsWithMedian(cv::InputArray src, int size, cv::OutputArray dst);
    
    void dilate(cv::InputArray src, int size, cv::OutputArray dst);
    void erode(cv::InputArray src, int size, cv::OutputArray dst);
    void open(cv::InputArray src, int size, cv::OutputArray dst);
    void close(cv::InputArray src, int size, cv::OutputArray dst);
    
    float overlap(cv::InputArray src1, cv::InputArray src2);
    float contention(cv::InputArray src1, cv::InputArray src2);
    cv::Mat replicate(cv::Mat src, int times);
    void mean(cv::InputArray acc, cv::InputArray count, cv::OutputArray mean);
    void stddev(cv::InputArray sqdiffsacc, cv::InputArray count, cv::InputArray err, cv::OutputArray stddev);
    
    void match(cv::Mat src, cv::Mat queries, cv::Mat& matched, cv::Mat& pending);
    template<typename T>
    int match(cv::Mat m, cv::Mat query, bool debug = false);
    
    void sortIdx(cv::InputArray src, cv::OutputArray indices, int flags, cv::OutputArray dst = cv::noArray());
    
    void harmonicMean(cv::InputArray src, cv::OutputArray dst, int dim);
    
    template<typename T>
    void convert(cv::Mat mat, std::vector<std::vector<T> >& vv);
    template<typename T>
    void convert(cv::Mat mat, std::list<std::vector<T> >& lv);
    template<typename T>
    void convert(std::vector<std::vector<T> > vv, cv::Mat& mat);
    template<typename T>
    cv::Mat convert(std::vector<std::vector<T> > vv);
    
    void vconcat(std::vector<cv::Mat> array, cv::Mat& mat);
    void hconcat(std::vector<cv::Mat> array, cv::Mat& mat);
    
    cv::Point diffIdx(cv::Mat src1, cv::Mat src2);
    void unique(cv::Mat M, int dim, cv::Mat& U, cv::Mat& I);
    
    void findRowsToColsAssignation(cv::Mat scores, cv::Mat& matches);
    
    void combine(cv::InputArray src1, cv::InputArray src2, int dim, cv::OutputArray dst);
}

#endif /* defined(__remedi2__cvxtended__) */
