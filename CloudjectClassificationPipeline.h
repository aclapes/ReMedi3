//
//  CloudjectClassificationPipeline.h
//  remedi3
//
//  Created by Albert Clapés on 12/03/15.
//
//

#ifndef __remedi3__CloudjectClassificationPipeline__
#define __remedi3__CloudjectClassificationPipeline__

#include <stdio.h>
#include <list>
#include <vector>

#include "Cloudject.hpp"

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

enum type {CCPIPELINE_NORM_MINMAX, CCPIPELINE_NORM_STD,
    CCPIPELINE_NORM_L1, CCPIPELINE_NORM_L2};

template<typename T>
class CloudjectClassificationPipelineBase
{
public:
    CloudjectClassificationPipelineBase();
    CloudjectClassificationPipelineBase(const CloudjectClassificationPipelineBase& rhs);
    CloudjectClassificationPipelineBase& operator=(const CloudjectClassificationPipelineBase& rhs);
    ~CloudjectClassificationPipelineBase();
    
    void setInputCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::vector<const char*> categories);
    
    void setDimRedVariance(double var);
    void setDimRedNumFeatures(int n);
    
    void setNormalization(int normType);
    void setClassifierType(std::string classifierName);

    void setValidation(int numFolds);
    void setClassifierValidationParameters(std::vector<std::vector<float> > classifierParams);
    
protected:

    boost::shared_ptr<std::list<Cloudject::Ptr> > m_InputCloudjects;
    std::vector<const char*> m_Categories;
    
    int m_NumFolds;
    cv::Mat m_Partitions;
    
    int m_NormType;
    cv::vector<cv::Mat> m_NormParams;
    
    double m_DimRedVariance;
    int m_DimRedNumFeatures;
    cv::PCA m_PCA;
    
    std::string m_ClassifierType;
    std::vector<std::vector<float> > m_ClassifierValParams;
    std::vector<std::vector<float> > m_ClassifierParams;
    std::vector<void*> m_ClassifierModels;
    
    float m_BestValPerformance;
    
    void getLabels(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::vector<const char*> categories, cv::Mat& Y);
    void createValidationPartitions(cv::Mat Y, int numFolds, cv::Mat& partitions);
    
    float validate(cv::Mat X, cv::Mat Y);
    void train(cv::Mat X, cv::Mat Y);
    void predict(cv::Mat X, cv::Mat& P, cv::Mat& F);
    
    std::vector<float> evaluate(cv::Mat Y, cv::Mat F);
    
private:
    void normalize(cv::Mat X, cv::Mat& Xn, std::vector<cv::Mat>& parameters); // train
    void normalize(cv::Mat X, std::vector<cv::Mat> parameters, cv::Mat& Xn); // test
//    void reduce(cv::Mat M, cv::Mat& Mr, cv::Mat& eigenvectors, cv::Mat& centers); // train
//    void reduce(cv::Mat M, cv::Mat eigenvectors, cv::Mat centers, cv::Mat& Mr); // test
    
    void getTrainingFold(cv::Mat X, cv::Mat Y, int t, cv::Mat& XTr, cv::Mat& YTr);
    void getTestFold(cv::Mat X, cv::Mat Y, int t, cv::Mat& XTe, cv::Mat& YTe);
};

template<typename T>
class CloudjectClassificationPipeline : public CloudjectClassificationPipelineBase<T>
{};

template<>
class CloudjectClassificationPipeline<pcl::PFHRGBSignature250> : public CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>
{
public:
    CloudjectClassificationPipeline();
    CloudjectClassificationPipeline(const CloudjectClassificationPipeline& rhs);
    CloudjectClassificationPipeline& operator=(const CloudjectClassificationPipeline& rhs);
    
    void setPointSamplingRate(double rate);
    void setQuantizationValidationParameters(std::vector<int> qs);
    float validate();
    std::vector<float> getQuantizationValidationPerformances();
    
    int getQuantizationParameter();
    
    void train();
    
    std::vector<float> predict(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::list<std::vector<int> >& predictions, std::list<std::vector<float> >& distsToMargin);
    std::vector<int> predict(Cloudject::Ptr cloudject, std::vector<float>& distsToMargin);
    
    typedef boost::shared_ptr<CloudjectClassificationPipeline<pcl::PFHRGBSignature250> > Ptr;

private:
    int m_q; // num of quantization vectors
    cv::vector<int> m_qValParam;
    cv::vector<float> m_qValPerfs;
    
    cv::Mat m_C; // the q vectors actually
        
    void bowSampleFromCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, int q, cv::Mat& W, cv::Mat& Q);
    void bowSampleFromCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, cv::Mat Q, cv::Mat& W);
};

#endif /* defined(__remedi3__CloudjectClassificationPipeline__) */
