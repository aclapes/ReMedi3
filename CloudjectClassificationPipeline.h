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
    boost::shared_ptr<std::list<Cloudject::Ptr> > getInputCloudjects();
    
    void setDimRedVariance(double var);
    void setDimRedNumFeatures(int n);
    
    void setNormalization(int normType);
    void setClassifierType(std::string classifierName);

    void setValidation(int numFolds);
    void setClassifierValidationParameters(std::vector<std::vector<float> > classifierParams);

    void setClassifierParameters(std::vector<std::vector<float> > classifierParams);

protected:

    boost::shared_ptr<std::list<Cloudject::Ptr> > m_InputCloudjects;
    std::vector<const char*> m_Categories;
    
    int m_NumFolds;
    cv::Mat m_Partitions;
    
    int m_NormType;
    cv::vector<cv::Mat> m_NormParams;
    
    double m_DimRedVariance;
    int m_DimRedNumFeatures;
    
    std::string m_ClassifierType;
    std::vector<std::vector<float> > m_ClassifierValParams;
    std::vector<std::vector<float> > m_ClassifierParams;
    std::vector<void*> m_ClassifierModels;
    
    float m_BestValPerformance;
    
    void getLabels(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::vector<const char*> categories, cv::Mat& Y);
    void createValidationPartitions(cv::Mat Y, int numFolds, cv::Mat& partitions);
    
    void validate(cv::Mat XTr, cv::Mat XTe, cv::Mat YTr, cv::Mat YTe);
    std::vector<std::vector<float> > getValidationParametersCombinations();
    cv::Mat getValidationPerformances();
    
    void train(cv::Mat X, cv::Mat Y);
    void predict(cv::Mat X, cv::Mat& P, cv::Mat& F);
    
    std::vector<float> evaluate(cv::Mat Y, cv::Mat F);
    
    void normalize(cv::Mat X, cv::Mat& Xn, std::vector<cv::Mat>& parameters); // train
    void normalize(cv::Mat X, std::vector<cv::Mat> parameters, cv::Mat& Xn); // test
    
    void reduce(cv::Mat X, cv::Mat& Xr, cv::PCA& pca); // train
    void reduce(cv::Mat X, cv::PCA pca, cv::Mat& Xr); // test
    
    void getTrainingCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, int t, std::list<Cloudject::Ptr>& cloudjectsTr);
    void getTestCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, int t, std::list<Cloudject::Ptr>& cloudjectsTe);
    
private:
    std::vector<std::vector<float> > m_ClassifiersValParamCombs;
    cv::Mat m_ClassifiersValPerfs; // Validation parameters [P]erformance
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
    void setQuantizationParameter(int q);
    
    void setGlobalQuantization(bool bGlobalQuantization = true);

    void setCentersStratification(bool bCentersStratification = true);
    void setPerCategoryReduction(bool bPerCategoryRed = true);
    
    void train();
    
    std::vector<float> predict(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::list<std::vector<int> >& predictions, std::list<std::vector<float> >& distsToMargin);
    std::vector<int> predict(Cloudject::Ptr cloudject, std::vector<float>& distsToMargin);
    
    typedef boost::shared_ptr<CloudjectClassificationPipeline<pcl::PFHRGBSignature250> > Ptr;

private:
    int m_q; // num of quantization vectors
    cv::vector<int> m_qValParam;
    cv::vector<float> m_qValPerfs;
    
    std::vector<cv::PCA> m_PCAs;
    cv::Mat m_Q; // the q vectors actually
    
    bool m_bGlobalQuantization;
    
    bool m_bCentersStratification;
    bool m_bPerStrateReduction;
    
    void cloudjectsToPointsSample(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, cv::Mat& X, cv::Mat& c);
    void cloudjectsToPointsSample(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::map<std::string,cv::Mat>& X);
    
    std::vector<cv::PCA> PCAs;
    
    // train bow
    void bow(cv::Mat X, int q, cv::Mat& Q);
    void bow(std::map<std::string,cv::Mat> X, int q, cv::Mat& Q);
    // test bow
    void bow(cv::Mat X, cv::Mat c, cv::Mat Q, cv::Mat& W);

//    void bowSampleFromCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, int q, cv::Mat& W, cv::Mat& Q);
//    void bowSampleFromCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, cv::Mat Q, cv::Mat& W);
    
    void reduce(cv::Mat X, int q, cv::Mat& Xr, std::vector<cv::PCA>& pcas); // train
    void reduce(cv::Mat X, const std::vector<cv::PCA>& pcas, cv::Mat& Xr); // test
};

#endif /* defined(__remedi3__CloudjectClassificationPipeline__) */
