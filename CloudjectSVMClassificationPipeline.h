//
//  CloudjectSVMClassificationPipeline.h
//  remedi3
//
//  Created by Albert Clapés on 12/03/15.
//
//

#ifndef __remedi3__CloudjectSVMClassificationPipeline__
#define __remedi3__CloudjectSVMClassificationPipeline__

#include <stdio.h>
#include <list>
#include <vector>

#include "Cloudject.hpp"

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

enum type {CCPIPELINE_NORM_MINMAX, CCPIPELINE_NORM_STD,
    CCPIPELINE_NORM_L1, CCPIPELINE_NORM_L2};

template<typename T>
class CloudjectSVMClassificationPipelineBase
{
public:
    CloudjectSVMClassificationPipelineBase();
    CloudjectSVMClassificationPipelineBase(const CloudjectSVMClassificationPipelineBase& rhs);
    CloudjectSVMClassificationPipelineBase& operator=(const CloudjectSVMClassificationPipelineBase& rhs);
    ~CloudjectSVMClassificationPipelineBase();
    
    void setInputCloudjects(boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjects);
    void setCategories(std::vector<const char*> categories);
//    boost::shared_ptr<std::list<Cloudject::Ptr> > getInputCloudjects();
    
    void setDimRedVariance(double var);
    void setDimRedNumFeatures(int n);
    
    void setNormalization(int normType);
    void setClassifierType(std::string classifierName);

    void setValidation(int numFolds);
    void setClassifierValidationParameters(std::vector<std::vector<float> > classifierParams);

    void setClassifierParameters(std::vector<std::vector<float> > classifierParams);

protected:

    boost::shared_ptr<std::list<MockCloudject::Ptr> > m_InputMockCloudjects;
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
    
    template<typename CloudjectTypePtr>
    void getLabels(boost::shared_ptr<typename std::list<CloudjectTypePtr> > cloudjects, std::vector<const char*> categories, cv::Mat& Y);
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
    
    template<typename CloudjectTypePtr>
    void getTrainingCloudjects(boost::shared_ptr<typename std::list<CloudjectTypePtr> > cloudjects, int t, typename std::list<CloudjectTypePtr>& cloudjectsTr);
    template<typename CloudjectTypePtr>
    void getTestCloudjects(boost::shared_ptr<typename std::list<CloudjectTypePtr> > cloudjects, int t, typename std::list<CloudjectTypePtr>& cloudjectsTr);
    
    void getTraining(cv::Mat X, cv::Mat c, int t, cv::Mat& XTr, cv::Mat& cTr);
    void getTest(cv::Mat X, cv::Mat c, int t, cv::Mat& XTe, cv::Mat& cTe);
    void getTraining(cv::Mat X, int t, cv::Mat& XTr);
    void getTest(cv::Mat X, int t, cv::Mat& XTe);
    void stratify(cv::Mat X, cv::Mat c, cv::Mat Y, std::map<std::string,cv::Mat>& XMap);
    
private:
    std::vector<std::vector<float> > m_ClassifiersValParamCombs;
    cv::Mat m_ClassifiersValPerfs; // Validation parameters [P]erformance
};

template<typename T>
class CloudjectSVMClassificationPipeline : public CloudjectSVMClassificationPipelineBase<T>
{};

template<>
class CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250> : public CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>
{
public:
    CloudjectSVMClassificationPipeline();
    CloudjectSVMClassificationPipeline(const CloudjectSVMClassificationPipeline& rhs);
    CloudjectSVMClassificationPipeline& operator=(const CloudjectSVMClassificationPipeline& rhs);
    
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
    
//    void loadTrain();
    void save(std::string filename, std::string extension = ".yml");
    bool load(std::string filename, std::string extension = ".yml");
    
    std::vector<float> predict(boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjects, std::list<std::vector<int> >& predictions, std::list<std::vector<float> >& distsToMargin);
    std::vector<int> predict(Cloudject::Ptr cloudject, std::vector<float>& distsToMargin);
    
    typedef boost::shared_ptr<CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250> > Ptr;

private:
    int m_q; // num of quantization vectors
    cv::vector<int> m_qValParam;
    cv::vector<float> m_qValPerfs;
    
    std::vector<cv::PCA> m_PCAs;
    cv::Mat m_Centers; // the q vectors actually
    
    bool m_bGlobalQuantization;
    
    bool m_bCentersStratification;
    bool m_bPerStrateReduction;
    
    void cloudjectsToPointsSample(boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjects, cv::Mat& X, cv::Mat& c);
    void cloudjectsToPointsSample(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, cv::Mat& X, cv::Mat& c);
    void cloudjectsToPointsSample(boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjects, std::map<std::string,cv::Mat>& X);
    
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

#endif /* defined(__remedi3__CloudjectSVMClassificationPipeline__) */
