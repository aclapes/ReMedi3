//
//  InteractionFilteringPipeline.h
//  remedi3
//
//  Created by Albert Clapés on 17/04/15.
//
//

#ifndef __remedi3__InteractionFilteringPipeline__
#define __remedi3__InteractionFilteringPipeline__

#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

class InteractionFilteringPipeline
{
public:
    InteractionFilteringPipeline();
    InteractionFilteringPipeline(const InteractionFilteringPipeline& rhs);
    InteractionFilteringPipeline& operator=(const InteractionFilteringPipeline& rhs);
    
    void setInputData(std::vector<std::vector<cv::Mat> > data);
    void setInputGroundtruth(std::vector<std::vector<cv::Mat> > groundtruth);
    
    void setBridgeFilter(bool filter = true); // a zero surrounded by ones become a one
    void setCleanFilter(bool filter = true); // a one surrounded by zeros become a zero
    void setMedianFilterSize(int size);
    void setOpeningClosingFilterSize(int size);
    
    void setEvaluationMetric(int metric);
    
    void filter(cv::Mat& output);

    void setValidationParameters(std::vector<float> bridge, std::vector<float> clean, std::vector<float> medianSizes, std::vector<float> opclSizes);
    void setValidationParametersCombination(std::vector<float> combination);
    void validate();
    void getValidationResults(std::vector<std::vector<float> >& combinations, std::vector<float>& performances);
    
    void save(std::string filename, std::string extension = ".yml");
    bool load(std::string filename, std::string extension = ".yml");
    
    typedef boost::shared_ptr<InteractionFilteringPipeline> Ptr;
    enum { FILTER_MORPH_CLOSING, FILTER_MORPH_OPENING };
    enum { EVALUATE_OVERLAP, EVALUATE_FSCORE };
        
private:
    
    //
    // Attributes
    //
    
    std::vector<std::vector<cv::Mat> > m_InputData;
    std::vector<std::vector<cv::Mat> > m_Groundtruth;
    
    std::vector<std::vector<cv::Mat> > m_OutputData;
    
    bool m_bBridge;
    bool m_bClean;
    int m_MedianFilterSize;
    int m_OpClFilterSize;
    
    int m_EvaluationMetric;
    
    std::vector<std::vector<float> > m_ValParams;
    std::vector<std::vector<float> > m_ValCombs;
    std::vector<float> m_ValPerfs;
    
    float m_Perf;
    
    //
    // Private methods
    //
    
    std::vector<cv::Vec4f> evaluateErrors(cv::Mat filtered, cv::Mat groundtruth);
//    std::vector<float> evaluateOverlap(cv::Mat filtered, cv::Mat groundtruth);
//    std::vector<float> evaluateFScore(cv::Mat filtered, cv::Mat groundtruth);
    
    void applyBridgeFilter(cv::Mat& output);
    void applyCleanFilter(cv::Mat& output);
    void applyMedianFilter(cv::Mat& output);
    void applyDilateFilter(cv::Mat& output);
    void applyErodeFilter(cv::Mat& output);
    void applyClosingFilter(cv::Mat& output);
    void applyOpeningFilter(cv::Mat& output);
};

#endif /* defined(__remedi3__InteractionFilteringPipeline__) */
