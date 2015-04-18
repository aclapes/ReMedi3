//
//  InteractionFilteringPipeline.cpp
//  remedi3
//
//  Created by Albert Clapés on 17/04/15.
//
//

#include "InteractionFilteringPipeline.h"
#include "statistics.h"
#include "cvxtended.h"

InteractionFilteringPipeline::InteractionFilteringPipeline()
: m_EvaluationMetric(InteractionFilteringPipeline::EVALUATE_OVERLAP)
{

}

InteractionFilteringPipeline::InteractionFilteringPipeline(const InteractionFilteringPipeline& rhs)
{
    *this = rhs;
}

InteractionFilteringPipeline& InteractionFilteringPipeline::operator=(const InteractionFilteringPipeline& rhs)
{
    if (this != &rhs)
    {
        m_InputData = rhs.m_InputData;
        m_Groundtruth = rhs.m_Groundtruth;
        
        m_OutputData = rhs.m_OutputData;
        
        m_bBridge = rhs.m_bBridge;
        m_bClean = rhs.m_bClean;
        m_MedianFilterSize = rhs.m_MedianFilterSize;
        m_OpClFilterSize = rhs.m_OpClFilterSize;
        
        m_EvaluationMetric = rhs.m_EvaluationMetric;
        
        m_ValParams = rhs.m_ValParams;
        m_ValCombs = rhs.m_ValCombs;
        m_ValPerfs = rhs.m_ValPerfs;
        
        m_Perf = rhs.m_Perf;
    }
    return *this;
}

void InteractionFilteringPipeline::setInputData(std::vector<std::vector<cv::Mat> > data)
{
    m_InputData = data;
}

void InteractionFilteringPipeline::setInputGroundtruth(std::vector<std::vector<cv::Mat> > groundtruth)
{
    m_Groundtruth = groundtruth;
}

void InteractionFilteringPipeline::setBridgeFilter(bool bFilter)
{
    m_bBridge = bFilter;
}

void InteractionFilteringPipeline::setCleanFilter(bool bFilter)
{
    m_bClean = bFilter;
}

void InteractionFilteringPipeline::setMedianFilterSize(int size)
{
    m_MedianFilterSize = size;
}

void InteractionFilteringPipeline::setOpeningClosingFilterSize(int size)
{
    m_OpClFilterSize = size;
}

void InteractionFilteringPipeline::setEvaluationMetric(int metric)
{
    m_EvaluationMetric = metric;
}

void InteractionFilteringPipeline::filter(cv::Mat& output)
{
    // do the filtering
    
    if (m_bBridge)
        applyBridgeFilter(output);
    
    if (m_bClean)
        applyBridgeFilter(output);
    
    if (m_MedianFilterSize > 1)
        applyBridgeFilter(output);
    
    if (m_OpClFilterSize < 0)
        applyClosingFilter(output);
    else if (m_OpClFilterSize > 0)
        applyOpeningFilter(output);
}

void InteractionFilteringPipeline::setValidationParameters(std::vector<float> bridge, std::vector<float> clean, std::vector<float> medianSizes, std::vector<float> opclSizes)
{
    m_ValParams.clear();
    
    if (!bridge.empty())
        m_ValParams.push_back(bridge);
    if (!clean.empty())
        m_ValParams.push_back(clean);
    if (!medianSizes.empty())
        m_ValParams.push_back(medianSizes);
    if (!opclSizes.empty())
        m_ValParams.push_back(opclSizes);
}
                            
void InteractionFilteringPipeline::setValidationParametersCombination(std::vector<float> combination)
{
    setBridgeFilter(combination[0]);
    setCleanFilter(combination[1]);
    setMedianFilterSize(combination[2]);
    setOpeningClosingFilterSize(combination[3]);
}

void InteractionFilteringPipeline::validate()
{
    m_ValPerfs.clear();

    expandParameters<float>(m_ValParams, m_ValCombs);
    
    int numOfViews = m_InputData.size();
    int numOfSequences = m_InputData[0].size();
    int numOfCategories = m_InputData[0][0].cols;
    
    m_ValPerfs.resize(m_ValCombs.size());
    for (int i = 0; i < m_ValCombs.size(); i++)
    {
        std::vector<std::vector<std::vector<cv::Vec4f> > > results (m_InputData.size(),
                                                                    std::vector<std::vector<cv::Vec4f> >(m_InputData[0].size())); // views x seqs
        for (int v = 0; v < m_InputData.size(); v++)
        {
            assert(m_InputData[v].size() == numOfSequences);
            for (int s = 0; s < m_InputData[v].size(); s++)
            {
                assert(m_InputData[v][s].cols = numOfCategories);
                
                setValidationParametersCombination(m_ValCombs[i]);
                
                cv::Mat output = m_InputData[v][s].clone();
                filter(output);
                
                results[v][s] = evaluateErrors(output, m_Groundtruth[v][s]);
            }
        }
        
        // Aggregate over fold's sequences
        std::vector<std::vector<cv::Vec4f> > resultsFold (numOfViews, std::vector<cv::Vec4f>(numOfCategories, cv::Vec4f(0,0,0,0)));
        for (int v = 0; v < results.size(); v++)
            for (int s = 0; s < results[v].size(); s++)
                for (int k = 0; k < results[v][s].size(); k++)
                    resultsFold[v][k] += results[v][s][k];
        
        m_ValPerfs[i] = .0f;
        for (int v = 0; v < numOfViews; v++)
        {
            float perfViewAcc = .0f;
            int nans = 0;
            for (int k = 0; k < numOfCategories; k++)
            {
                float p = computeOverlap(resultsFold[v][k][0], resultsFold[v][k][1], resultsFold[v][k][2]);
                
                if (std::isnan(p)) nans++;
                else perfViewAcc += p;
            }
            
            m_ValPerfs[i] += ( perfViewAcc / (numOfCategories - nans) ) / numOfViews;
        }
    }
}

void InteractionFilteringPipeline::getValidationResults(std::vector<std::vector<float> >& combinations, std::vector<float>& performances)
{
    combinations = m_ValCombs;
    performances = m_ValPerfs;
}

void InteractionFilteringPipeline::save(std::string filename, std::string extension)
{
    std::string filenameWithSuffix = filename + extension; // extension is expected to already include the ext dot (.xxx)
    cv::FileStorage fs;
    
    if (extension.compare(".xml") == 0)
        fs.open(filenameWithSuffix, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_XML);
    else // yml
        fs.open(filenameWithSuffix, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    
    // TODO
    
    fs.release();
}

bool InteractionFilteringPipeline::load(std::string filename, std::string extension)
{
    std::string filenameWithSuffix = filename + extension; // extension is expected to already include the ext dot (.xxx)
    cv::FileStorage fs;
    
    if (extension.compare(".xml") == 0)
        fs.open(filenameWithSuffix, cv::FileStorage::READ | cv::FileStorage::FORMAT_XML);
    else // yml
        fs.open(filenameWithSuffix, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    
    // TODO
    
    fs.release();
}

std::vector<cv::Vec4f> InteractionFilteringPipeline::evaluateErrors(cv::Mat filtered, cv::Mat groundtruth)
{
    std::vector<cv::Vec4f> evaluation (groundtruth.cols);
    
    for (int j = 0; j < groundtruth.cols; j++)
    {
        for (int i = 0; i < groundtruth.rows; i++)
        {
            if (filtered.at<int>(i,j) == 1 && groundtruth.at<int>(i,j) == 1)
                evaluation[j][0]++; // tp
            else if (filtered.at<int>(i,j) == 1 && groundtruth.at<int>(i,j) == 0)
                evaluation[j][1]++; // fp
            else if (filtered.at<int>(i,j) == 0 && groundtruth.at<int>(i,j) == 1)
                evaluation[j][2]++; // fn
            else
                evaluation[j][3]++; // tn
        }
    }
    
    return evaluation;
}

//std::vector<float> InteractionFilteringPipeline::evaluateOverlap(cv::Mat filtered, cv::Mat groundtruth)
//{
//    std::vector<float> evaluation (m_InputData.cols);
//    
//    for (int j = 0; j < m_InputData.cols; j++)
//    {
//        cv::Mat a = filtered.col(j) & groundtruth.col(j);
//        cv::Mat b = filtered.col(j) | groundtruth.col(j);
//        evaluation[j] = cv::sum(a).val[0] / cv::sum(b).val[0];
//    }
//    
//    return evaluation;
//}
//
//std::vector<float> InteractionFilteringPipeline::evaluateFScore(cv::Mat filtered, cv::Mat groundtruth)
//{
//    std::vector<float> evaluation (m_OutputData.cols);
//    
//    for (int j = 0; j < m_OutputData.cols; j++)
//    {
//        float tp = cv::sum(filtered.col(j) & groundtruth.col(j)).val[0];
//        float fp = cv::sum(filtered.col(j) & ~groundtruth.col(j)).val[0];
//        float fn = cv::sum(~filtered.col(j) & groundtruth.col(j)).val[0];
//        
//        evaluation[j] =  2.f*tp / (2*tp + fp + fn);
//    }
//    
//        
//    return evaluation;
//}

void InteractionFilteringPipeline::applyBridgeFilter(cv::Mat& output)
{
    cv::Mat _output (output.rows, output.cols, output.type());
    
    for (int j = 0; j < output.cols; j++)
    {
        if (output.at<int>(1,j) == 1)
            _output.at<int>(0,j) = 1;
        
        for (int i = 1; i < output.rows - 1; i++)
        {
            if (output.at<int>(i-1,j) == 1 && output.at<int>(i+1,j) == 1)
                _output.at<int>(i,j) = 1;
            else
                _output.at<int>(i,j) = output.at<int>(i,j);
        }
        
        if (output.at<int>(output.rows-2,j) == 1)
            _output.at<int>(output.rows-1,j) = 1;
    }
    
    output = _output;
}

void InteractionFilteringPipeline::applyCleanFilter(cv::Mat& output)
{
    cv::Mat _output (output.rows, output.cols, output.type());
    
    for (int j = 0; j < output.cols; j++)
    {
        if (output.at<int>(1,j) == 0)
            _output.at<int>(0,j) = 0;
        
        for (int i = 1; i < output.rows - 1; i++)
        {
            if (output.at<int>(i-1,j) == 0 && output.at<int>(i+1,j) == 0)
                _output.at<int>(i,j) = 0;
            else
                _output.at<int>(i,j) = output.at<int>(i,j);
        }
        
        if (output.at<int>(output.rows-2,j) == 0)
            _output.at<int>(output.rows-1,j) = 0;
    }
    
    output = _output;
}

void InteractionFilteringPipeline::applyMedianFilter(cv::Mat& output)
{
    cv::Mat _output (output.rows, output.cols, output.type());
    
    for (int j = 0; j < output.cols; j++)
    {
        for (int i = 0; i < m_MedianFilterSize; i++)
            _output.at<int>(i,j) = output.at<int>(i,j);
        
        for (int i = m_MedianFilterSize/2+1; i < (output.rows - m_MedianFilterSize/2+1); i++)
        {
            int ones = 0;
            for (int k = -m_MedianFilterSize; k <= m_MedianFilterSize; k++)
                if (_output.at<int>(i+k,j) > 0) ones++;
            if (ones > m_MedianFilterSize)
                _output.at<int>(i,j) = 1;
        }
        
        for (int i = output.rows - m_MedianFilterSize; i < output.rows; i++)
            _output.at<int>(i,j) = output.at<int>(i,j);
    }
    
    output = _output;
}

void InteractionFilteringPipeline::applyDilateFilter(cv::Mat& output)
{
    cv::Mat _output (output.rows, output.cols, output.type());
    
    for (int j = 0; j < output.cols; j++)
    {
        for (int i = m_OpClFilterSize; i < (output.rows - m_OpClFilterSize); i++)
        {
            if (output.at<int>(i,j) == 1)
            {
                for (int k = -m_OpClFilterSize; k < m_OpClFilterSize; k++)
                    _output.at<int>(i+k,j) = 1;
            }
        }
    }
    
    output = _output;
}

void InteractionFilteringPipeline::applyErodeFilter(cv::Mat& output)
{
    cv::Mat _output (output.rows, output.cols, output.type());
    
    for (int j = 0; j < output.cols; j++)
    {
        for (int i = m_OpClFilterSize; i < (output.rows - m_OpClFilterSize); i++)
        {
            if (output.at<int>(i,j) == 0)
            {
                for (int k = -m_OpClFilterSize; k < m_OpClFilterSize; k++)
                    _output.at<int>(i+k,j) = 0;
            }
        }
    }
    
    output = _output;
}

void InteractionFilteringPipeline::applyClosingFilter(cv::Mat& output)
{
    applyDilateFilter(output);
    applyErodeFilter(output);
}

void InteractionFilteringPipeline::applyOpeningFilter(cv::Mat& output)
{
    applyErodeFilter(output);
    applyDilateFilter(output);
}