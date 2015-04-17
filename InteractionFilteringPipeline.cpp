//
//  InteractionFilteringPipeline.cpp
//  remedi3
//
//  Created by Albert Clapés on 17/04/15.
//
//

#include "InteractionFilteringPipeline.h"

InteractionFilteringPipeline::InteractionFilteringPipeline()
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
        m_Input = rhs.m_Input;
        
        bPositiveBridge = rhs.bPositiveBridge;
        bNegativeBridge = rhs.bNegativeBridge;
        m_MedianFilterSize = rhs.m_MedianFilterSize;
    }
    return *this;
}

void InteractionFilteringPipeline::setInputData(cv::Mat input)
{
    m_Input = input;
}

void InteractionFilteringPipeline::setPositiveBridgeFilter(bool filter)
{
    bPositiveBridge = filter;
}

void InteractionFilteringPipeline::setNegativeBridgeFilter(bool filter)
{
    bNegativeBridge = filter;
}

void InteractionFilteringPipeline::setMedianFilter(int size)
{
    m_MedianFilterSize = size;
}

void InteractionFilteringPipeline::filter(cv::Mat& output)
{
    // do the filtering
}

