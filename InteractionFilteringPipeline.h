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
    
    void setInputData(cv::Mat input);
    
    void setPositiveBridgeFilter(bool filter = true); // a zero surrounded by ones become a one
    void setNegativeBridgeFilter(bool filter = true); // a one surrounded by zeros become a zero
    void setMedianFilter(int size);
    
    void filter(cv::Mat& output);
    
    boost::shared_ptr<InteractionFilteringPipeline> Ptr;
        
private:
    
    cv::Mat m_Input;
    
    bool bPositiveBridge;
    bool bNegativeBridge;
    int m_MedianFilterSize;
};

#endif /* defined(__remedi3__InteractionFilteringPipeline__) */
