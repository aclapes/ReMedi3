#pragma once

#include "Frame.h"

// OpenCV
#include <opencv2/opencv.hpp>

#include <boost/shared_ptr.hpp>

class ColorFrame : public Frame
{
public:
	// Constructors
	ColorFrame();
	ColorFrame(cv::Mat mat);
    ColorFrame(cv::Mat mat, cv::Mat mask);
	ColorFrame(const ColorFrame& rhs);
    
    ColorFrame& operator=(const ColorFrame& rhs);
    
    typedef boost::shared_ptr<ColorFrame> Ptr;
};

