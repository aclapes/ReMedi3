#include "ColorFrame.h"

ColorFrame::ColorFrame()
  : Frame()
{
}

ColorFrame::ColorFrame(cv::Mat mat)
  : Frame(mat)
{
    
}

ColorFrame::ColorFrame(cv::Mat mat, cv::Mat mask)
  : Frame(mat, mask)
{
    
}

ColorFrame::ColorFrame(const ColorFrame& rhs)
  : Frame(rhs)
{
    *this = rhs;
}

ColorFrame& ColorFrame::operator=(const ColorFrame& rhs)
{
    if (this != &rhs)
    {
        
    }
    return *this;
}