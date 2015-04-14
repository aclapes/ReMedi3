#include "Frame.h"

Frame::Frame()
{
}

Frame::Frame(cv::Mat mat)
: m_Mat(mat)
{
}

Frame::Frame(cv::Mat mat, cv::Mat mask)
: m_Mat(mat), m_Mask(mask)
{
}

Frame::~Frame(void)
{
}

Frame::Frame(const Frame& rhs)
{
	*this = rhs;
}

Frame& Frame::operator=(const Frame& rhs)
{
	if (this != &rhs)
    {
        m_Mat = rhs.m_Mat;
        m_Mask = rhs.m_Mask;
        
        m_Path = rhs.m_Path;
        m_Filename = rhs.m_Filename;
    }

	return *this;
}

bool Frame::isValid()
{
	return m_Mat.rows > 0 && m_Mat.cols > 0;
}

int Frame::getResX()
{
    return m_Mat.cols;
}

int Frame::getResY()
{
    return m_Mat.rows;
}

cv::Mat Frame::get()
{
	return m_Mat;
}

void Frame::get(cv::Mat& mat)
{
	mat = m_Mat;
}

cv::Mat Frame::getMask()
{
	return m_Mask;
}

void Frame::getMask(cv::Mat& mask)
{
	mask = m_Mask;
}

void Frame::set(cv::Mat mat)
{
	mat.copyTo(m_Mat);
}

void Frame::set(cv::Mat mat, cv::Mat mask)
{
	m_Mat = mat;
    m_Mask = mask;
}

void Frame::setMask(cv::Mat mask)
{
    m_Mask = mask;
}

cv::Mat Frame::getMasked()
{
    cv::Mat masked;
    
    getMasked(masked);
    
    return masked;
}

void Frame::getMasked(cv::Mat& masked)
{
    m_Mat.copyTo(masked, m_Mask);
}

void Frame::setPath(std::string path)
{
    m_Path = path;
}

std::string Frame::getPath()
{
    return m_Path;
}

void Frame::setFilename(std::string filename)
{
    m_Filename = filename;
}

std::string Frame::getFilename()
{
    return m_Filename;
}