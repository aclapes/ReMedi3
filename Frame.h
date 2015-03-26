#pragma once

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

#include <string>

using namespace std;

class Frame
{
public:
	// Constructors
    Frame();
	Frame(cv::Mat mat);
    Frame(cv::Mat mat, cv::Mat mask);
	Frame(const Frame&);
	~Frame(void);

	// Operators
	Frame& operator=(const Frame& rhs);

	// Methods
	bool isValid();
    int getResX();
    int getResY();
    
	void set(cv::Mat mat);
	void set(cv::Mat mat, cv::Mat mask);
    void setMask(cv::Mat mask);
    
    cv::Mat get();
	void get(cv::Mat& mat);
    cv::Mat getMask();
	void getMask(cv::Mat& mask);
    
    cv::Mat getMasked();
    void getMasked(cv::Mat& masked);
    
    void setPath(string path);
    string getPath();

    string getFilename();
    
    typedef boost::shared_ptr<Frame> Ptr;
    
protected:
	// Members
	cv::Mat m_Mat;
    cv::Mat m_Mask;
};

