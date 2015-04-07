//
//  Region.h
//  remedi3
//
//  Created by Albert Clapés on 22/03/15.
//
//

#ifndef __remedi3__Region__
#define __remedi3__Region__

#include <stdio.h>
#include <opencv2/opencv.hpp>

class Region
{
public:
    Region()
    {}
    
    Region(int id, cv::Rect rect)
    : m_ID(id), m_Rect(rect)
    {}
    
    Region(const Region& rhs)
    {
        *this = rhs;
    }
    
    Region& operator=(const Region& rhs)
    {
        if (this != &rhs)
        {
            m_Filepath  = rhs.m_Filepath;
            m_ID        = rhs.m_ID;
            m_Rect      = rhs.m_Rect;
            
            m_Mask     = rhs.m_Mask;
        }
        return *this;
    }
    
    std::string getFilepath() const
    {
        return m_Filepath;
    }
    
    void setFilepath(std::string filepath)
    {
        m_Filepath = filepath;
    }
    
    int getID() const
    {
        return m_ID;
    }
    
    cv::Rect getRect() const
    {
        return m_Rect;
    }
    
    virtual void allocateMask() = 0;
    
    cv::Mat getMask() const
    {
        return m_Mask;
    }
    
    void setMask(cv::Mat mask)
    {
        m_Mask = mask;
    }
    
    void releaseMask()
    {
        m_Mask = cv::Mat();
    }
    
    std::string toString(std::string delimiter)
    {
        std::ostringstream oss;
        oss << m_ID << delimiter
            << m_Rect.x << delimiter
            << m_Rect.y << delimiter
            << (m_Rect.x + m_Rect.width - 1) << delimiter
            << (m_Rect.y + m_Rect.height - 1);
        
        return oss.str();
    }
    
private:
    
    std::string             m_Filepath;
    
    int                     m_ID;
    cv::Rect                m_Rect;
    
    cv::Mat                 m_Mask;
};

#endif /* defined(__remedi3__Region__) */
