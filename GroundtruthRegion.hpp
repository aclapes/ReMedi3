//
//  GroundtruthRegion.cpp
//  remedi3
//
//  Created by Albert Clapés on 10/03/15.
//
//

#ifndef __remedi3__GroundtruthRegion__
#define __remedi3__GroundtruthRegion__

#include <stdio.h>
#include "Region.hpp"

class GroundtruthRegion : public Region
{
public:
    GroundtruthRegion() : Region()
    {}

    GroundtruthRegion(int id, cv::Rect rect)
    : Region(id,rect)
    {}

    GroundtruthRegion(const GroundtruthRegion& rhs)
    : Region(rhs)
    {
        *this = rhs;
    }

    GroundtruthRegion& operator=(const GroundtruthRegion& rhs)
    {
        if (this != &rhs)
        {
            Region::operator=(rhs);
            m_Category  = rhs.m_Category;
        }
        return *this;
    }
    
    void allocateMask()
    {
        if (Region::getMask().empty())
        {
            cv::Mat aux = cv::imread(Region::getFilepath(), CV_LOAD_IMAGE_UNCHANGED);
            cv::Mat mask = (aux > 0); // groundtruth has assigned intermediate values (200, 201, ..)
            Region::setMask( cv::Mat(mask, Region::getRect()) );
        }
    }
    
    void setCategory(std::string category)
    {
        m_Category = category;
    }

    std::string getCategory()
    {
        return m_Category;
    }

    std::string toString(std::string delimiter)
    {
        std::string s = Region::toString(delimiter) + delimiter
                      + m_Category;
        
        return s;
    }
    
private:
    std::string             m_Category;
};

#endif /* defined(__remedi3__GroundtruthRegion__) */
