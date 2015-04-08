//
//  ForegroundRegion.hpp
//  remedi3
//
//  Created by Albert Clapés on 22/03/15.
//
//

#ifndef remedi3_ForegroundRegion_hpp
#define remedi3_ForegroundRegion_hpp

#include <stdio.h>
#include "Region.hpp"

class ForegroundRegion : public Region
{
public:
    ForegroundRegion() : Region()
    {}
    
    ForegroundRegion(int id, cv::Rect rect)
    : Region(id,rect)
    {}
    
    ForegroundRegion(const ForegroundRegion& rhs)
    : Region(rhs)
    {
        *this = rhs;
    }
    
    ForegroundRegion(const Region& rhs)
    : Region(rhs)
    {
        *this = rhs;
    }
    
    ForegroundRegion& operator=(const ForegroundRegion& rhs)
    {
        if (this != &rhs)
        {
            Region::operator=(rhs);
            m_Labels  = rhs.m_Labels;
        }
        return *this;
    }
    
    ForegroundRegion& operator=(const Region& rhs)
    {
        if (this != &rhs)
        {
            Region::operator=(rhs);
        }
        return *this;
    }
    
    void allocateMask()
    {
        if (Region::getMask().empty())
            Region::setMask(cv::imread(Region::getFilepath(), CV_LOAD_IMAGE_UNCHANGED));
    }

    bool isLabel(std::string l) const
    {
        return m_Labels.find(l) != m_Labels.end();
    }
    
    std::set<std::string> getLabels() const
    {
        return m_Labels;
    }
    
    void setLabels(std::set<std::string> labels)
    {
        m_Labels = labels;
    }
    
    void removeAllLabels()
    {
        m_Labels.clear();
    }
    
    bool hasLabels()
    {
        return m_Labels.size() > 0;
    }
    
    int getNumOfLabels()
    {
        return m_Labels.size();
    }
    
    void addLabel(std::string label)
    {
        m_Labels.insert(label);
    }
    
    std::string toString(std::string delimiter)
    {
        std::string s = Region::toString(delimiter) + delimiter;
        
        std::set<std::string>::iterator it;
        for (it = m_Labels.begin(); it != m_Labels.end(); ++it)
            s += (*it + delimiter);
        
        return s;
    }
    
private:
    std::set<std::string>             m_Labels;
};

#endif
