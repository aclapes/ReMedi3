//
//  ColorDepthFrame.cpp
//  remedi
//
//  Created by Albert Clapés on 11/07/14.
//
//

#include "ColorDepthFrame.h"

ColorDepthFrame::ColorDepthFrame()
{
    
}

ColorDepthFrame::ColorDepthFrame(cv::Mat colorMat, cv::Mat depthMat)
: ColorFrame(colorMat), DepthFrame(depthMat)//, m_pColoredCloud(new pcl::PointCloud<pcl::PointXYZRGB>)
{
    // Create a point cloud
//    MatToColoredPointCloud(DepthFrame::get(), ColorFrame::get(), *m_pColoredCloud);
}

ColorDepthFrame::ColorDepthFrame(cv::Mat colorMat, cv::Mat depthMat, cv::Mat colorMask, cv::Mat depthMask)
: ColorFrame(colorMat, colorMask), DepthFrame(depthMat, depthMask)//, m_pColoredCloud(new pcl::PointCloud<pcl::PointXYZRGB>)
{
    // Create a point cloud
//    MatToColoredPointCloud(DepthFrame::get(), ColorFrame::get(), *m_pColoredCloud);
}

ColorDepthFrame::ColorDepthFrame(const ColorDepthFrame& rhs)
: ColorFrame(rhs), DepthFrame(rhs)
{
    m_Mask = rhs.m_Mask;
    
    m_Path = rhs.m_Path;
    m_Filename = rhs.m_Filename;
}

ColorDepthFrame& ColorDepthFrame::operator=(const ColorDepthFrame& rhs)
{
    if (this != &rhs)
    {
        ColorFrame::operator=(rhs);
        DepthFrame::operator=(rhs);
        
        m_Mask = rhs.m_Mask;
        m_Path = rhs.m_Path;
        m_Filename = rhs.m_Filename;
    }
    
    return *this;
}

void ColorDepthFrame::setPath(string path)
{
    m_Path = path;
}

string ColorDepthFrame::getPath()
{
    return m_Path;
}

void ColorDepthFrame::setFilename(std::string filename)
{
    m_Filename = filename;
}

string ColorDepthFrame::getFilename()
{
    return m_Filename;
}

void ColorDepthFrame::set(cv::Mat color, cv::Mat depth)
{
    setColor(color);
    setDepth(depth);
}

void ColorDepthFrame::setColor(cv::Mat color)
{
    ColorFrame::set(color);
}

void ColorDepthFrame::setDepth(cv::Mat depth)
{
    DepthFrame::set(depth);
}

void ColorDepthFrame::setColor(cv::Mat color, cv::Mat mask)
{
    ColorFrame::set(color, mask);
}

void ColorDepthFrame::setDepth(cv::Mat depth, cv::Mat mask)
{
    DepthFrame::set(depth, mask);
}

void ColorDepthFrame::setMask(cv::Mat mask)
{
    m_Mask = mask;
}

void ColorDepthFrame::setColorMask(cv::Mat mask)
{
    ColorFrame::setMask(mask);
}

void ColorDepthFrame::setDepthMask(cv::Mat mask)
{
    DepthFrame::setMask(mask);
}

void ColorDepthFrame::get(cv::Mat& color, cv::Mat& depth)
{
    ColorFrame::get(color);
    DepthFrame::get(depth);
}

cv::Mat ColorDepthFrame::getColor()
{
    return ColorFrame::get();
}

cv::Mat ColorDepthFrame::getDepth()
{
    return DepthFrame::get();
}

void ColorDepthFrame::getColor(cv::Mat& color)
{
    ColorFrame::get(color);
}

void ColorDepthFrame::getDepth(cv::Mat& depth)
{
    DepthFrame::get(depth);
}

cv::Mat ColorDepthFrame::getMask()
{
    return m_Mask;
}

cv::Mat ColorDepthFrame::getColorMask()
{
    return ColorFrame::getMask();
}

cv::Mat ColorDepthFrame::getDepthMask()
{
    return DepthFrame::getMask();
}

void ColorDepthFrame::getColorMask(cv::Mat& colorMask)
{
    ColorFrame::getMask(colorMask);
}

void ColorDepthFrame::getDepthMask(cv::Mat& depthMask)
{
    DepthFrame::getMask(depthMask);
}

void ColorDepthFrame::getColoredPointCloud(pcl::PointCloud<pcl::PointXYZRGB>& coloredCloud, bool bMasked)
{
    if (bMasked && !m_Mask.empty())
        MatToColoredPointCloud(DepthFrame::get(), ColorFrame::get(), m_Mask, coloredCloud);
    else
        MatToColoredPointCloud(DepthFrame::get(), ColorFrame::get(), coloredCloud);
}

void ColorDepthFrame::getColoredPointCloud(cv::Mat mask, pcl::PointCloud<pcl::PointXYZRGB>& coloredCloud)
{
    MatToColoredPointCloud(DepthFrame::get(), ColorFrame::get(), mask, coloredCloud);
}

void ColorDepthFrame::getRegisteredColoredPointCloud(pcl::PointCloud<pcl::PointXYZRGB>& coloredCloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pColoredPointCloudTmp (new pcl::PointCloud<pcl::PointXYZRGB>);
    getColoredPointCloud(*pColoredPointCloudTmp);
    
    pcl::transformPointCloud(*pColoredPointCloudTmp, coloredCloud, m_T);
}

void ColorDepthFrame::getRegisteredColoredPointCloud(cv::Mat mask, pcl::PointCloud<pcl::PointXYZRGB>& coloredCloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pColoredPointCloudTmp (new pcl::PointCloud<pcl::PointXYZRGB>);
    getColoredPointCloud(mask, *pColoredPointCloudTmp);
    
    pcl::transformPointCloud(*pColoredPointCloudTmp, coloredCloud, m_T);
}

void ColorDepthFrame::getRegisteredAndReferencedColoredPointCloud(pcl::PointCloud<pcl::PointXYZRGB>& coloredCloud)
{
    // Register the reference point
    pcl::PointXYZ regReferencePoint = getRegisteredReferencePoint();
    
    // Build a translation matrix to the registered reference the cloud after its own registration
    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
    E.col(3) = regReferencePoint.getVector4fMap(); // set translation column
    
    // Apply registration first and then referenciation (right to left order in matrix product)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pColoredPointCloudTmp (new pcl::PointCloud<pcl::PointXYZRGB>);
    getColoredPointCloud(*pColoredPointCloudTmp);
    
    pcl::transformPointCloud(*pColoredPointCloudTmp, coloredCloud, E.inverse() * m_T);
}

void ColorDepthFrame::getRegisteredAndReferencedColoredPointCloud(cv::Mat mask, pcl::PointCloud<pcl::PointXYZRGB>& coloredCloud)
{
    // Register the reference point
    pcl::PointXYZ regReferencePoint = getRegisteredReferencePoint();
    
    // Build a translation matrix to the registered reference the cloud after its own registration
    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
    E.col(3) = regReferencePoint.getVector4fMap(); // set translation column
    
    // Apply registration first and then referenciation (right to left order in matrix product)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pColoredPointCloudTmp (new pcl::PointCloud<pcl::PointXYZRGB>);
    getColoredPointCloud(mask, *pColoredPointCloudTmp);
    
    pcl::transformPointCloud(*pColoredPointCloudTmp, coloredCloud, E.inverse() * m_T);
}

void ColorDepthFrame::registerColoredPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, pcl::PointCloud<pcl::PointXYZRGB>& regColoredCloud)
{
    pcl::transformPointCloud(*coloredCloud, regColoredCloud, m_T);
}