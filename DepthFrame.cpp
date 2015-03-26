#include "DepthFrame.h"

#include <pcl/common/transforms.h>

DepthFrame::DepthFrame() : Frame()
{
}

DepthFrame::DepthFrame(cv::Mat mat) : Frame(mat / 8)//, m_pCloud(new pcl::PointCloud<pcl::PointXYZ>)
{
	// Player index map
    cv::Mat bitmask (mat.rows, mat.cols, mat.type(), cv::Scalar(7));
    cv::bitwise_and(mat, bitmask, m_UIDMat);
	m_UIDMat.convertTo(m_UIDMat, CV_8UC1);
    
    // Create a point cloud
//    MatToPointCloud(Frame::get(), *m_pCloud);
}

DepthFrame::DepthFrame(cv::Mat mat, cv::Mat mask) : Frame(mat / 8, mask)//, m_pCloud(new pcl::PointCloud<pcl::PointXYZ>)
{
	// Player index map
    cv::Mat bitmask (mat.rows, mat.cols, mat.type(), cv::Scalar(7));
    cv::bitwise_and(mat, bitmask, m_UIDMat);
	m_UIDMat.convertTo(m_UIDMat, CV_8UC1);
    
    // Create a point cloud
//    MatToPointCloud(Frame::get(), *m_pCloud);
}

DepthFrame::DepthFrame(const DepthFrame& rhs) : Frame(rhs)
{
    *this = rhs;
}

DepthFrame& DepthFrame::operator=(const DepthFrame& rhs)
{
    if (this != &rhs)
    {
        Frame::operator=(rhs);
        m_UIDMat = rhs.m_UIDMat;
        m_ReferencePoint = rhs.m_ReferencePoint;
        m_T = rhs.m_T;
    }
    
	return *this;
}

void DepthFrame::set(cv::Mat mat)
{
    Frame::set(mat / 8);
}

void DepthFrame::set(cv::Mat mat, cv::Mat mask)
{
    Frame::set(mat / 8, mask);
    
//    MatToPointCloud(Frame::getMask(), *m_pCloud);
}

void DepthFrame::getPointCloud(pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    if (Frame::getMask().empty())
        MatToPointCloud(Frame::get(), cloud);
    else
        MatToPointCloud(Frame::get(), Frame::getMask(), cloud);
}

void DepthFrame::getPointCloud(cv::Mat mask, pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    MatToPointCloud(Frame::get(), mask, cloud);
}

void DepthFrame::setReferencePoint(pcl::PointXYZ reference)
{
    m_ReferencePoint = reference;
}

pcl::PointXYZ DepthFrame::getReferencePoint() const
{
    return m_ReferencePoint;
}

void DepthFrame::setRegistrationTransformation(Eigen::Matrix4f T)
{
    m_T = T;
}

Eigen::Matrix4f DepthFrame::getRegistrationTransformation() const
{
    return m_T;
}

pcl::PointXYZ DepthFrame::getRegisteredReferencePoint()
{
    // Register the reference point
    pcl::PointXYZ regReferencePoint;
    
    regReferencePoint.getVector4fMap() = (m_T * m_ReferencePoint.getVector4fMap());
    
    return regReferencePoint;
}

void DepthFrame::getRegisteredPointCloud(pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pCloudTmp (new pcl::PointCloud<pcl::PointXYZ>);
    getPointCloud(*pCloudTmp);
    
    pcl::transformPointCloud(*pCloudTmp, cloud, m_T);
}

void DepthFrame::getRegisteredAndReferencedPointCloud(pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    pcl::PointXYZ regReferencePoint = getRegisteredReferencePoint();
    
    // Build a translation matrix to the registered reference the cloud after its own registration
    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
    E.col(3) = regReferencePoint.getVector4fMap(); // set translation column
    
    // Apply registration first and then referenciation (right to left order in matrix product)
    pcl::PointCloud<pcl::PointXYZ>::Ptr pCloudTmp (new pcl::PointCloud<pcl::PointXYZ>);
    getPointCloud(*pCloudTmp);
    
    pcl::transformPointCloud(*pCloudTmp, cloud, E.inverse() * m_T);
}

void DepthFrame::getDeregisteredPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pRegisteredCloud, pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    Eigen::Matrix4f iT = m_T.inverse();
    pcl::transformPointCloud(*pRegisteredCloud, cloud, iT);
}

pcl::PointXYZ DepthFrame::registratePoint(pcl::PointXYZ src)
{
    pcl::PointCloud<pcl::PointXYZ> pCloudTmp, pRegCloudTmp;
    pCloudTmp.push_back(src);
    
    pcl::transformPointCloud(pCloudTmp, pRegCloudTmp, m_T);
    return pRegCloudTmp.points[0];
}

pcl::PointXYZ DepthFrame::registrateAndReferencePoint(pcl::PointXYZ src)
{
    pcl::PointXYZ regReferencePoint = getRegisteredReferencePoint();
    
    // Build a translation matrix to the registered reference the cloud after its own registration
    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
    E.col(3) = regReferencePoint.getVector4fMap(); // set translation column
    
    pcl::PointCloud<pcl::PointXYZ> pCloudTmp, pRegCloudTmp;
    pCloudTmp.push_back(src);
    
    pcl::transformPointCloud(pCloudTmp, pRegCloudTmp, E.inverse() * m_T);
    return pRegCloudTmp.points[0];
}

Eigen::Vector4f DepthFrame::registratePoint(Eigen::Vector4f src)
{
    Eigen::Vector4f _src = src;
    _src.w() = 1;
    return m_T * _src;
}

Eigen::Vector4f DepthFrame::registrateAndReferencePoint(Eigen::Vector4f src)
{
    pcl::PointXYZ regReferencePoint = getRegisteredReferencePoint();
    
    // Build a translation matrix to the registered reference the cloud after its own registration
    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
    E.col(3) = regReferencePoint.getVector4fMap(); // set translation column
    
    Eigen::Vector4f _src = src;
    _src.w() = 1;
    return E.inverse() * m_T * _src;
}