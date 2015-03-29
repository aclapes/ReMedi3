#include "conversion.h"

#include <math.h>

void normalsToAngles(pcl::PointCloud<pcl::Normal>::Ptr pNormals, cv::Mat& angles)
{
    cv::Mat theta (pNormals->height, pNormals->width, cv::DataType<float>::type);
    cv::Mat phi (pNormals->height, pNormals->width, cv::DataType<float>::type);
    
    for (unsigned int y = 0; y < pNormals->height; y++) for (unsigned int x = 0; x < pNormals->width; x++)
    {
        
        float nx = pNormals->at(x,y).normal_x;
        float ny = pNormals->at(x,y).normal_y;
        float nz = pNormals->at(x,y).normal_z;
        
        float r = sqrt(powf(nx,2) + powf(ny,2) + powf(nz,2));
        
        theta.at<float>(y,x) = acos(nz / r ) * 180.0 / 3.1415926535;
        phi.at<float>(y,x)   = atan(ny / nx) * 180.0 / 3.1415926535;
    }
    
    std::vector<cv::Mat> anglesChannels (2);
    anglesChannels[0] = theta;
    anglesChannels[1] = phi;
    cv::merge(anglesChannels.data(), 2, angles);
}

void pcl2cv(pcl::PointXYZ p, cv::Mat& m)
{
    m = cv::Mat(1, 3, cv::DataType<float>::type);
    
    m.at<float>(0,0) = p.x;
    m.at<float>(0,1) = p.y;
    m.at<float>(0,2) = p.z;
}

void pcl2cv(pcl::PointXYZRGB p, cv::Mat& m)
{
    m = cv::Mat(1, 6, cv::DataType<float>::type);
    
    m.at<float>(0,0) = p.x;
    m.at<float>(0,1) = p.y;
    m.at<float>(0,2) = p.z;
    m.at<float>(0,3) = p.r;
    m.at<float>(0,4) = p.g;
    m.at<float>(0,2) = p.b;
}

void ProjectiveToRealworld(pcl::PointXYZ p, int xres, int yres, int x0, int y0, pcl::PointXYZ& rw)
{
    float invfocal = (1/285.63f) / (xres/320.f);
    
    rw.x = (((p.x+x0) - xres/2) * invfocal * p.z) / 1000.f,
    rw.y = (((p.y+y0) - yres/2) * invfocal * p.z) / 1000.f,
    rw.z = p.z / 1000.f;
}

void RealworldToProjective(pcl::PointXYZ rw, int xres, int yres, int x0, int y0, pcl::PointXYZ& p)
{
    float invfocal = (1/285.63f) / (xres/320.f);
    
    p.x = (int) ( ((rw.x) / (invfocal * rw.z)) + (xres / 2.f) ) + x0;
    p.y = (int) ( ((rw.y) / (invfocal * rw.z)) + (yres / 2.f) ) + y0;
    p.z = rw.z * 1000.f;
}

void ProjectiveToRealworld(pcl::PointXYZ p, int xres, int yres, pcl::PointXYZ& rw)
{
    RealworldToProjective(p, xres, yres, 0, 0, rw);
}

void RealworldToProjective(pcl::PointXYZ rw, int xres, int yres, pcl::PointXYZ& p)
{
    ProjectiveToRealworld(rw, xres, yres, 0, 0, p);
}

void ProjectiveToRealworld(pcl::PointCloud<pcl::PointXYZ>::Ptr pProjCloud, pcl::PointCloud<pcl::PointXYZ>& realCloud)
{
    realCloud.height = pProjCloud->height;
    realCloud.width  = pProjCloud->width;
    realCloud.resize( pProjCloud->height * pProjCloud->width );
    
    for (unsigned int y = 0; y < realCloud.height; y++) for (unsigned int x = 0; x < realCloud.width; x++)
    {
        pcl::PointXYZ realPoint;
        ProjectiveToRealworld(pProjCloud->at(x,y), pProjCloud->width, pProjCloud->height, realPoint);
        realCloud.at(x,y) = realPoint;
    }
}

void RealworldToProjective(pcl::PointCloud<pcl::PointXYZ>::Ptr pRealCloud, pcl::PointCloud<pcl::PointXYZ>& projCloud)
{
    projCloud.height = pRealCloud->height;
    projCloud.width  = pRealCloud->width;
    projCloud.resize( pRealCloud->height * pRealCloud->width );
    
    for (unsigned int y = 0; y < projCloud.height; y++) for (unsigned int x = 0; x < projCloud.width; x++)
    {
        pcl::PointXYZ projPoint;
        RealworldToProjective(pRealCloud->at(x,y), pRealCloud->width, pRealCloud->height, projPoint);
        projCloud.at(x,y) = projPoint;
    }
}

void EigenToPointXYZ(Eigen::Vector4f eigen, pcl::PointXYZ& p)
{
    p.x = eigen.x();
    p.y = eigen.y();
    p.z = eigen.z();
}

pcl::PointXYZ EigenToPointXYZ(Eigen::Vector4f eigen)
{
    pcl::PointXYZ p;
    
    p.x = eigen.x();
    p.y = eigen.y();
    p.z = eigen.z();
    
    return p;
}

void MatToPointCloud(cv::Mat mat, pcl::PointCloud<pcl::PointXYZ>& cloud)
{
	cloud.height = mat.rows;
    cloud.width =  mat.cols;
    cloud.resize(cloud.height * cloud.width);
    cloud.is_dense = true;
    
    float invfocal = (1/285.63f) / (mat.cols/320.f); // 3.501e-3f / (mat.cols/320.f); // Kinect inverse focal length. If depth map resolution
    
	float z;
	float rwx, rwy, rwz;
	pcl::PointXYZ p;
    
    for (unsigned int y = 0; y < cloud.height; y++) for (unsigned int x = 0; x < cloud.width; x++)
    {
		//z_us = mat.at<unsigned short>(y,x) >> 3;
		z = (float) mat.at<unsigned short>(y,x);
        if (z == 0)
            continue;
        
		rwx = (x - 320.0) * invfocal * z;
		rwy = (y - 240.0) * invfocal * z;
		rwz = z;
        
		p.x = rwx/1000.f;
		p.y = rwy/1000.f;
		p.z = rwz/1000.f;
		cloud.at(x,y) = p;
    }
}

void MatToColoredPointCloud(cv::Mat depth, cv::Mat color, pcl::PointCloud<pcl::PointXYZRGB>& cloud)
{
	cloud.height = depth.rows;
    cloud.width =  depth.cols;
    cloud.resize(cloud.height * cloud.width);
    cloud.is_dense = true;
    
    float invfocal = (1/285.63f) / (depth.cols/320.f); // Kinect inverse focal length. If depth map resolution
    
	float z;
	float rwx, rwy, rwz;
	pcl::PointXYZRGB p;
    
    for (unsigned int y = 0; y < cloud.height; y++) for (unsigned int x = 0; x < cloud.width; x++)
    {
		//z_us = mat.at<unsigned short>(y,x) >> 3;
		z = (float) depth.at<unsigned short>(y,x) /*z_us*/;
        
		rwx = (x - 320.0) * invfocal * z;
		rwy = (y - 240.0) * invfocal * z;
		rwz = z;
        
		p.x = rwx/1000.f;
		p.y = rwy/1000.f;
		p.z = rwz/1000.f;
        
        cv::Vec3b c = color.at<cv::Vec3b>(y,x);
        p.b = c[0];
        p.g = c[1];
        p.r = c[2];
        
		cloud.at(x,y) = p;
    }
}

void MatToColoredPointCloud(cv::Mat depth, cv::Mat color, cv::Mat mask, pcl::PointCloud<pcl::PointXYZRGB>& cloud)
{
    cloud.height = depth.rows;
    cloud.width =  depth.cols;
    cloud.resize(cloud.height * cloud.width);
    cloud.is_dense = true;
    
    float invfocal = (1/285.63f) / (depth.cols/320.f); // Kinect inverse focal length. If depth map resolution
    
    float z;
    float rwx, rwy, rwz;
    pcl::PointXYZRGB p;
    
    for (unsigned int y = 0; y < cloud.height; y++) for (unsigned int x = 0; x < cloud.width; x++)
    {
        if (mask.at<unsigned char>(y,x) > 0)
        {
            z = (float) depth.at<unsigned short>(y,x) /*z_us*/;
            
            rwx = (x - 320.0) * invfocal * z;
            rwy = (y - 240.0) * invfocal * z;
            rwz = z;
            
            p.x = rwx/1000.f;
            p.y = rwy/1000.f;
            p.z = rwz/1000.f;
            
            cv::Vec3b c = color.at<cv::Vec3b>(y,x);
            p.b = c[0];
            p.g = c[1];
            p.r = c[2];
            
            cloud.at(x,y) = p;
        }
    }
}

void MatToPointCloud(cv::Mat mat, cv::Mat mask, pcl::PointCloud<pcl::PointXYZ>& cloud)
{
	cloud.height = mat.rows;
    cloud.width =  mat.cols;
    cloud.resize(cloud.height * cloud.width);
    cloud.is_dense = true;
    
    // Easy to handle the conversion this way
    cv::Mat temp = cv::Mat(mat.size(), CV_32F);
    mat.convertTo(temp, CV_32F);
    
    float invfocal = (1/285.63f) / (mat.cols/320.f); // Kinect inverse focal length. If depth map resolution
    
	unsigned short z_us;
	float z;
	float rwx, rwy, rwz;
	pcl::PointXYZ p;

    for (unsigned int y = 0; y < cloud.height; y++) for (unsigned int x = 0; x < cloud.width; x++)
    {
		z = (mask.at<unsigned char>(y,x) == 255) ? ((float) mat.at<unsigned short>(y,x)) : 0;
             
		rwx = (x - 320.0) * invfocal * z;
		rwy = (y - 240.0) * invfocal * z;
		rwz = z;
              
		p.x = rwx/1000.f; 
		p.y = rwy/1000.f;
		p.z = rwz/1000.f;
		cloud.at(x,y) = p;
    }
}


void PointCloudToMat(pcl::PointCloud<pcl::PointXYZ>& cloud, cv::Mat& mat)
{
	mat = cv::Mat::zeros(cloud.height, cloud.width, CV_16UC1);
    
    float invfocal = (1/285.63f) / (mat.cols/320.f); // Kinect inverse focal length. If depth map resolution
    
    for (unsigned int i = 0; i < cloud.points.size(); i++)
    {
        float rwx = cloud.points[i].x;
        float rwy = cloud.points[i].y;
        float rwz = cloud.points[i].z;
            
        float x, y, z;
    
        if (rwz > 0)
        {
            z = rwz * 1000.f;
			x = std::floor( ((rwx ) / (invfocal * rwz )) + (cloud.width / 2.f) );
			y = std::floor( ((rwy ) / (invfocal * rwz )) + (cloud.height / 2.f) );

			if (x > 0 && x < cloud.width && y > 0 && y < cloud.height)
            {
                mat.at<unsigned short>(y,x) = z;
            }
        }
    }
}

void PointCloudToMat(pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud, int height, int width, cv::Mat& mat)
{
    mat = cv::Mat::zeros(height, width, CV_16UC1);
    
    float invfocal = (1/285.63f) / (width/320.f); // Kinect inverse focal length. If depth map resolution
    
    for (unsigned int i = 0; i < pCloud->points.size(); i++)
    {
        float rwx = pCloud->points[i].x;
        float rwy = pCloud->points[i].y;
        float rwz = pCloud->points[i].z;
        
        float x, y, z;
        
        if (rwz > 0)
        {
            z = rwz * 1000.f;
			x = std::floor( ((rwx ) / (invfocal * rwz )) + (width / 2.f) );
			y = std::floor( ((rwy ) / (invfocal * rwz )) + (height / 2.f) );
            
			if (x > 0 && x < width && y > 0 && y < height)
            {
                mat.at<unsigned short>(y,x) = z;
            }
        }
    }
}

void ColoredPointCloudToMat(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud, int height, int width, int x0, int y0, cv::Mat& mat)
{
    cv::Mat tmp (height, width, CV_16UC1, cv::Scalar(0));
    cv::Mat c  (height, width, CV_8UC3, cv::Scalar(0));
    
    float invfocal = (1/285.63f) / (width/320.f); // Kinect inverse focal length. If depth map resolution
    
    for (unsigned int i = 0; i < pCloud->points.size(); i++)
    {
        float rwx = pCloud->points[i].x;
        float rwy = pCloud->points[i].y;
        float rwz = pCloud->points[i].z;
        
        float x, y, z;
        
        if (rwz > 0)
        {
            x = std::floor( (rwx/(invfocal * rwz)) + (width/2.f) );// + x0;
            y = std::floor( (rwy/(invfocal * rwz)) + (height/2.f) );// + y0;
            z = rwz * 1000.f;
            
            if (x > 0 && x < width && y > 0 && y < height)
            {
                tmp.at<unsigned short>(y,x) = z;
                c.at<cv::Vec3b>(y,x) = cv::Vec3b(pCloud->points[i].b, pCloud->points[i].g, pCloud->points[i].r);
            }
        }
    }
    
    mat = cv::Mat(tmp, cv::Rect(x0,y0,pCloud->width,pCloud->height));
}

void ColoredPointCloudToMat(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud, int height, int width, cv::Mat& mat)
{
    ColoredPointCloudToMat(pCloud, height, width, 0, 0, mat);
}

void MaskDensePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pCloudSrc, cv::Mat maskMat,
	pcl::PointCloud<pcl::PointXYZ>& cloudTgt)
{
	for (unsigned int y = 0; y < pCloudSrc->height; y++) for (unsigned int x = 0; x < pCloudSrc->width; x++)
	{
		unsigned char maskValue;

		if (maskMat.type() == CV_8U || maskMat.type() == CV_8UC1)
			maskValue = maskMat.at<unsigned char>(y,x);
		else if (maskMat.type() == CV_32F || maskMat.type() == CV_32FC1)
			maskValue = static_cast<unsigned char>(maskMat.at<float>(y,x));

		if (maskValue > 0)
		{
			cloudTgt.points.push_back(pCloudSrc->points[y * pCloudSrc->width + x]);
		}
	}

	cloudTgt.width = cloudTgt.points.size();
	cloudTgt.height = 1;
	cloudTgt.is_dense = false;
}


void passthroughFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud, pcl::PointXYZ min, pcl::PointXYZ max, 
	pcl::PointCloud<pcl::PointXYZ>& cloudF)
{
	cloudF = *pCloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pCloudF (&cloudF);

	pcl::PassThrough<pcl::PointXYZ> pt;

	pt.setInputCloud(pCloudF);
	pt.setFilterFieldName("x");
	pt.setFilterLimits(min.x, max.x);
	pt.filter(*pCloudF);

	pt.setInputCloud(pCloudF);
	pt.setFilterFieldName("y");
	pt.setFilterLimits(min.y, max.y);
	pt.filter(*pCloudF);

	pt.setInputCloud(pCloudF);
	pt.setFilterFieldName("z");
	pt.setFilterLimits(min.z, max.z);
	pt.filter(*pCloudF);
}


void enclosure(cv::Mat src, cv::Mat& dst, int size)
{
	// Erode and dilate mask
	int erosion_size = size;
	int dilation_size = size;

	cv::Mat erode_element = cv::getStructuringElement( cv::MORPH_ERODE,
                                cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                cv::Point( erosion_size, erosion_size ) );
	cv::Mat dilate_element = cv::getStructuringElement( cv::MORPH_DILATE,
                                cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                cv::Point( dilation_size, dilation_size ) );

	cv::erode(src, src, erode_element);
	cv::dilate(src, dst, dilate_element);
}

template<typename PointT>
void biggestEuclideanCluster(typename pcl::PointCloud<PointT>::Ptr pCloud, int minSz, int maxSz, float clusterTol,
	pcl::PointCloud<PointT>& cluster)
{
    // Not even try
    // bc the entire cloud is smaller than the minSz
    if (pCloud->points.size() < minSz)
        return;
    
	typename pcl::PointCloud<PointT>::Ptr pNanFreeCloud (new pcl::PointCloud<PointT>);

	std::vector< int > indices;
	pcl::removeNaNFromPointCloud(*pCloud,*pNanFreeCloud,indices);

	// Creating the KdTree object for the search method of the extraction
	typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
	tree->setInputCloud (pNanFreeCloud);
    
	std::vector<pcl::PointIndices> clusterIndices;
    
	pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setInputCloud (pNanFreeCloud);
	ec.setClusterTolerance (clusterTol); // 2cm
	ec.setMinClusterSize (minSz);
	ec.setMaxClusterSize (maxSz);
	ec.setSearchMethod (tree);

	ec.extract (clusterIndices);

    if (clusterIndices.size() > 0)
    {
        // biggest is the first in the list of extractions' indices
        std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin();
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
            cluster.points.push_back (pNanFreeCloud->points[*pit]);

        cluster.width = cluster.points.size ();
        cluster.height = 1;
        cluster.is_dense = false;
    }
}


float euclideanDistance(Eigen::Vector4f u, Eigen::Vector4f v)
{
	return std::sqrtf(std::powf(u.x() - v.x(), 2) + std::powf(u.y() - v.y(), 2) + std::powf(u.z() - v.z(), 2));
}

cv::Mat bgr2hs(cv::Mat bgrMat)
{
    cv::Mat hsvMat, hsMat;
    std::vector<cv::Mat> hsvChannels;
    
    cv::cvtColor(bgrMat, hsvMat, CV_BGR2HSV);
    cv::split(hsvMat, hsvChannels);
    
    hsvChannels[0].convertTo(hsvChannels[0], cv::DataType<float>::type);
    hsvChannels[1].convertTo(hsvChannels[1], cv::DataType<float>::type);
    cv::merge(hsvChannels.data(), 2, hsMat);
    
    return hsMat;
}

cv::Mat bgr2hs0(cv::Mat bgrMat)
{
    cv::Mat hsvMat, hsMat;
    std::vector<cv::Mat> hsvChannels;
    
    cv::cvtColor(bgrMat, hsvMat, CV_BGR2HSV);
    cv::split(hsvMat, hsvChannels);
    
    cv::Mat nullMat (hsvMat.size(), cv::DataType<float>::type, cv::Scalar(0));
    hsvChannels.pop_back();
    hsvChannels.push_back(nullMat);
    
    hsvChannels[0].convertTo(hsvChannels[0], cv::DataType<float>::type);
    hsvChannels[1].convertTo(hsvChannels[1], cv::DataType<float>::type);
    cv::merge(hsvChannels.data(), 3, hsMat);
    
    return hsMat;
}

cv::Mat bgr2hsv(cv::Mat bgrMat)
{
    cv::Mat hsvMat;
    
    cv::cvtColor(bgrMat, hsvMat, CV_BGR2HSV);

    return hsvMat;
}

cv::Mat bgrd2hsd(cv::Mat bgrMat, cv::Mat depthMat)
{
    // BGR to HSV
    cv::Mat hsvMat;
    cv::cvtColor(bgrMat, hsvMat, CV_BGR2HSV);
    
    // HSV into separate mats (Hue, sat, and value mats)
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvMat, hsvChannels);
    
    // Transform to float hue, saturation, and depth
    cv::Mat hueFMat, satFMat, depthFMat;
    hsvChannels[0].convertTo(hueFMat, cv::DataType<float>::type);
    hsvChannels[1].convertTo(satFMat, cv::DataType<float>::type);
    depthMat.convertTo(depthFMat, cv::DataType<float>::type);
    
    // Merge into same mat the previous transformed
    cv::Mat hsdMat;
    std::vector<cv::Mat> hsdChannels (3);
    hsdChannels[0] = hueFMat;
    hsdChannels[1] = satFMat;
    hsdChannels[2] = (depthFMat / 4096.f) * 255.f;
    cv::merge(hsdChannels.data(), 3, hsdMat);
    
    return hsdMat;
}

cv::Mat bgrdn2hsdn(cv::Mat bgrMat, cv::Mat depthMat, cv::Mat normalsMat)
{
    // BGR to HSV
    cv::Mat hsvMat;
    cv::cvtColor(bgrMat, hsvMat, CV_BGR2HSV);
    
    // HSV into separate mats (Hue, sat, and value mats)
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvMat, hsvChannels);
    
    // Transform to float hue, saturation, and depth
    cv::Mat hueFMat, satFMat, depthFMat;
    hsvChannels[0].convertTo(hueFMat, cv::DataType<float>::type);
    hsvChannels[1].convertTo(satFMat, cv::DataType<float>::type);
    depthMat.convertTo(depthFMat, cv::DataType<float>::type);
    
    std::vector<cv::Mat> normalsChannels;
    cv::split(normalsMat, normalsChannels);
    cv::Mat thetaFMat, phiFMat;
    normalsChannels[0].convertTo(thetaFMat, cv::DataType<float>::type);
    normalsChannels[1].convertTo(phiFMat, cv::DataType<float>::type);
    
    // Merge into same mat the previous transformed
    cv::Mat hsdnMat;
    std::vector<cv::Mat> hsdnChannels (5);
    hsdnChannels[0] = hueFMat;
    hsdnChannels[1] = satFMat;
    hsdnChannels[2] = (depthFMat / 4096.f) * 255.f;
    hsdnChannels[3] = (thetaFMat / (2.f * 3.141592)) * 255.f;
    hsdnChannels[4] = (phiFMat / (2.f * 3.141592)) * 255.f;
    cv::merge(hsdnChannels.data(), 5, hsdnMat);
    
    return hsdnMat;
}

void translate(const pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud, pcl::PointXYZ t, pcl::PointCloud<pcl::PointXYZ>& cloudCentered)
{
    cloudCentered.width  = pCloud->width;
    cloudCentered.height = pCloud->height;
    cloudCentered.resize( pCloud->width * pCloud->height );

    for (int i = 0; i < pCloud->points.size(); i++)
    {
        cloudCentered.points[i].x = pCloud->points[i].x - t.x;
        cloudCentered.points[i].y = pCloud->points[i].y - t.y;
        cloudCentered.points[i].z = pCloud->points[i].z - t.z;
    }
}

void translate(const pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud, Eigen::Vector4f t, pcl::PointCloud<pcl::PointXYZ>& cloudCentered)
{
    cloudCentered.width  = pCloud->width;
    cloudCentered.height = pCloud->height;
    cloudCentered.resize( pCloud->width * pCloud->height );
    
    for (int i = 0; i < pCloud->points.size(); i++)
    {
        cloudCentered.points[i].x = pCloud->points[i].x - t.x();
        cloudCentered.points[i].y = pCloud->points[i].y - t.y();
        cloudCentered.points[i].z = pCloud->points[i].z - t.z();
    }
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n)
{
    std::ostringstream out;
    out << std::setprecision(n) << a_value;
    return out.str();
}

template <typename T>
std::string to_str(cv::Mat values, std::string separator)
{
    std::string out = to_str(values.at<T>(0,0));
    for (int i = 0; i < values.rows; i++)
        for (int j = ((i > 0) ? 0 : 1); j < values.cols; j++)
            out += separator + to_str(values.at<T>(i,j));
    
    return out;
}

template <typename T>
std::string to_string_with_precision(cv::Mat values, std::string separator, const int n)
{
    std::string out = to_string_with_precision<T>(values.at<T>(0,0), n);
    for (int i = 0; i < values.rows; i++)
        for (int j = ((i > 0) ? 0 : 1); j < values.cols; j++)
            out += separator + to_string_with_precision<T>(values.at<T>(i,j), n);
    
    return out;
}

std::string to_str(int i)
{
	return std::to_string(static_cast<long long>(i));
}


std::string getFilenameFromPath(std::string path)
{
    std::string aux = path;
    const size_t last_slash_idx = aux.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
        aux.erase(0, last_slash_idx + 1);

    const size_t period_idx = aux.rfind('.');
    if (std::string::npos != period_idx)
        aux.erase(period_idx);

    return aux;
}

int searchByName(std::vector<std::string> paths, std::string filename)
{
    for (int i = 0; i < paths.size(); i++)
    {
        if (filename == getFilenameFromPath(paths[i]))
            return i;
    }
    
    return -1;
}

int searchByName(std::vector<std::pair<std::string,std::string> > paths, std::string filename)
{
    for (int i = 0; i < paths.size(); i++)
    {
        if (filename == getFilenameFromPath(paths[i].first) || filename == getFilenameFromPath(paths[i].second))
            return i;
    }
    
    return -1;
}

template<typename T>
cv::Mat wToMat(std::vector<std::vector<T> > w)
{
    cv::Mat mat (w.size(), w[0].size(), cv::DataType<T>::type);
    
    for (int i = 0; i < w.size(); i++)
    {
        assert (w[i].size() == w[0].size());
        for (int j = 0; j < w[i].size(); j++)
        {
            mat.at<T>(i,j) = w[i][j];
        }
    }
    
    return mat;
}

template<>
cv::Mat wToMat(std::vector<std::vector<pcl::PointXYZ> > w)
{
    cv::Mat mat (w.size(), w[0].size(), CV_32FC(3));
    
    for (int i = 0; i < w.size(); i++)
    {
        assert (w[i].size() == w[0].size());
        for (int j = 0; j < w[i].size(); j++)
        {
            mat.at<cv::Vec3f>(i,j) = cv::Vec3f(w[i][j].x, w[i][j].y, w[i][j].z);
        }
    }
    
    return mat;
}

template<typename T>
std::vector<std::vector<T> > matTow(cv::Mat mat)
{
    std::vector<std::vector<T> > w (mat.rows);
    
    for (int i = 0; i < mat.rows; i++)
    {
        w[i].resize(mat.cols);
        for (int j = 0; j < mat.cols; j++)
        {
            w[i][j] = mat.at<T>(i,j);
        }
    }
    
    return w;
}

template<>
std::vector<std::vector<pcl::PointXYZ> > matTow(cv::Mat mat)
{
    std::vector<std::vector<pcl::PointXYZ> > w (mat.rows);
    
    for (int i = 0; i < mat.rows; i++)
    {
        w[i].resize(mat.cols);
        for (int j = 0; j < mat.cols; j++)
        {
            w[i][j].x = mat.at<cv::Vec3f>(i,j)[0];
            w[i][j].y = mat.at<cv::Vec3f>(i,j)[1];
            w[i][j].z = mat.at<cv::Vec3f>(i,j)[2];
        }
    }
    
    return w;
}


template<typename PointT>
pcl::PointXYZ computeCentroid(pcl::PointCloud<PointT> pCloud)
{
    int n = pCloud.points.size();
    
    double x = .0;
    double y = .0;
    double z = .0;
    
    int c = 0;
    for (int i = 0; i < n; ++i)
    {
        if (pCloud.points[i].z > .0)
        {
            x += pCloud.points[i].x;
            y += pCloud.points[i].y;
            z += pCloud.points[i].z;
            c++;
        }
    }
    
    return pcl::PointXYZ(x/c, y/c, z/c);
}

template std::string to_string_with_precision<float>(const float a_value, const int n);
template std::string to_string_with_precision<double>(const double a_value, const int n);

//template std::string to_string<int>(cv::Mat values, std::string separator);
template std::string to_string_with_precision<float>(cv::Mat values, std::string separator, const int n);
template std::string to_string_with_precision<double>(cv::Mat values, std::string separator, const int n);

template cv::Mat wToMat(std::vector<std::vector<int> > w);
template cv::Mat wToMat(std::vector<std::vector<float> > w);
template cv::Mat wToMat(std::vector<std::vector<double> > w);

template std::vector<std::vector<int> > matTow(cv::Mat mat);
template std::vector<std::vector<float> > matTow(cv::Mat mat);
template std::vector<std::vector<double> > matTow(cv::Mat mat);

template void biggestEuclideanCluster<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ>::Ptr, int, int, float, pcl::PointCloud<pcl::PointXYZ>&);
template void biggestEuclideanCluster<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB>::Ptr, int, int, float, pcl::PointCloud<pcl::PointXYZRGB>&);

template pcl::PointXYZ computeCentroid(pcl::PointCloud<pcl::PointXYZ> pCloud);
template pcl::PointXYZ computeCentroid(pcl::PointCloud<pcl::PointXYZRGB> pCloud);