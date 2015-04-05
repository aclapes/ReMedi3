#pragma once

// OpenCV
#include <opencv2/opencv.hpp>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>

void normalsToAngles(pcl::PointCloud<pcl::Normal>::Ptr pNormals, cv::Mat& angles);

void pcl2cv(pcl::PointXYZ p, cv::Mat& m);
void pcl2cv(pcl::PointXYZRGB p, cv::Mat& m);

void ProjectiveToRealworld(pcl::PointXYZ p, int xres, int yres, int x0, int y0, pcl::PointXYZ& rw);
void RealworldToProjective(pcl::PointXYZ rw, int xres, int yres, int x0, int y0, pcl::PointXYZ& p);
void ProjectiveToRealworld(pcl::PointXYZ p, int xres, int yres, pcl::PointXYZ& rw);
void RealworldToProjective(pcl::PointXYZ rw, int xres, int yres, pcl::PointXYZ& p);

void ProjectiveToRealworld(pcl::PointCloud<pcl::PointXYZ>::Ptr pProjCloud, pcl::PointCloud<pcl::PointXYZ>& realCloud);
void RealworldToProjective(pcl::PointCloud<pcl::PointXYZ>::Ptr pRealCloud, pcl::PointCloud<pcl::PointXYZ>& projCloud);

void EigenToPointXYZ(Eigen::Vector4f eigen, pcl::PointXYZ& p);
pcl::PointXYZ EigenToPointXYZ(Eigen::Vector4f eigen);

void MatToPointCloud(cv::Mat depth, pcl::PointCloud<pcl::PointXYZ>&);

void MatToPointCloud(cv::Mat depth, cv::Mat mask, pcl::PointCloud<pcl::PointXYZ>&);

void MatToColoredPointCloud(cv::Mat depth, cv::Mat color, pcl::PointCloud<pcl::PointXYZRGB>& cloud);
void MatToColoredPointCloud(cv::Mat depth, cv::Mat color, cv::Mat mask, pcl::PointCloud<pcl::PointXYZRGB>& cloud);

void PointCloudToMat(pcl::PointCloud<pcl::PointXYZ>&, cv::Mat&);
void PointCloudToMat(pcl::PointCloud<pcl::PointXYZ>::Ptr, int height, int width, cv::Mat&);
void ColoredPointCloudToMat(pcl::PointCloud<pcl::PointXYZRGB>::Ptr, int height, int width, cv::Mat&);
void ColoredPointCloudToMat(pcl::PointCloud<pcl::PointXYZRGB>::Ptr, int height, int width, int x0, int y0, cv::Mat&);

void MaskDensePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr, cv::Mat, pcl::PointCloud<pcl::PointXYZ>&);

void passthroughFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointXYZ min, pcl::PointXYZ max, 
	pcl::PointCloud<pcl::PointXYZ>& cloudF);

void enclosure(cv::Mat src, cv::Mat& dst, int size);

template<typename PointT>
void biggestEuclideanCluster(typename pcl::PointCloud<PointT>::Ptr cloud, int minSz, int maxSz, float clusterTol,
	pcl::PointCloud<PointT>& cluster);

float euclideanDistance(Eigen::Vector4f, Eigen::Vector4f);

cv::Mat bgr2hs(cv::Mat bgrMat);
cv::Mat bgr2hsv(cv::Mat bgrMat);
cv::Mat bgr2hs0(cv::Mat bgrMat);
cv::Mat bgrd2hsd(cv::Mat bgrMat, cv::Mat depthMat);
cv::Mat bgrdn2hsdn(cv::Mat bgrMat, cv::Mat depthMat, cv::Mat normalsMat);

void translate(const pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud, pcl::PointXYZ t, pcl::PointCloud<pcl::PointXYZ>& cloudCentered);
void translate(const pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud, Eigen::Vector4f t, pcl::PointCloud<pcl::PointXYZ>& cloudCentered);

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6);

template<typename T>
std::string to_str(cv::Mat values, std::string separator);
template<typename T>
std::string to_string_with_precision(cv::Mat values, std::string separator, const int n = 6);

std::string getFilenameFromPath(std::string path);

std::string to_str(int i);

int searchByName(std::vector<std::string> paths, std::string filename);
int searchByName(std::vector<std::pair<std::string,std::string> > paths, std::string filename);

template<typename T>
cv::Mat wToMat(std::vector<std::vector<T> > w);
template<typename T>
std::vector<std::vector<T> > matTow(cv::Mat mat);

template<typename PointT>
pcl::PointXYZ computeCentroid(const pcl::PointCloud<PointT>& pCloud);
template<typename PointT>
void computeRectangle3D(const pcl::PointCloud<PointT>& pCloud, pcl::PointXYZ& min, pcl::PointXYZ& max);