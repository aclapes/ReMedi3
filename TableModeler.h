#pragma once

#include "DepthFrame.h"
#include "ColorDepthFrame.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>

using namespace std;

class TableModeler
{
    typedef pcl::PointXYZ PointT;
    typedef pcl::PointXYZRGB ColorPointT;
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
    typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
    typedef pcl::PointCloud<pcl::PointXYZRGB> ColorPointCloud;
    typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorPointCloudPtr;
    typedef pcl::visualization::PCLVisualizer Visualizer;
    typedef pcl::visualization::PCLVisualizer::Ptr VisualizerPtr;
    
public:
	TableModeler();
    TableModeler(const TableModeler& rhs);
    
    TableModeler& operator=(const TableModeler& rhs);
    
	void setInputFrames(vector<DepthFrame::Ptr> frames);
    void setInputFrames(vector<ColorDepthFrame::Ptr> pFrames);

	void setLeafSize(float);
	void setNormalRadius(float);
	void setSACIters(int);
	void setSACDistThresh(float);
	void setYOffset(float);
    void setInteractionBorder(float);
    void setConfidenceLevel(int level);
    
	void model();
    
    void maskTabletop(vector<DepthFrame::Ptr>& frames);
    void getTabletopMask(vector<DepthFrame::Ptr> frames, vector<cv::Mat>& tops);
    void getTabletopMask(vector<ColorDepthFrame::Ptr> frames, vector<cv::Mat>& tops);
    void getTabletopMask(DepthFrame::Ptr pFrame, cv::Mat& top);
    void segmentTabletopCloud(PointCloudPtr pCloud, PointCloud& top);
    
    void maskInteraction(vector<DepthFrame::Ptr>& frames);
    void getInteractionMask(vector<DepthFrame::Ptr> frames, vector<cv::Mat>& interactions);
    void getInteractionMask(vector<ColorDepthFrame::Ptr> frames, vector<cv::Mat>& interactions);
    void getInteractionMask(DepthFrame::Ptr pFrame, cv::Mat& interaction);
    void segmentInteractionCloud(PointCloudPtr pCloud, PointCloud& interaction);
    
    typedef boost::shared_ptr<TableModeler> Ptr;
    
private:
	void estimate(PointCloudPtr pCloud, float leafSize, PointT reference, PointCloud& plane, PointCloud& nonPlane, pcl::ModelCoefficients& coefficients);
    
    /** \brief Get the orthogonal transformation of a plane so as translate it to the (0,0,0) and to align its normal to one of the coordinate system axis
     *  \param pPlaneCoefficients The coefficients of the plane (normal's components and bias)
     *  \param T The 3D affine transformation itself
     */
    void getTransformation(PointCloudPtr pPlane, pcl::ModelCoefficients::Ptr pCoefficients, PointT referencePoint, PointT contourPoint, Eigen::Affine3f& T);
    
    /** \brief Get the 3D bounding box containing a plane
     *  \param pPlane A plane
     *  \param d Dimension in which the plane is constant
     *  \param confidence Confidence level to discard outliers in the d dimension
     *  \param min Minimum value of the 3D bounding box
     *  \param max Maximum value of the 3D bounding box
     */
    void getMinMax3D(PointCloudPtr pPlane, int d, int confidence, PointT& min, pcl::PointXYZ& max);
    
    /** \brief Determines wheter or not a cloud includes the origin, i.e. the (0,0,0) point (or a very near one)
     *  \param pCloud A cloud
     *  \return Whether or not includes the origin
     */
    bool isCloudIncludingOrigin(PointCloudPtr pCloud, float leafSize);
    
    /** \brief Determines wheter or not a cloud includes the reference point
     *  \param pCloud A cloud
     *  \param reference A reference point
     *  \return Whether or not includes the reference point
     */
    bool isCloudIncludingReference(PointCloudPtr pCloud, float leafSize, PointT reference);

    /** \brief Filter the points given a 3D bounding box.
     *  \param pCloud A cloud to filter
     *  \param min The minimum of the box
     *  \param max The maximum of the box
     *  \param cloudFiltered Points of the cloud kept (insiders)
     *  \param cloudRemoved Points of the cloud removed (outsiders)
     */
    void filter(PointCloudPtr pCloud, PointT min, PointT max, PointCloud& cloudFiltered, PointCloud& cloudRemoved);
    
    /** \brief Extract the contour of the potential table plane
     *  \param pDepthFrame Frame in which the table plane is present
     *  \param max closingLevel In case the table plane cloud is containing small holes
     *  \param tableCloud The extracted table plane
     *  \param tableCoefficients The four table cofficients, i.e. Ax + By + Cz + D = 0
     *  \param tableContourCloud The organized cloud representing the table plane's contour line
     */
    void extractTableContour(DepthFrame::Ptr pDepthFrame, int closingLevel, PointCloud& planeCloud, pcl::ModelCoefficients& tableCoefficients, PointCloud& contourCloud);
    
    /** \brief Find the closest point of the contour to the reference point
     *  \param pContourCloud The organized cloud representing a contour line
     *  \param referencePoint Reference point
     *  \param nearestContourPoint Point of contour cloud closest to the refrence
     */
    void findClosestContourPointToReference(PointCloudPtr pContourCloud, PointT referencePoint, PointT& closestPoint);
    
    /** \brief Segments the cloud region contained in a 3D bounding box bounded by (min, max)
     *  \param pCloud The cloud to segment
     *  \param min The min 3D point
     *  \param max The max 3D point
     *  \param segmentedCloud The segmented part
     */
    void minMaxSegmentation(PointCloudPtr pCloud, PointT min, PointT max, PointCloud& segmentedCloud);
    
////    void transform(pcl::PointCloud<pcl::PointXYZ>::Ptr pPlane, Eigen::Affine3f transformation, pcl::PointCloud<pcl::PointXYZ> planeTransformed);
//

//
//	void segmentTableTop(pcl::PointCloud<pcl::PointXYZ>::Ptr,
//                         pcl::PointXYZ, pcl::PointXYZ, float offset,
//                         Eigen::Affine3f, Eigen::Affine3f,
//                         pcl::PointCloud<pcl::PointXYZ>&);
//    

//
//    void segmentInteractionRegion(pcl::PointCloud<pcl::PointXYZ>::Ptr,
//                                  pcl::PointXYZ, pcl::PointXYZ, float offset,
//                                  Eigen::Affine3f, Eigen::Affine3f,
//                                  pcl::PointCloud<pcl::PointXYZ>&);
//
//	pcl::PointCloud<pcl::PointXYZ>::Ptr m_pCloudA, m_pCloudB;
////    pcl::PointXYZ m_originA, m_originB;
    
    
    void visualizePlaneEstimation(PointCloudPtr pScene, PointCloudPtr pTable, PointCloudPtr pSceneTransformed, PointCloudPtr pTableTransformed, PointT min, PointT max);
    
    //
    // Members
    //
    
	float m_LeafSize;
	float m_NormalRadius;
	float m_SACIters;
	float m_SACDistThresh;
    float m_InteractionBorder;
    float m_YOffset;
    float m_ConfidenceLevel;
    
    vector<DepthFrame::Ptr> m_pFrames;
    
    Eigen::Affine3f m_Transformation;
    PointT m_Min, m_Max;
    
//    pcl::ModelCoefficients::Ptr m_pPlaneCoeffsA, m_pPlaneCoeffsB;
//
//	pcl::PointXYZ m_MinA, m_MaxA, m_MinB, m_MaxB;
//	float m_OffsetA, m_OffsetB; // sum to the axis corresponding to plane's normal (probably the y dimension)
//	Eigen::Affine3f m_ytonA, m_ytonB; // y axis to n plane normal vector transformtion
//    Eigen::Affine3f m_ytonAInv, m_ytonBInv; // y axis to n plane normal vector transformtion
//
//
//    bool m_bLimitsNegative;
};