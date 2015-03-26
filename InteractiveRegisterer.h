#pragma once

#include "InteractiveRegisterer.h"
#include "DepthFrame.h"
#include "ColorFrame.h"
#include "ColorDepthFrame.h"
#include "Sequence.h"
#include "constants.h"

#define MEASURE_FUNCTION_TIME
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/common/distances.h>
#include <pcl/correspondence.h>
#include <pcl/io/pcd_io.h>

#include <pcl/filters/voxel_grid.h>

#include <opencv2/opencv.hpp>

using namespace std;

//static const float g_Colors[][3] = {
//    {1, 0, 0},
//    {0, 1, 0},
//    {0, 0, 1},
//    {1, 0, 1},
//    {1, 1, 0},
//    {0, 1, 1},
//    {.5, 0, 0},
//    {0, .5, 0},
//    {0, 0, .5},
//    {1, 0, .5},
//    {1, .5, 0},
//    {1, .5, .5},
//    {0, 1, .5},
//    {.5, 1, 0},
//    {.5, 0, .5},
//    {0, .5, .5}
//};

//////////////////////////////////////////////

class InteractiveRegisterer
{
    typedef pcl::PointXYZ Point;
    typedef pcl::PointXYZRGB ColorPoint;
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
    typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
    typedef pcl::PointCloud<pcl::PointXYZRGB> ColorPointCloud;
    typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorPointCloudPtr;
    typedef pcl::visualization::PCLVisualizer Visualizer;
    typedef pcl::visualization::PCLVisualizer::Ptr VisualizerPtr;
    
public:    
	InteractiveRegisterer();
    InteractiveRegisterer(const InteractiveRegisterer& rhs);
    
    InteractiveRegisterer& operator=(const InteractiveRegisterer& rhs);
    
    /** \brief Set the number of correspondences
     */
	void setNumPoints(int numOfPoints);
    /** \brief Set the frames used to establish the correspondences
     */
    void setInputFrames(vector<ColorDepthFrame::Ptr> frames);
    
    /** \brief Set the distribution of the viewports in the visualizer
     * \param wndHeight window height (visor)
     * \param wndWidth window width (visor)
     * \param vp number of vertical ports (visor)
     * \param hp number of horizontal ports (visor)
     * \param camDist camera dist to (0,0,0) (visor)
     * \param markerRadius marker sphere radius (visor)
     */
    void setVisualizerParameters(int wndHeight, int wndWidth, int vp, int hp, float camDist, float markerRadius);
    
    /** \brief Manually interact with a visualizer to place the markers that serve as correspondences
     */
    void setCorrespondencesManually();
    void saveCorrespondences(string path, string extension);
    void loadCorrespondences(string path, string extension);
    
    /** \brief Estimate the orthogonal transformations in the n-1 views respect to the 0-th view
     */
    void computeTransformations();
    
    /** \brief Registrate the member frames
     *  \param frames A vector of V depth frames
     */
    void registrate(vector<DepthFrame::Ptr>& frames);
    
    /** \brief Registrate the member frames
     *  \param frames A vector of V colordepth frames
     */
    void registrate(vector<ColorDepthFrame::Ptr>& frames);
    
    /** \brief Set the registration of the several views in the sequence
     *  \param seq A sequence of Depth frames
     */
    void registrate(Sequence<DepthFrame>& regSeq);
    
    /** \brief Set the registration of the several views in the sequence
     *  \param seq An sequence of ColorDepth frames
     */
    void registrate(Sequence<ColorDepthFrame>& regSeq);
    
    /** \brief Registers all the frames to the 0-th frame and return the corresponding registered clouds
     *  \param pUnregFrames A list of unregistered clouds
     *  \param pRegClouds A list of registered clouds
     */
    void registrate(vector<ColorDepthFrame::Ptr> pUnregFrames, vector<PointCloudPtr>& pRegClouds);
    
    /** \brief Registers all the clouds to the 0-th cloud and return the registered clouds
     *  \param pUnregClouds A list of unregistered clouds
     *  \param pRegClouds A list of registered clouds
     */
    void registrate(vector<PointCloudPtr> pUnregClouds, vector<PointCloudPtr>& pRegClouds);

    /** \brief Set the position of the camera in the PCLVisualizer
     */
    void setDefaultCamera(VisualizerPtr pViz, int vid);
    
    
    typedef boost::shared_ptr<InteractiveRegisterer> Ptr;
    
private:
    /** \brief Callback function to deal with keyboard presses in visualizer
     */
    void keyboardCallback(const pcl::visualization::KeyboardEvent& event, void* ptr);
    /** \brief Callback function to deal with mouse presses in visualizer
     */
    void mouseCallback(const pcl::visualization::MouseEvent& event, void* ptr);
    /** \brief Callback function to deal with point picking actions in visualizer (shift+mouse press)
     */
    void ppCallback(const pcl::visualization::PointPickingEvent& event, void* ptr);
    
    /** \brief Get the set of first correspondences across the views
     *  \return correspondences List containing the first correspondences
     */
    vector<pcl::PointXYZ> getFirstCorrespondences();
    
    /** \brief Compute the orthogonal tranformation of a set of points to another (from src to target)
     * \param pSrcMarkersCloud Source set of markers to register to target
     * \param pTgtMarkersCloud Target set of markers
     * \param T The 4x4 orthogonal transformation matrix
     */
    void getTransformation(const PointCloudPtr pSrcMarkersCloud, const PointCloudPtr pTgtMarkersCloud, Eigen::Matrix4f& T);

    
    //
    // Members
    //
    
    // Number of points to compute the transformation
    int m_NumOfPoints;

    bool m_bMark;   // mark keyboard button is down
    
    int m_Pendents;
    pcl::CorrespondencesPtr m_pCorrespondences;
    
    VisualizerPtr m_pViz;
    vector<int> m_VIDs; // viewport identifiers
    int m_WndHeight, m_WndWidth;
    int m_Vp, m_Hp;
    float m_CameraDistance;
    float m_MarkerRadius;
    int m_Viewport; // current viewport

    vector<ColorDepthFrame::Ptr> m_pFrames;
    
    vector<ColorPointCloudPtr> m_pClouds;
    vector<PointCloudPtr> m_pMarkers;

    vector<Eigen::Matrix4f> m_Transformations;
    vector<Eigen::Matrix4f> m_ITransformations;
};