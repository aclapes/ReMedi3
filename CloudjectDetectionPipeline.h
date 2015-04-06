//
//  CloudjectDetectionPipeline.h
//  remedi3
//
//  Created by Albert Clapés on 06/04/15.
//
//

#ifndef __remedi3__CloudjectDetectionPipeline__
#define __remedi3__CloudjectDetectionPipeline__

#include <stdio.h>

#include "constants.h"

#include "Sequence.h"
#include "ColorDepthFrame.h"

#include "InteractiveRegisterer.h"
#include "TableModeler.h"
#include "CloudjectSVMClassificationPipeline.h"

#include "GroundtruthRegion.hpp"
#include "ForegroundRegion.hpp"

#include "conversion.h"
#include "cvxtended.h"
#include "pclxtended.h"

#include "io.h"

//
// Typedefs
// -----------------------------------------------------------------------------
typedef std::map<std::string,std::vector<bool> >  ViewInteraction;
typedef std::map<std::string,std::vector<ForegroundRegion> > ViewDetection;
typedef std::map<std::string,std::map<std::string,std::map<std::string,GroundtruthRegion> > > ViewGroundtruth;

typedef std::map<std::string,ViewInteraction> SequenceInteraction;
typedef std::map<std::string,ViewDetection> SequenceDetection;
typedef std::map<std::string,ViewGroundtruth> SequenceGroundtruth;

typedef std::map<std::string,SequenceInteraction> Interaction;
typedef std::map<std::string,SequenceDetection> Detection;
typedef std::map<std::string,SequenceGroundtruth> Groundtruth;

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB ColorPointT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointCloud<pcl::PointXYZRGB> ColorPointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorPointCloudPtr;
// -----------------------------------------------------------------------------

//
// Auxiliary classes
//

class DetectionResult
{
public:
    int tp;
    int fp;
    int fn;
    
    DetectionResult() : tp(0), fp(0), fn(0) {}
    DetectionResult (const DetectionResult& rhs) { *this = rhs; }
    
    DetectionResult& operator=(const DetectionResult& rhs)
    {
        if (this != &rhs)
        {
            tp = rhs.tp;
            fp = rhs.fp;
            fn = rhs.fn;
        }
        return *this;
    }
    
    DetectionResult& operator+=(const DetectionResult& rhs)
    {
        tp += rhs.tp;
        fp += rhs.fp;
        fn += rhs.fn;
        
        return *this;
    }
    
    cv::Vec3f toVector()
    {
        return cv::Vec3f(tp,fp,fn);
    }
};


//
// Main class
//

class CloudjectDetectionPipeline
{
public:
    CloudjectDetectionPipeline();
    CloudjectDetectionPipeline(const CloudjectDetectionPipeline& rhs);
    
    CloudjectDetectionPipeline& operator=(const CloudjectDetectionPipeline& rhs);
    
    void setInputSequences(const std::vector<Sequence<ColorDepthFrame>::Ptr>& sequences);
    void setCategories(const std::vector<const char*>& categories);
    
    void setInteractiveRegisterer(InteractiveRegisterer::Ptr ir);
    void setTableModeler(TableModeler::Ptr tm);
    
    void setDetectionGroundtruth(const Groundtruth& gt);
    
    void setClassificationPipeline(CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr pipeline);
    
    void setMultiviewDetectionStrategy(int strategy);
    void setMultiviewActorCorrespondenceThresh(float thresh);
    void setInteractionThresh(float thresh);
    
    void setMultiviewLateFusionNormalization(std::vector<std::vector<float> > scalings);
    
    void detect();
    
    std::vector<DetectionResult> getDetectionResults();
    
    typedef boost::shared_ptr<CloudjectDetectionPipeline> Ptr;
    
private:
    //
    // Attributes
    //
    
    std::vector<Sequence<ColorDepthFrame>::Ptr> m_Sequences;
    std::vector<const char*> m_Categories;
    
    InteractiveRegisterer::Ptr m_pRegisterer;
    TableModeler::Ptr m_pTableModeler;
    
    Groundtruth m_Gt;
    
    int m_MultiviewDetectionStrategy;
    float m_MultiviewActorCorrespThresh;
    float m_InteractionThresh;
    
    std::vector<std::vector<float> > m_LateFusionScalings;
    
    CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr m_ClassificationPipeline;
    
    std::vector<DetectionResult> m_DetectionResults;
    
    //
    // Private methods
    //
    
    void detectMonocular();
    void detectMultiview();
    
    void findActors(std::vector<Cloudject::Ptr> candidates, std::vector<ColorPointCloudPtr> interactors, std::vector<Cloudject::Ptr>& actors, float leafSize, std::vector<Cloudject::Ptr>& interactedActors);
    void findInteractions(std::vector<ColorPointCloudPtr> candidates, std::vector<ColorPointCloudPtr> interactors, std::vector<bool>& mask, float leafSize);
    bool isInteractive(ColorPointCloudPtr tabletopRegionCluster, ColorPointCloudPtr interactionCloud, float tol);
    
    void findCorrespondences(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, float tol, std::vector<std::vector<std::pair<int,Cloudject::Ptr> > >& correspondences, std::vector<std::vector<PointT> >& positions);
    void getDetectionPositions(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, bool bRegistrate, std::vector<std::vector<pcl::PointXYZ> >& positions);

    void evaluateFrame(const std::map<std::string,std::map<std::string,GroundtruthRegion> >& gt, const std::vector<ForegroundRegion>& dt, DetectionResult& result);
    void evaluateFrame(const vector<std::map<std::string,std::map<std::string,GroundtruthRegion> > >& gt, const vector<std::map<std::string,std::map<std::string,pcl::PointXYZ> > >& gtCentroids, const std::vector<std::vector<std::pair<int, Cloudject::Ptr> > >& correspondences, DetectionResult& result);
    void evaluateFrame2(const std::vector<ColorDepthFrame::Ptr>& frames, const vector<std::map<std::string,std::map<std::string,GroundtruthRegion> > >& gt, std::vector<std::vector<std::pair<int, Cloudject::Ptr> > >& correspondences, std::vector<std::vector<DetectionResult> >& result);
    void evaluateFrame2(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& gt, std::vector<ForegroundRegion>& dt, std::vector<DetectionResult>& result);
    
    template<typename RegionT>
    pclx::Rectangle3D computeRectangleFromRegion(ColorDepthFrame::Ptr frame, RegionT region);
    template<typename RegionT>
    pcl::PointXYZ computeCentroidFromRegion(ColorDepthFrame::Ptr frame, RegionT region);
    void precomputeRectangles3D(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& annotations, std::map<std::string,std::map<std::string,pclx::Rectangle3D> >& rectangles);
    void precomputeRectangles3D(ColorDepthFrame::Ptr frame, const std::vector<ForegroundRegion>& detections, std::vector<pclx::Rectangle3D>& rectangles);
    void precomputeCentroids(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& annotations, std::map<std::string,std::map<std::string,pcl::PointXYZ> >& centroids);
    void precomputeCentroids(ColorDepthFrame::Ptr frame, const std::vector<ForegroundRegion>& detections, std::vector<pcl::PointXYZ>& centroids);
};

#endif /* defined(__remedi3__CloudjectDetectionPipeline__) */
