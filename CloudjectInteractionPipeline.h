//
//  CloudjectInteractionPipeline.h
//  remedi3
//
//  Created by Albert Clapés on 06/04/15.
//
//

#ifndef __remedi3__CloudjectInteractionPipeline__
#define __remedi3__CloudjectInteractionPipeline__

#include <stdio.h>

#include "constants.h"

#include "Sequence.h"
#include "ColorDepthFrame.h"

#include "BackgroundSubtractor.h"
#include "InteractiveRegisterer.h"
#include "TableModeler.h"
#include "CloudjectSVMClassificationPipeline.h"

#include "GroundtruthRegion.hpp"
#include "ForegroundRegion.hpp"

#include "conversion.h"
#include "cvxtended.h"
#include "pclxtended.h"

#include "io.h"

#include <pcl/filters/voxel_grid.h>
#include <boost/shared_ptr.hpp>

#include <opencv2/opencv.hpp>

//
// Typedefs
// -----------------------------------------------------------------------------
typedef std::map<std::string,std::vector<int> >  ViewInteraction;
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

typedef pcl::VoxelGrid<pcl::PointXYZRGB> VoxelGrid;
typedef boost::shared_ptr<pcl::VoxelGrid<pcl::PointXYZRGB> > VoxelGridPtr;
// -----------------------------------------------------------------------------

//
// Auxiliary classes
//


class Result
{
public:
    int tp, fp, fn, tn;
    
    Result() : tp(0), fp(0), fn(0), tn(0) {}
    Result (const Result& rhs) { *this = rhs; }
    
    Result& operator=(const Result& rhs)
    {
        if (this != &rhs)
        {
            tp = rhs.tp;
            fp = rhs.fp;
            fn = rhs.fn;
            tn = rhs.tn;
        }
        return *this;
    }
    
    Result& operator+=(const Result& rhs)
    {
        tp += rhs.tp;
        fp += rhs.fp;
        fn += rhs.fn;
        tn += rhs.tn;
        
        return *this;
    }
    
    cv::Vec4f toVector()
    {
        return cv::Vec4f(tp,fp,fn,tn);
    }
    
    void clear() { tp = fp = fn = tn = 0; }
};


//
// Main class
//

class CloudjectInteractionPipeline
{
public:
    CloudjectInteractionPipeline();
    CloudjectInteractionPipeline(const CloudjectInteractionPipeline& rhs);
    
    CloudjectInteractionPipeline& operator=(const CloudjectInteractionPipeline& rhs);
    
    void setInputSequences(const std::vector<Sequence<ColorDepthFrame>::Ptr>& sequences);
    void setCategories(const std::vector<const char*>& categories);
    
    void setBackgroundSubtractor(BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::Ptr bs);
    void setInteractiveRegisterer(InteractiveRegisterer::Ptr ir);
    void setTableModeler(TableModeler::Ptr tm);
    
    void setGroundtruth(const Groundtruth& gt);
    void setInteractionGroundtruth(const Interaction& iact);
    
    void setClassificationPipeline(CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr pipeline);
    
    void setDetectionCorrespondenceThresh(float thresh); // tolerance when matching to groundtruth
    void setCorrespondenceCriterion(int criterion); // but how we match against groundtruth?
    void setInteractionThresh(float thresh); // distance subject-object to declare interaction
    
    void setMultiviewStrategy(int strategy); // monocular or multiview
    void setMultiviewLateFusionStrategy(int strategy);
    void setMultiviewCorrespondenceThresh(float thresh);
    
    void setLeafSize(Eigen::Vector3f leafSize);
    
    
    void setMultiviewLateFusionNormalization(std::vector<std::vector<float> > scalings);
    
    void setValidationParameters(std::vector<float> detectionCorrespThreshs,
                                 std::vector<float> correspCriteria,
                                 std::vector<float> interactionThreshs,
                                 std::vector<float> multiviewLateFusionStrategies,
                                 std::vector<float> multiviewCorrespThreshs);
    
    void setValidationInteractionOverlapCriterion(int crit);
    
    void validateDetection();
    void validateInteraction();
    void predictInteraction();

    float getValidationPerformance();
    
    void save(std::string filename, std::string extension = ".yml");
    bool load(std::string filename, std::string extension = ".yml");
    
    void detect();
    std::vector<std::vector<Result> > getDetectionResults();
    void detectInteraction();
    std::vector<std::vector<Result> > getInteractionResults();
    
    typedef boost::shared_ptr<CloudjectInteractionPipeline> Ptr;
    
    enum { DETECT_MONOCULAR, DETECT_MULTIVIEW };
    enum { MULTIVIEW_LF_OR, MULTIVIEW_LFSCALE_DEV, MULTIVIEW_LF_FURTHEST, MULTIVIEW_LFSCALE_SUMDIV };
    enum { CORRESP_INC, CORRESP_OVL };
    enum { INTERACTION_OVL_OVERALL, INTERACTION_OVL_BEGINEND };
    
private:
    //
    // Attributes
    //
    
    std::vector<Sequence<ColorDepthFrame>::Ptr> m_Sequences;
    std::vector<const char*> m_Categories;
    
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2,ColorDepthFrame>::Ptr m_pSubtractor;
    InteractiveRegisterer::Ptr m_pRegisterer;
    TableModeler::Ptr m_pTableModeler;
    CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr m_ClassificationPipeline;
    
    Groundtruth m_Gt;
    Interaction m_IAct;
    int m_CorrespCriterion;
    float m_DetectionCorrespThresh;
    float m_InteractionThresh;
    
    int m_MultiviewStrategy;
    int m_MultiviewLateFusionStrategy;
    float m_MultiviewCorrespThresh;
    Eigen::Vector3f m_LeafSize;
    
    std::vector<std::vector<float> > m_LateFusionScalings; // in case of multiview
    
    std::vector<std::vector<float> > m_ValParams;
    std::vector<std::vector<float> > m_ValCombs;
    int m_ValInteractionOvlCriterion;
    float m_ValPerf;
    int m_ValIdx;
    
    float m_Perf;
    
    std::vector<std::vector<Result> > m_DetectionResults;
    std::vector<std::vector<Result> > m_InteractionCombResults;
    
    std::vector<std::list<std::vector<int> > > m_InteractionPredictions; // views x frames x Binary interaction outputs
    std::vector<cv::Mat> m_InteractionResults; // as many as views, combinations x categories x 4 (channesls)
    
    //
    // Private methods
    //
    
    void setValidationParametersCombination(std::vector<float> combination); // of parameters
    
    void detectMonocular();
    void detectMultiview();
    void detectInteractionMonocular();
    void detectInteractionMultiview();
    
    void findActors(std::vector<ColorPointCloudPtr> actors, std::vector<ColorPointCloudPtr> interactors, std::vector<ColorPointCloudPtr>& nonInteractedActors, Eigen::Vector3f leafSize, std::vector<ColorPointCloudPtr>& interactedActors);
    void findInteractions(std::vector<ColorPointCloudPtr> candidates, std::vector<ColorPointCloudPtr> interactors, std::vector<bool>& mask, Eigen::Vector3f leafSize);
    bool isInteractive(ColorPointCloudPtr tabletopRegionCluster, ColorPointCloudPtr interactionCloud, float tol);
    
    void findCorrespondences(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, float tol, std::vector<std::vector<std::pair<int,Cloudject::Ptr> > >& correspondences, std::vector<std::vector<VoxelGridPtr> >& grids);
    void getDetectionGrids(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, bool bRegistrate, std::vector<std::vector<VoxelGridPtr> >& grids);
    
    void maxPooling(std::vector<Cloudject::Ptr> correspondences, std::vector<int>& indices);
    void maxPooling(std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences, std::vector<int>& indices);
    
    void interactionFromNegativeDetections(std::vector<int> negatives, std::vector<int>& interaction);
    
    void evaluateDetectionInFrame(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& gt, std::vector<Cloudject::Ptr> detections, std::vector<int> indices, Eigen::Vector3f leafSize, std::vector<Result>& result);
    void evaluateDetectionInFrame(const std::vector<ColorDepthFrame::Ptr>& frames, const vector<std::map<std::string,std::map<std::string,GroundtruthRegion> > >& gt, std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences, std::vector<int> indices, Eigen::Vector3f leafSize, std::vector<Result>& result);
    
    void evaluateInteractionInFrame(std::vector<int> prediction, std::vector<int> groundtruth, std::vector<Result>& result);
    
    template<typename RegionT>
    VoxelGridPtr computeGridFromRegion(ColorDepthFrame::Ptr frame, RegionT region, Eigen::Vector3f leafSize, bool bRegistrate = false);
    void precomputeGrids(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& annotations, Eigen::Vector3f leafSize, std::map<std::string,std::map<std::string,VoxelGridPtr> >& grids, bool bRegistrate = false);
    void precomputeGrids(ColorDepthFrame::Ptr frame, const std::vector<ForegroundRegion>& detections, Eigen::Vector3f leafSize, std::vector<VoxelGridPtr>& grids, bool bRegistrate = false);
    
    void fuseCorrespondences(const std::vector<std::vector<std::pair<int, Cloudject::Ptr> > >& correspondences);
    
    void firstDerivative(std::vector<int> inputPrev, std::vector<int> inputCurr, std::vector<int>& derivative);
    
    void findInclusions(std::vector<Cloudject::Ptr> cloudjectsIn, std::vector<ColorPointCloudPtr> clouds, Eigen::Vector3f leafSize, std::vector<Cloudject::Ptr>& cloudjectsOut);
};

#endif /* defined(__remedi3__CloudjectInteractionPipeline__) */
