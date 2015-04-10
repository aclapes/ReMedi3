//
//  ReMedi.h
//  remedi2
//
//  Created by Albert Clapés on 13/07/14.
//
//

#ifndef remedi2_ReMedi_h
#define remedi2_ReMedi_h

#include "Sequence.h"
#include "ColorDepthFrame.h"

#include "InteractiveRegisterer.h"
#include "TableModeler.h"
#include "BackgroundSubtractor.h"
#include "BoundingBoxesGenerator.h" // Oscar's

#include "GroundtruthRegion.hpp"
#include "ForegroundRegion.hpp"
#include "Cloudject.hpp"
#include "CloudjectSVMClassificationPipeline.h"

#include "pclxtended.h"
#include "io.h"

#include <set>
#include <iostream>

#include <boost/variant.hpp>

using namespace std;

//
// Typedefs
//

// Each map representing a view its cardinality being: #{frames}
typedef std::map<std::string,std::vector<int> >  ViewInteraction;
typedef std::map<std::string,std::vector<ForegroundRegion> > ViewDetection;
typedef std::map<std::string,std::map<std::string,std::map<std::string,GroundtruthRegion> > > ViewGroundtruth;

// Each map representing a sequence its cardinality being: #{views}
typedef std::map<std::string,ViewInteraction> SequenceInteraction;
typedef std::map<std::string,ViewDetection> SequenceDetection;
typedef std::map<std::string,ViewGroundtruth> SequenceGroundtruth;

// Each map representing the dataset its cardinality being: #{sequences}
typedef std::map<std::string,SequenceInteraction> Interaction;
typedef std::map<std::string,SequenceDetection> Detection;
typedef std::map<std::string,SequenceGroundtruth> Groundtruth;

// PCL structures' typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB ColorPointT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointCloud<pcl::PointXYZRGB> ColorPointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorPointCloudPtr;
typedef pcl::visualization::PCLVisualizer Visualizer;
typedef pcl::visualization::PCLVisualizer::Ptr VisualizerPtr;
typedef pcl::FPFHSignature33 FPFHSignature;
typedef pcl::PFHRGBSignature250 PFHRGBSignature;
typedef pcl::PointCloud<FPFHSignature> FPFHDescription;
typedef pcl::PointCloud<FPFHSignature>::Ptr FPFHDescriptionPtr;
typedef pcl::PointCloud<PFHRGBSignature> PFHRGBDescription;
typedef pcl::PointCloud<PFHRGBSignature>::Ptr PFHRGBDescriptionPtr;

class ReMedi
{
public:
    ReMedi();
    ReMedi(const ReMedi& rhs);
    
    ReMedi& operator=(const ReMedi& rhs);
    
    //
    // Public methods
    //
    
    InteractiveRegisterer::Ptr getRegisterer();
    TableModeler::Ptr getTableModeler();
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::Ptr getBackgroundSubtractor();
    
    /** \brief Set the sequences of frames used to learn the background model, for its further subtraction
     *  \param pBgSeq Background sequence
     */
    void setBackgroundSequence(Sequence<ColorDepthFrame>::Ptr pBgSeq);
    Sequence<ColorDepthFrame>::Ptr getBackgroundSequence();
    

    /** \brief Set the list of sequences to be processed
     *  \param pSequences Input sequences
     */
    void setInputSequences(std::vector<Sequence<ColorDepthFrame>::Ptr> pSequences);
    
    /** \brief Set resolutions
     *  \param xres
     *  \param yres
     */
    void setFramesResolution(int xres, int yres);
    
    /** \brief Set an static frame for multiple purposes
     *  \param fid Frame numerical ID
     */
    void setDefaultFrame(int fid);
    
    /** \brief Set the parameters of the interactive registerer
     * \param  p : number of points used to compute the orthogonal transformation among views
     * \param  wndHeight : window height (visor)
     * \param  wndWidth : window width (visor)
     * \param  vp : number of vertical ports (visor)
     * \param  hp : number of horizontal ports (visor)
     * \param  camDist : camera dist to (0,0,0) (visor)
     * \param  markerRadius : marker sphere radius (visor)
     */
    void setRegistererParameters(int p, int wndHeight, int wndWidth, int vp, int hp, float camDist, float markerRadius);
    
    /** \brief Set the parameters of the table modeler
     * \param  leafsz : leaf size in the voxel grid downsampling (speeds up the normal computation)
     * \param  normrad : neighborhood radius in normals estimation
     * \param  sacIters : num of iterations in RANSAC used for plane estimation
     * \param  sacDist : distance threshold to consider outliers in RANSAC
     * \param  yoffset : tabletop offset
     * \param  border : distance to the border table when considering blobs
     * \param  condifence : statistical removal of tabletop outlier points (along the y dimension)
     */
    void setTableModelerParameters(float leafsz, float normrad, int sacIters, float sacDist, float yoffset, float border, int confidence);
    
    /** \brief Set the parameters of the background subtractor
     *  \param n The number of samples used for modelling
     *  \param m The modality used to model the background (color, color with shadows, or depth)
     *  \param k The maximum number of components per mixture in each pixel
     *  \param lrate The learning rate for the adaptive background model
     *  \param bgratio The background ratio
     *  \param vargen The variance threshold to consider generation of new mixture components
     *  \param level The size (2*level+1) of the kernel convolved to perform mathematical morphology (opening) to the background mask
     */
    void setSubtractorParameters(int n, int m, int k, float lrate, float bgratio, float vargen, float opening);

    void initialize();
    
    // Normal functioning of the system
    void run();

//    static void loadGroundtruth(std::string parent, std::vector<std::string> seqDirnames, std::string subdir, std::string dirname, std::vector<const char*> views, std::vector<const char*> objectsNames, Groundtruth& gt);
    
//    static void segmentBlobs(cv::Mat fgMask, cv::Mat errMask, std::vector<BoundingBox> boxesstd::vector<Blob>& blobs);
//    static void labelBlobs(cv::Mat fgMask, cv::Mat errMask, std::vector<BoundingBox> boxes, std::vector<Annotation> annots, std::vector<Blob>& blobs, bool bDebug = false);
//    static void blobsToCloudjects(ColorDepthFrame::Ptr frame, std::vector<Blob> blobs, std::vector<Cloudject>& cloudjects);


//    static void quantizeDescriptions(std::list<Description> descriptions, int k, QDataSample& qsample);
//    static void quantizeDescriptions(std::list<Description> descriptions, cv::Mat centers, QDataSample& qsample);

//    static void makeTrainingInstances(std::list<Cloudject> cloudjects, std::list<Description> descriptions, SequenceAnnotation annotations, std::list<Cloudject>& cloudjectsTr, std::list<Description>& descriptionsTr);
//    static void makeTestInstances(std::list<Cloudject> cloudjects, std::list<Description> descriptions, std::map<std::string,std::vector<Annotation> > annotationsMap, std::list<Cloudject>& cloudjectsTe, std::list<Description>& descriptionsTe);
    
//    static void getTrainingData(std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::map<std::string,std::map<std::string,SequenceAnnotation> > annotations, std::vector<int> partitions, int p, std::list<Cloudject>& cloudjects, std::list<Description>& descriptions);
//    static void getTestData(std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, /*std::vector<std::vector<std::map<std::string,std::vector<Annotation> > > > annotation,*/ std::vector<int> partitions, int p, std::list<Cloudject>& cloudjects, std::list<Description>& descriptions);

//    static void getTrainingData(std::list<Description> data, cv::Mat partition, int p, std::list<Description>& dataTr);
//    static void getTestData(std::list<Description> data, cv::Mat partition, int p, std::list<Description>& dataTe);
    
//    static cv::Mat getLabels(std::list<std::set<std::pair<std::string,float> > > labels);
//    static cv::Mat getMultilabels(std::list<std::set<std::pair<std::string,float> > > labels);
    
//    static void measurePredictionErrors(std::list<Cloudject> cloudjects, cv::Mat L, int* tp, int* fp, int* fn);
    
//    static void makeForegroundPredictions(std::list<Cloudject> cloudjects, cv::Mat L, std::map<std::string,std::map<std::string,SequencePrediction> >& predictions);
    
//    static bool isHit(Prediction p, Annotation a);
//    static void evaluatePrediction(std::map<std::string,std::map<std::string,SequencePrediction> > predictions, std::map<std::string,std::map<std::string,SequenceAnnotation> > annotations, std::vector<const char*> objectsNames, std::vector<int>& tp, std::vector<int>& fp, std::vector<int>& fn);
//    static void _evaluatePrediction(SequencePrediction predictions, SequenceAnnotation annotations, std::map<std::string,int>& mtp, std::map<std::string,int>& mfp, std::map<std::string,int>& mfn);

    typedef boost::shared_ptr<ReMedi> Ptr;
    
private:
    //
    // Attributes
    //
    
    Sequence<ColorDepthFrame>::Ptr m_pBgSeq;
    std::vector<Sequence<ColorDepthFrame>::Ptr> m_pSequences;
    int m_fID;
    
    int m_XRes, m_YRes;
    
    InteractiveRegisterer::Ptr m_pRegisterer;
    bool m_bSetRegistererCorrespondences;
    
    TableModeler::Ptr m_pTableModeler;
    
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::Ptr m_pBackgroundSubtractor;
};

// Static methods

namespace remedi
{
    void loadSequences(std::string parent, std::vector<std::string>& seqDirnames, std::vector<int>& sbjIds);
    void loadFramesPaths(std::string parent, std::vector<std::string> seqDirnames, std::string subdir, std::string dirname, std::vector<const char*> views, std::vector< std::vector<std::string> >& paths);
    void loadDelaysFile(std::string parent, std::string filename, std::vector<std::vector<int> >& delays);

    void segmentCloudjects(ColorDepthFrame::Ptr frame, cv::Mat foreground, std::vector<Region> boxes, std::vector<Cloudject::Ptr>& cloudjects);

    void loadGroundtruth(const std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, Groundtruth& gt);
    void getGroundtruthForTraining(const std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, const std::vector<const char*> objectsLabels, const Groundtruth& gt, Groundtruth& gtTr);
    
    void loadInteraction(const std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<const char*> objectLabels, Interaction& iact);
    
    void getTrainingCloudjectsWithDescriptor(std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> partitions, int p, std::vector<const char*> annotationLabels, const Groundtruth& gt, std::string descriptorType, std::list<MockCloudject::Ptr>& cloudjects);
    void getTestCloudjectsWithDescriptor(std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> partitions, int p, const Groundtruth& gt, std::string descriptorType, std::list<MockCloudject::Ptr>& cloudjects);

    void getTestSequences(const std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> partitions, int p, std::vector<Sequence<ColorDepthFrame>::Ptr>& sequencesTe);
        
    void findContentions(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, float tol, std::vector<std::vector<std::pair<int,Cloudject::Ptr> > >& contentions, std::vector<std::vector<pclx::Rectangle3D> >& rectangles);
    void getDetectionRectangles(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, bool bRegistrate, std::vector<std::vector<pclx::Rectangle3D> >& rectangles);
    
    void computeScalingFactorsOfCategories(cv::Mat predictions, cv::Mat distsToMargin, cv::Mat& factors);
}

#endif
