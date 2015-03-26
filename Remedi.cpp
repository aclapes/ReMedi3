//
//  ReMedi.cpp
//  remedi2
//
//  Created by Albert Clapés on 13/07/14.
//
//

#include "ReMedi.h"
#include "cvxtended.h"
#include "statistics.h"

#include <boost/timer.hpp>
#include <boost/filesystem.hpp>
#include <boost/assign/std/vector.hpp>

#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/features/pfhrgb.h>

using namespace boost::assign;
namespace fs = boost::filesystem;

//#define DEBUG_TRAINING_CONSTRUCTION_OVERLAPS
//#define DEBUG_TRAINING_CONSTRUCTION_SELECTION

ReMedi::ReMedi()
: m_fID(0),
  m_pRegisterer(new InteractiveRegisterer),
  m_bSetRegistererCorrespondences(false),
  m_pTableModeler(new TableModeler)
{
    m_pBackgroundSubtractor = BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::Ptr(new BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>);
}

ReMedi::ReMedi(const ReMedi& rhs)
{
    *this = rhs;
}

ReMedi& ReMedi::operator=(const ReMedi& rhs)
{
    if (this != &rhs)
    {
        m_pBgSeq = rhs.m_pBgSeq;
        m_pSequences = rhs.m_pSequences;
        m_fID = rhs.m_fID;
        
        m_pRegisterer = rhs.m_pRegisterer;
        m_bSetRegistererCorrespondences = rhs.m_bSetRegistererCorrespondences;
        
        m_pTableModeler = rhs.m_pTableModeler;
        
        m_pBackgroundSubtractor = rhs.m_pBackgroundSubtractor;
    }
    
    return *this;
}

InteractiveRegisterer::Ptr ReMedi::getRegisterer()
{
    return m_pRegisterer;
}

TableModeler::Ptr ReMedi::getTableModeler()
{
    return m_pTableModeler;
}

BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::Ptr ReMedi::getBackgroundSubtractor()
{
    return m_pBackgroundSubtractor;
}

/** \brief Set the sequences of frames used to learn the background model, for its further subtraction
 *  \param pBgSeq Background sequence
 */
void ReMedi::setBackgroundSequence(Sequence<ColorDepthFrame>::Ptr pBgSeq)
{
    m_pBgSeq = pBgSeq;
}

Sequence<ColorDepthFrame>::Ptr ReMedi::getBackgroundSequence()
{
    return m_pBgSeq;
}

/** \brief Set the list of sequences to be processed
 *  \param pSequences Input sequences
 */
void ReMedi::setInputSequences(std::vector<Sequence<ColorDepthFrame>::Ptr> pSequences)
{
    m_pSequences = pSequences;
}

/** \brief Set resolutions
 *  \param xres
 *  \param yres
 */
void ReMedi::setFramesResolution(int xres, int yres)
{
    m_XRes = xres;
    m_YRes = yres;
}

/** \brief Set an static frame for multiple purposes
 *  \param fid Frame numerical ID
 */
void ReMedi::setDefaultFrame(int fid)
{
    m_fID = fid;
}

/** \brief Set the parameters of the interactive registerer
 * \param  p : number of points used to compute the orthogonal transformation among views
 * \param  wndHeight : window height (visor)
 * \param  wndWidth : window width (visor)
 * \param  vp : number of vertical ports (visor)
 * \param  hp : number of horizontal ports (visor)
 * \param  camDist : camera dist to (0,0,0) (visor)
 * \param  markerRadius : marker sphere radius (visor)
 */
void ReMedi::setRegistererParameters(int p, int wndHeight, int wndWidth, int vp, int hp, float camDist, float markerRadius)
{
    m_bSetRegistererCorrespondences = (p != -1); // 3 is the minimum num of correspondences to compute a transformation in a 3D space
    m_pRegisterer->setNumPoints(p);
    
    m_pRegisterer->setVisualizerParameters(wndHeight, wndWidth, vp, hp, camDist, markerRadius);
}

/** \brief Set the parameters of the table modeler
 * \param  leafsz : leaf size in the voxel grid downsampling (speeds up the normal computation)
 * \param  normrad : neighborhood radius in normals estimation
 * \param  sacIters : num of iterations in RANSAC used for plane estimation
 * \param  sacDist : distance threshold to consider outliers in RANSAC
 * \param  yoffset : tabletop offset
 * \param  border : distance to the border table when considering blobs
 * \param  condifence : statistical removal of tabletop outlier points (along the y dimension)
 */
void ReMedi::setTableModelerParameters(float leafsz, float normrad, int sacIters, float sacDist, float yoffset, float border, int confidence)
{
    m_pTableModeler->setLeafSize(leafsz);
    m_pTableModeler->setNormalRadius(normrad);
    m_pTableModeler->setSACIters(sacIters);
    m_pTableModeler->setSACDistThresh(sacDist);
    m_pTableModeler->setYOffset(yoffset);
    m_pTableModeler->setInteractionBorder(border);
    m_pTableModeler->setConfidenceLevel(confidence);
}

/** \brief Set the parameters of the background subtractor
 *  \param n The number of samples used for modelling
 *  \param m The modality used to model the background (color, color with shadows, or depth)
 *  \param k The maximum number of components per mixture in each pixel
 *  \param lrate The learning rate for the adaptive background model
 *  \param bgratio The background ratio
 *  \param vargen The variance threshold to consider generation of new mixture components
 */
void ReMedi::setSubtractorParameters(int n, int m, int k, float lrate, float bgratio, float vargen, float opening)
{
    m_pBackgroundSubtractor->setFramesResolution(m_XRes, m_YRes);
    
    // If models have not to be computed, we save the background seq loading time
    m_pBackgroundSubtractor->setBackgroundSequence(m_pBgSeq, n);
    
    m_pBackgroundSubtractor->setModality(m);
    m_pBackgroundSubtractor->setNumOfMixtureComponents(k);
    m_pBackgroundSubtractor->setLearningRate(lrate);
    m_pBackgroundSubtractor->setBackgroundRatio(bgratio);
    m_pBackgroundSubtractor->setVarThresholdGen(vargen);
    m_pBackgroundSubtractor->setOpeningSize(opening);
}

void ReMedi::initialize()
{
    // Register views (and select new correspondences if it's the case)

    m_pRegisterer->setInputFrames( m_pBgSeq->getFrames(m_fID) );
    
    if (!m_bSetRegistererCorrespondences)
        m_pRegisterer->loadCorrespondences("registerer_correspondences", "pcd");
    else
    {
        m_pRegisterer->setCorrespondencesManually();
        m_pRegisterer->saveCorrespondences("registerer_correspondences", "pcd");
    }
    
    m_pRegisterer->computeTransformations();
    
    std::vector<ColorDepthFrame::Ptr> frames;
    m_pRegisterer->registrate(frames);

    // Model the tables
    
    m_pTableModeler->setInputFrames(frames);
    m_pTableModeler->model();
}

void ReMedi::run()
{
    initialize();
}

void remedi::loadSequences(std::string parent, std::vector<std::string>& seqDirnames, std::vector<int>& sbjIds)
{
    seqDirnames.clear();
    sbjIds.clear();
    
    const char* path = parent.c_str();
	if( boost::filesystem::exists( path ) )
	{
		boost::filesystem::directory_iterator iter(path);
        boost::filesystem::directory_iterator end;
        int i = -1;
        std::string sid = "";
		for( ; iter != end ; ++iter )
		{
			if ( boost::filesystem::is_directory( *iter ) )
            {
                std::string dirnameStr = iter->path().stem().string();
                seqDirnames.push_back(dirnameStr);
                
                std::vector<std::string> dirnameStrL;
                boost::split(dirnameStrL, dirnameStr, boost::is_any_of("_"));
                
                if (dirnameStrL[1] != "bs")
                {
                    if (dirnameStrL[1] != sid)
                    {
                        i++;
                        sid = dirnameStrL[1];
                    }
                    sbjIds.push_back(i);
                }
            }
        }
    }
}

void remedi::loadFramesPaths(std::string parent, std::vector<std::string> seqDirnames, std::string subdir, std::string dirname, std::vector<const char*> views, std::vector< std::vector<std::string> >& paths)
{
    paths.clear();
 	paths.resize(seqDirnames.size());
    for (int s = 0; s < seqDirnames.size(); s++)
    {
        std::string pathTmp = parent + seqDirnames[s] + "/" + subdir + dirname;
        
        std::vector<std::string> viewPaths (views.size());
        for (int v = 0; v < views.size(); v++)
            viewPaths[v] = pathTmp + views[v];
        
        paths[s] = viewPaths;
    }
}

void remedi::loadDelaysFile(std::string parent, std::string filename, std::vector<std::vector<int> >& delays)
{
    ifstream inFile;
    std::string path = parent + filename;
    inFile.open(path, ios::in);
    
    assert (inFile.is_open());
    
    std::string line;
    while (getline(inFile, line))
    {
        std::vector<std::string> line_words;
        boost::split(line_words, line, boost::is_any_of(" "));
        
        assert(line_words.size() == 2);
        
        std::vector<std::string> delays_subwords;
        boost::split(delays_subwords, line_words[1], boost::is_any_of(","));
        
        std::vector<int> delaysInLine;
        for (int i = 0; i < delays_subwords.size(); i++)
            delaysInLine.push_back( stoi(delays_subwords[i]) );
        
        delays.push_back(delaysInLine);
    }
    
    inFile.close();
}


void remedi::loadGroundtruth(const std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, Groundtruth& gt)
{
    gt.clear();
    
    for (int s = 0; s < sequences.size(); s++)
    {
        std::string sequencesParent = string(PARENT_PATH) + string(SEQUENCES_SUBDIR)
        + sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        std::string seqName = sequences[s]->getName();

        sequences[s]->restart();
        while (sequences[s]->hasNextFrames())
        {
            sequences[s]->next();
            vector<string> fids = sequences[s]->getFramesFilenames();
            
            for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
            {
                std::string viewName = sequences[s]->getViewName(v);
                std::string frameName = fids[v];

                remedi::io::readAnnotationRegions(sequencesParent + std::string(FOREGROUND_GROUNDTRUTH_DIRNAME) + sequences[s]->getViewName(v), fids[v], gt[seqName][viewName][frameName], true);
            }
        }
    }
}

cv::Rect rectangleIntersection(cv::Rect a, cv::Rect b)
{
    int aib_min_x = max(a.x, b.x);
    int aib_min_y = max(a.y, b.y);
    int aib_max_x = min(a.x + a.width - 1, b.x + b.width - 1);
    int aib_max_y = min(a.y + a.height - 1, b.y + b.height - 1);
    
    int width = aib_max_x - aib_min_x + 1;
    int height = aib_max_y - aib_min_y + 1;
    cv::Rect aib (aib_min_x,
                  aib_min_y,
                  width > 0 ? width : 0,
                  height > 0 ? height : 0);
    return aib;
}

// Auxiliary function for varianceSubsampling(...)
float rectangleOverlap(cv::Rect a, cv::Rect b)
{
    cv::Rect aib = rectangleIntersection(a,b);
    
    float overlap = ((float) aib.area()) / (a.area() + b.area() - aib.area());
    
    return overlap;
}

// Auxiliary function for getTrainingCloudjects(...)
// Check the amount of inclusion of "a" in "b"
float rectangleInclusion(cv::Rect a, cv::Rect b)
{
    cv::Rect aib = rectangleIntersection(a,b);
    
    float inclusion = ((float) aib.area()) / min(a.area(),b.area());
    
    return inclusion;
}

void remedi::getGroundtruthForTraining(const std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, const std::vector<const char*> annotationLabels, const Groundtruth& gt, Groundtruth& gtTr)
{
    gtTr.clear();
    
    for (int s = 0; s < sequences.size(); s++)
    {
        std::string sequencesParent = string(PARENT_PATH) + string(SEQUENCES_SUBDIR)
        + sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        
        sequences[s]->restart();
        while (sequences[s]->hasNextFrames())
        {
            vector<string> prevFids;
            if (sequences[s]->hasPreviousFrames())
                prevFids = sequences[s]->getFramesFilenames();
            
            sequences[s]->next();
            vector<string> fids = sequences[s]->getFramesFilenames();
            
            for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
            {
                std::map<std::string,std::map<std::string,GroundtruthRegion> > annotationsF;
                annotationsF = gt.at(sequences[s]->getName()).at(sequences[s]->getViewName(v)).at(fids[v]);
            
                std::map<std::string,std::map<std::string,GroundtruthRegion> > annotationsTrF;
                
                if (prevFids.empty())
                {
                    // If first frame, all can be training
                    annotationsTrF = annotationsF;
                }
                else
                {
                    // If not the first frame, check objects weren't exactly the
                    // same in the previous. Including the repeated in training
                    // could bias toward the so repeated examples.
                    
                    std::map<std::string,std::map<std::string,GroundtruthRegion> > prevAnnotationsF;
                    prevAnnotationsF = gt.at(sequences[s]->getName()).at(sequences[s]->getViewName(v)).at(prevFids[v]);
                    
                    // We don't want to learn "arms" or "others" instances, but
                    // only those in annotationLabels, i.e. {cup, book, dish, plate, pillbox}
                    
                    for (int i = 0; i < annotationLabels.size(); i++)
                    {
                        std::string lbl = annotationLabels[i];
                        
                        if (annotationsF.count(lbl) && !prevAnnotationsF.count(lbl))
                            annotationsTrF[lbl] = annotationsF[lbl];
                        else if (prevAnnotationsF.count(lbl) && annotationsF.count(lbl))
                        {
                            std::map<std::string,GroundtruthRegion>::iterator it;
                            for (it = annotationsF[lbl].begin(); it != annotationsF[lbl].begin(); ++it)
                            {
                                GroundtruthRegion a = it->second;
                                std::map<std::string,GroundtruthRegion>::iterator jt;
                                for (jt = prevAnnotationsF[lbl].begin(); jt != prevAnnotationsF[lbl].begin(); ++jt)
                                {
                                    GroundtruthRegion preva = jt->second;
                                    float ovl = rectangleOverlap(a.getRect(), preva.getRect());
                                    if (ovl < 0.9)
                                        annotationsTrF[lbl][std::to_string(a.getID())] = a;
                                }
                            }
                        }
                    }
                }
                
                gtTr[sequences[s]->getName()][sequences[s]->getViewName(v)][fids[v]] = annotationsTrF;
            }
        }
    }
}

void remedi::segmentCloudjects(ColorDepthFrame::Ptr frame, cv::Mat foreground, std::vector<Region> regions, std::vector<Cloudject::Ptr>& cloudjects)
{
    cloudjects.resize(regions.size());
    
    for (int i = 0; i < regions.size(); i++)
    {
        // Create a cloudject and fill it with
        Cloudject::Ptr pCj (new Cloudject(regions[i]));
        
        // (1) cropped fg mask
        cv::Rect r = regions[i].getRect();
        
        cv::Mat cropFg (foreground, r); // eventually crop
        cv::Mat cropErrMask (frame->getDepth() == 0, r);
        
        cv::Mat cropFgMask = ((cropFg == ((int) regions[i].getID())) > 0) & ~cropErrMask;
        pCj->setRegionMask(cropFgMask);
        
        // (2) and PFHRGB point cloud within the cropped fg mask
        cv::Mat cropColor (frame->getColor(), r);
        cv::Mat cropDepth (frame->getDepth(), r);
    
        int pos[2] = {r.x, r.y}; // otherwise the conversion assumes 640x480 or (320x240)
        remedi::MatToColoredPointCloud(cropDepth, cropColor, cropFgMask, pos, *(pCj->getCloud()));
    
        // Return it in a list
        cloudjects[i] = pCj;
    }
}

// Weighted overlap between two Region objects. It uses the rectangles
// for the efficient computation of the intersetion. If the intersection
// is not empty, compute in the intersection region the overlap measure.
// Eventually, weight this overlap by the minimum size of the two
// non-cropped overlapping regions.
float weightedOverlapBetweenRegions(Region& region1, Region& region2)
{
    cv::Rect rect1 = region1.getRect();
    cv::Rect rect2 = region2.getRect();
    cv::Rect irect = rectangleIntersection(rect1, rect2);
    
    // RETURN 0 (no intersection)
    if ( !(irect.width > 0 && irect.height > 0) )
        return .0f;

    cv::Mat patch1 = region1.getMask();
    cv::Mat patch2 = region2.getMask();
    
    // Roi the intersection in the two patches
    cv::Mat roi1 ( patch1, cv::Rect(irect.x - rect1.x, irect.y - rect1.y,
                                      irect.width, irect.height) );
    cv::Mat roi2 ( patch2, cv::Rect(irect.x - rect2.x, irect.y - rect2.y,
                                      irect.width, irect.height) );
    assert (roi1.rows == roi2.rows && roi1.cols == roi2.cols);
    
    cv::Mat iroi = roi1 & roi2;
    
    int count1 = cv::countNonZero(patch1); // A
    int count2 = cv::countNonZero(patch2); // B
    int icount = cv::countNonZero(iroi); // A,B , i.e. A and B
    
    // Overlap calculion based on Jaccard index: |A,B|/|AUB|
    float ovl = ((float) icount) / (count1 + count2 - icount); // {AUB} = A + B - {A,B}
    float weight = (((float) icount) / min(count1,count2)); // scale in proportion to min(A,B)
    
    float wovl = ovl * weight;
    
    // RETURN the weighted overlap
    return wovl;
    
}

void remedi::getTrainingCloudjectsWithDescriptor(std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> partitions, int p, std::vector<const char*> annotationLabels, const Groundtruth& gt, std::string descriptorType, std::list<Cloudject::Ptr>& cloudjects)
{
    cloudjects.clear();
    
    // Among all the instances, pick up as training candidates the ones maximizing
    // annotation variance i.e., if an objet is still during 10 frames, consider only
    // the first frame. This minimizes the amount of training and maximizes the inter
    // and intravariation of examples.
    // If required, an stochastic process can decide to pick some of such "stationary"
    // examples (check for bCriterion2 var).
    
    Groundtruth gtTr;
    remedi::getGroundtruthForTraining(sequences, annotationLabels, gt, gtTr);

#ifdef DEBUG_TRAINING_CONSTRUCTION_SELECTION
    std::map<std::string,std::vector<ForegroundRegion> > examples;
#endif //DEBUG_TRAINING_CONSTRUCTION_SELECTION
    
    //
    // Considering the training groundtruth, build the set of cloudjects.
    //
    
    std::cout << "Creating the sample of training cloudjects .." << std::endl;
    boost::timer t;
    
    for (int s = 0; s < sequences.size(); s++)
    {
        if (partitions[s] != p)
        {
            std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
            + sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
            
            std::cout << resultsParent << std::endl;
            
            sequences[s]->restart();
            while(sequences[s]->hasNextFrames())
            {
#if defined (DEBUG_TRAINING_CONSTRUCTION_OVERLAPS) || defined (DEBUG_TRAINING_CONSTRUCTION_SELECTION)
                std::vector<ColorDepthFrame::Ptr> frames = sequences[s]->nextFrames();
#else
                sequences[s]->next();
#endif
                std::vector<std::string> fids = sequences[s]->getFramesFilenames();
                
                for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
                {
#ifdef DEBUG_TRAINING_CONSTRUCTION_OVERLAPS
                    bool bDebug = true;
//                    if (sequences[s]->getName().compare("04_p2_2") == 0 && std::string(sequences[s]->getViewName(v)).compare("1") == 0 && fids[v].compare("00030") == 0)
//                        bDebug = true;
#endif //DEBUG_TRAINING_CONSTRUCTION_OVERLAPS
                    
                    // Iterate over the foreground regions in this frame F
                    
                    std::vector<ForegroundRegion> regionsF;
                    remedi::io::readPredictionRegions(resultsParent + string(CLOUDJECTS_DIRNAME)
                                                      + sequences[s]->getViewName(v), fids[v], regionsF, true);
                    
                    // Obtain the annotations of the instances present in the scene
                    // The process is driven by the annotations in gtTr, and gt
                    // to get the k-1 labels in any of the foreground regions
                    // picked by gtTr.
                    std::map<std::string,std::map<std::string,GroundtruthRegion> > annotationsTrF = gtTr.at(sequences[s]->getName()).at(sequences[s]->getViewName(v)).at(fids[v]);
                    std::map<std::string,std::map<std::string,GroundtruthRegion> > annotationsF = gt.at(sequences[s]->getName()).at(sequences[s]->getViewName(v)).at(fids[v]);
                    
                    // If regionsF's instances are suitable (check criteria below),
                    // include them in further stages of the training (regionsTrF).
                    
                    std::vector<ForegroundRegion> regionsTrF;
                    for (int i = 0; i < regionsF.size(); i++)
                    {
                        regionsF[i].allocateMask();
                        
#ifdef DEBUG_TRAINING_CONSTRUCTION_OVERLAPS
                        cv::Mat drawableImg;

                        if (bDebug)
                        {
                            frames[v]->getColor().copyTo(drawableImg);
                            cv::Rect r1 = regionsF[i].getRect();

                            if (annotationsF.size() > 0)
                            {
                                cv::Point r1min (r1.x,r1.y);
                                cv::Point r1max (r1.x + r1.width - 1, r1.y + r1.height - 1);
                                cv::rectangle(drawableImg, r1min, r1max, cv::Scalar(255,0,0));
                                
                                cv::imshow("region", regionsF[i].getMask());
                                cv::imshow("drawable", drawableImg);
                                cv::waitKey();
                            }
                        }
#endif //DEBUG_TRAINING_CONSTRUCTION_OVERLAPS

                        bool bIsTrainingRegionCandidate = false;

                        std::map<std::string,std::map<std::string,GroundtruthRegion> >::iterator it;
                        for (it = annotationsF.begin(); it != annotationsF.end(); ++it)
                        {
                            std::map<std::string,GroundtruthRegion>::iterator jt;
                            for (jt = it->second.begin(); jt != it->second.end(); ++jt)
                            {
                                GroundtruthRegion a = jt->second;
                                a.allocateMask();
                                float wovl = weightedOverlapBetweenRegions(regionsF[i], a);
                                
#ifdef DEBUG_TRAINING_CONSTRUCTION_OVERLAPS
                                if (bDebug)
                                {
                                    cv::Rect r2 = a.getRect();
                                    cv::Point r2min (r2.x,r2.y);
                                    cv::Point r2max (r2.x + r2.width - 1, r2.y + r2.height - 1);
                                    cv::rectangle(drawableImg, r2min, r2max, cv::Scalar(0,255,0));
                                    
                                    std::cout << wovl << std::endl;
                                    
                                    cv::imshow("annotation", a.getMask());
                                    cv::imshow("drawable", drawableImg);
                                    cv::waitKey();
                                }
#endif //DEBUG_TRAINING_CONSTRUCTION_OVERLAPS

                                a.releaseMask();
                                if (wovl > .01f) regionsF[i].addLabel(jt->second.getCategory());
                                
                                if (annotationsTrF.count(a.getCategory()) > 0 && annotationsTrF.at(a.getCategory()).count(std::to_string(a.getID())) > 0)
                                    bIsTrainingRegionCandidate = true;
                            }
                        }
                        

                        // Check criteria for training inclusion:
                        
//                        // (1a) Exactly one label, that is either "arms" or "others"?
//                        bool bCriterion1a = regionsF[i].getNumOfLabels() == 1 && (regionsF[i].isLabel("arms") || regionsF[i].isLabel("others"));
//                        // (1b) Exactly two labels that are respectively "arms" and "others"?
//                        bool bCriterion1b = regionsF[i].getNumOfLabels() == 2 && regionsF[i].isLabel("arms") && regionsF[i].isLabel("others");
                        // (1a) Exactly one label, that is either "arms" or "others"?
                        bool bCriterionAux = (!regionsF[i].hasLabels() || regionsF[i].isLabel("arms") || regionsF[i].isLabel("others"));
                        // (1) Is a labeled example, but none of the two former cases
                        bool bCriterion1 = bIsTrainingRegionCandidate && !bCriterionAux; //!bCriterion1a && !bCriterion1b;
                        
                        // (2) Is a non-labeled example (pure noise). Thus, a stochastic can decide
                        // to include a proportion of this noise
                        bool bCriterion2 = !regionsF[i].hasLabels() && ((double) rand() / (RAND_MAX)) <= TRAINING_BLOB_SAMPLING_RATE;
                        
                        if ( bCriterion1 || bCriterion2 )
                        {
                            regionsTrF.push_back(regionsF[i]);
                            
#ifdef DEBUG_TRAINING_CONSTRUCTION_SELECTION
                            ForegroundRegion r;
                            cv::Mat roi (frames[v]->getColor(), regionsF[i].getRect());
                            cv::Mat tmp;
                            roi.copyTo(tmp, regionsF[i].getMask());
                            r.setMask(tmp);
                            
                            r.setFilepath(sequences[s]->getName() + "-" + sequences[s]->getViewName(v) + "-" + fids[v]);
                            
                            std::set<std::string> labels = regionsF[i].getLabels();
                            r.setLabels(labels);
                            
                            std::set<std::string>::iterator jt;
                            for (jt = labels.begin(); jt != labels.end(); ++jt)
                                examples[*jt].push_back(r);
                            
                            if (labels.empty())
                                examples["nolabel"].push_back(r);
#endif //DEBUG_TRAINING_CONSTRUCTION_SELECTION
                        }
#ifndef DEBUG_TRAINING_CONSTRUCTION_SELECTION
                        regionsF[i].releaseMask();
#else
                        cv::Mat roi (frames[v]->getColor(), regionsF[i].getRect());
                        cv::Mat maskedroi;
                        roi.copyTo(maskedroi, regionsF[i].getMask());
                        regionsF[i].setMask(maskedroi);
#endif //DEBUG_TRAINING_CONSTRUCTION_SELECTION
                    }
                    
                    // Read the cloudjects representing the selected regions
                    std::vector<Cloudject::Ptr> cloudjectsTrF;
                    remedi::io::mockreadCloudjectsWithDescriptor(resultsParent + string(CLOUDJECTS_DIRNAME) + sequences[s]->getViewName(v), fids[v], regionsTrF, descriptorType, cloudjectsTrF, true);
                    
                    // Assign the already prepared labels and push
                    for (int i = 0; i < cloudjectsTrF.size(); i++)
                        cloudjects.push_back(cloudjectsTrF[i]);
                }
            }
        }
    }
    
    std::cout << "Created a training sample of " << cloudjects.size() << " cloudjects (in " << t.elapsed() << " secs.)" << std::endl;
    
#ifdef DEBUG_TRAINING_CONSTRUCTION_SELECTION
    std::list<Cloudject::Ptr>::iterator ct;
    for (ct = cloudjects.begin(); ct != cloudjects.end(); ++ct)
    {
        cv::imshow("cloudjects", (*ct)->getRegionMask());
        
        std::set<std::string> labels = (*ct)->getRegionLabels();
        std::set<std::string>::iterator jt;
        for (jt = labels.begin(); jt != labels.end(); ++jt)
            std::cout << *jt << " ";
        std::cout << std::endl;
        
        cv::waitKey();
    }
    
    std::map<std::string,std::vector<ForegroundRegion> >::iterator it;
    for (it = examples.begin(); it != examples.end(); ++it)
    {
        std::string className = it->first;
        
        std::vector<ForegroundRegion> classExamples = it->second;
        
        for (int i = 0; i < classExamples.size(); i += 10)
        {
            cv::imshow(className, classExamples[i].getMask());
            
            std::cout << std::to_string(i) + "/" + std::to_string(classExamples.size()) << ", ";
            std::cout << classExamples[i].getFilepath() << ", ";
            
            std::set<std::string> labels = classExamples[i].getLabels();
            std::set<std::string>::iterator jt;
            for (jt = labels.begin(); jt != labels.end(); ++jt)
                std::cout << *jt << " ";
            std::cout << std::endl;
            
            cv::waitKey();
        }
    }
#endif //DEBUG_TRAINING_CONSTRUCTION_SELECTION
}

void remedi::getTestCloudjectsWithDescriptor(std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> partitions, int p, const Groundtruth& gt, std::string descriptorType, std::list<Cloudject::Ptr>& cloudjects)
{
    cloudjects.clear();
    
    std::cout << "Creating the sample of testing cloudjects .." << std::endl;
    boost::timer t;
    
    for (int s = 0; s < sequences.size(); s++)
    {
        if (partitions[s] == p)
        {
            std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
            + sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
            
            std::cout << resultsParent << std::endl;
            
            sequences[s]->restart();
            while(sequences[s]->hasNextFrames())
            {
                sequences[s]->next();
                std::vector<std::string> fids = sequences[s]->getFramesFilenames();
                
                for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
                {
                    // Iterate over the foreground regions in this frame F
                    std::vector<ForegroundRegion> regionsF;
                    remedi::io::readPredictionRegions(resultsParent + string(CLOUDJECTS_DIRNAME)
                                                      + sequences[s]->getViewName(v), fids[v], regionsF, true);
                    
                    std::map<std::string,std::map<std::string,GroundtruthRegion> > annotationsF = gt.at(sequences[s]->getName()).at(sequences[s]->getViewName(v)).at(fids[v]);
                    
                    std::vector<ForegroundRegion> regionsTeF;
                    for (int i = 0; i < regionsF.size(); i++)
                    {
                        regionsF[i].allocateMask();
                        
                        std::map<std::string,std::map<std::string,GroundtruthRegion> >::iterator it;
                        for (it = annotationsF.begin(); it != annotationsF.end(); ++it)
                        {
                            std::map<std::string,GroundtruthRegion>::iterator jt;
                            for (jt = it->second.begin(); jt != it->second.end(); ++jt)
                            {
                                GroundtruthRegion a = jt->second;
                                a.allocateMask();
                                float wovl = weightedOverlapBetweenRegions(regionsF[i], a);
                                a.releaseMask();
                                
                                if (wovl > .01f) regionsF[i].addLabel(jt->second.getCategory());
                            }
                        }
                        
                        regionsTeF.push_back(regionsF[i]);
                    }
                    
                    // Read the cloudjects representing the selected regions
                    std::vector<Cloudject::Ptr> cloudjectsTeF;
                    remedi::io::mockreadCloudjectsWithDescriptor(resultsParent + string(CLOUDJECTS_DIRNAME) + sequences[s]->getViewName(v), fids[v], regionsTeF, descriptorType, cloudjectsTeF, true);
                    
                    // Assign the already prepared labels and push
                    for (int i = 0; i < cloudjectsTeF.size(); i++)
                        cloudjects.push_back(cloudjectsTeF[i]);
                }
            }
        }
    }
    
    std::cout << "Created a test sample of " << cloudjects.size() << " cloudjects (in " << t.elapsed() << " secs.)" << std::endl;
}

void remedi::getTestSequences(std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> partitions, int p, std::vector<Sequence<ColorDepthFrame>::Ptr>& sequencesTe)
{
    for (int s = 0; s < sequences.size(); s++)
    {
        if (partitions[s] == p)
            sequencesTe.push_back(sequences[s]);
    }
}

// TODO: adapt Groundtruth's new structure
// Evaluate detection performance in a frame
void ReMedi::evaluateFrame(const std::map<std::string,std::map<std::string,GroundtruthRegion> > gt, const std::vector<ForegroundRegion> dt, int& tp, int& fp, int& fn)
{
    // Iterate over the dt vector to determine TP and FP
    tp = fp = fn = 0;
    
    for (int i = 0; i < dt.size(); i++)
    {
        const std::set<std::string> labels = dt[i].getLabels();
        
        std::set<std::string>::iterator it;
        for (it = labels.begin(); it != labels.end(); ++it)
        {
            if (gt.count(*it) == 0)
            {
                fp++;
            }
            else
            {
                std::map<std::string,GroundtruthRegion> catgt = gt.at(*it);
                
                std::map<std::string,GroundtruthRegion>::iterator jt;
                for (jt = catgt.begin(); jt != catgt.end(); ++jt)
                    (rectangleOverlap(dt[i].getRect(),jt->second.getRect()) > 0) ? tp++ : fp++;
            }
        }
    }
    
    // Count the total no. annotations, useful to compute FN
    int n = 0;
    std::map<std::string,std::map<std::string,GroundtruthRegion> >::const_iterator it;
    for (it = gt.begin(); it != gt.end(); ++it)
        n += it->second.size();
    
    // Compute FN as: n - TP
    fn = n - tp;
}

void ReMedi::evaluate(const std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<const char*> objectsLabels, const Groundtruth& gt, const Detection& dt, int& tp, int& fp, int& fn)
{
    tp = fp = fn = 0;
    for (int s = 0; s < sequences.size(); s++)
    {
        sequences[s]->restart();
        while (sequences[s]->hasNextFrames())
        {
            sequences[s]->next();
            std::vector<std::string> fids = sequences[s]->getFramesFilenames();
            for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
            {
                int tpF, fpF, fnF;
                evaluateFrame(gt.at(sequences[s]->getName()).at(sequences[s]->getViewName(v)).at(fids[v]),
                              dt.at(sequences[s]->getName()).at(sequences[s]->getViewName(v)).at(fids[v]),
                              tpF, fpF, fnF);
                tp += tpF;
                fp += fpF;
                fn += fnF;
            }
        }
    }
}

void ReMedi::detect(const std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<const char*> objectsLabels, const Groundtruth& gt, Detection& dt, const CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::Ptr pipeline)
{
    for (int s = 0; s < sequences.size(); s++)
    {
        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
        + sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        
#ifdef DO_VISUALIZE_DETECTIONS
        pcl::visualization::PCLVisualizer::Ptr pVis (new pcl::visualization::PCLVisualizer);
        
        int V = sequences[s]->getNumOfViews();
        std::vector<int> vp (V);
        for (int v = 0; v < V; v++)
        {
            pVis->createViewPort(v*(1.f/V), 0, (v+1)*(1.f/V), 1, vp[v]);
            pVis->setBackgroundColor (1, 1, 1, vp[v]);
            m_pRegisterer->setDefaultCamera(pVis, vp[v]);
        }
#endif
        
        sequences[s]->restart();
        while (sequences[s]->hasNextFrames())
        {
            vector<ColorDepthFrame::Ptr> frames = sequences[s]->nextFrames();
            m_pRegisterer->setInputFrames(frames);
            m_pRegisterer->registrate(frames);

            std::vector<cv::Mat> tabletops;
            std::vector<cv::Mat> interactions;
            m_pTableModeler->getTabletopMask(frames, tabletops);
            m_pTableModeler->getInteractionMask(frames, interactions);
            
            vector<string> fids = sequences[s]->getFramesFilenames();
            
            std::cout << fids[0] << std::endl;

            // Prepare the interactors from the different views
            
            std::vector<ColorPointCloudPtr> interactors (sequences[s]->getNumOfViews());
            for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
            {
                interactors[v] = ColorPointCloudPtr(new ColorPointCloud);
                cv::Mat aux;
                cvx::open(interactions[v] & ~tabletops[v], 2, aux);
                frames[v]->getRegisteredColoredPointCloud(aux, *(interactors[v]));
            }
            
            // Distinguish actors from interactors
            std::vector<std::vector<Cloudject::Ptr> > nonInteractedActors (sequences[s]->getNumOfViews());
            std::vector<std::vector<Cloudject::Ptr> > interactedActors (sequences[s]->getNumOfViews());
            
            for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
            {
                std::vector<ForegroundRegion> regionsF;
                remedi::io::readPredictionRegions(resultsParent + string(CLOUDJECTS_DIRNAME)
                                                  + sequences[s]->getViewName(v), fids[v], regionsF, true);
                
                // Read the cloudjects representing the selected regions
                std::vector<Cloudject::Ptr> cloudjectsF;
                remedi::io::readCloudjectsWithDescriptor(resultsParent + string(CLOUDJECTS_DIRNAME) + sequences[s]->getViewName(v), fids[v], regionsF, "pfhrgb250", cloudjectsF, true);
                
                for (int i = 0; i < cloudjectsF.size(); i++)
                    cloudjectsF[i]->setRegistrationTransformation(frames[v]->getRegistrationTransformation());
                
                findActors(cloudjectsF, interactors, nonInteractedActors[v], 0.02, interactedActors[v]);
            }

            // Find the correspondences among actor clouds
            std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences;
            std::vector<std::vector<PointT> > positions;
            remedi::findCorrespondences(frames, nonInteractedActors, OR_CORRESPONDENCE_TOLERANCE, correspondences, positions); // false (not to registrate)
            
            for (int i = 0; i < correspondences.size(); i++)
            {
                std::vector<std::vector<float> > distsToMargin (correspondences[i].size());
                for (int v = 0; v < correspondences[i].size(); v++)
                    pipeline->predict(correspondences[i][v].second, distsToMargin[v]);
                
                std::vector<int> predictions (objectsLabels.size());
                for (int j = 0; j < objectsLabels.size(); j++)
                {
                    float distToMarginFused = .0f;
                    for (int v = 0; v < correspondences[i].size(); v++)
                    {
                        pcl::PointXYZ c = correspondences[i][v].second->getCloudCentroid();
                        float wsq = 1.f / (pow(c.x,2) + pow(c.y,2) + pow(c.z,2));
                        distToMarginFused += ((wsq * distsToMargin[v][j]) / correspondences[i].size());
                    }
                    predictions[j] = (distToMarginFused < 0) ? 1 : -1;
                }
                
                for (int v = 0; v < correspondences[i].size(); v++)
                {
                    ForegroundRegion r = correspondences[i][v].second->getRegion();
                    for (int j = 0; j < objectsLabels.size(); j++)
                        if (predictions[j] > 0) r.addLabel(objectsLabels[j]);
                    
                    dt[sequences[s]->getName()][sequences[s]->getViewName(v)][fids[v]].push_back(r);
                }
            }
            
            for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
            {
                int tp, fp, fn;
                evaluateFrame(gt.at(sequences[s]->getName()).at(sequences[s]->getViewName(v)).at(fids[v]),
                              dt.at(sequences[s]->getName()).at(sequences[s]->getViewName(v)).at(fids[v]),
                              tp, fp, fn);
                std::cout << std::to_string(tp) << "," << std::to_string(fp) << "," << std::to_string(fn) << std::endl;
            }
            
#ifdef DO_VISUALIZE_DETECTIONS
            for (int v = 0; v < V; v++)
            {
                pVis->removeAllPointClouds(vp[v]);
                pVis->removeAllShapes(vp[v]);
                
                for (int i = 0; i < interactors.size(); i++)
                    pVis->addPointCloud(interactors[i], "interactor" + std::to_string(v) + "-" + std::to_string(i), vp[v] );
                
                for (int i = 0; i < interactedActors[v].size(); i++)
                {
                    pVis->addPointCloud(interactedActors[v][i]->getRegisteredCloud(), "nactor" + std::to_string(v) + "-" + std::to_string(i), vp[v] );
                    pVis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 0, "nactor" + std::to_string(v) + "-" + std::to_string(i), vp[v]);
                }
                
                for (int i = 0; i < positions[v].size(); i++)
                {
                    pVis->addSphere(positions[v][i], 0.025, ((v == 0) ? 1 : 0), ((v == 0) ? 0 : 1), 0, "sphere" + std::to_string(v) + "-" + std::to_string(i), vp[0] );
                }
            }
            
            for (int i = 0; i < correspondences.size(); i++) for (int j = 0; j < correspondences[i].size(); j++)
            {
                int v = correspondences[i][j].first;
                pVis->addPointCloud(correspondences[i][j].second->getRegisteredCloud(), "actor" + std::to_string(v) + "-" + std::to_string(i), vp[0] );
                pVis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, g_Colors[i%14][0], g_Colors[i%14][1], g_Colors[i%14][2], "actor" + std::to_string(v) + "-" + std::to_string(i), vp[0]);
            }
            
            pVis->spinOnce();
#endif
        }
    }
}

void downsample(ColorPointCloudPtr pCloud, float leafSize, ColorPointCloud& cloudFiltered)
{
    if (leafSize == 0.f)
    {
        cloudFiltered = *pCloud; // copy
    }
    else
    {
        pcl::VoxelGrid<ColorPointT> vg;
        vg.setInputCloud(pCloud);
        vg.setLeafSize(leafSize, leafSize, leafSize);
        vg.filter(cloudFiltered);
    }
}

void ReMedi::findActors(std::vector<Cloudject::Ptr> candidates, std::vector<ColorPointCloudPtr> interactors, std::vector<Cloudject::Ptr>& actors, float leafSize, std::vector<Cloudject::Ptr>& interactedActors)
{
    actors.clear();
    interactedActors.clear();
    
    std::vector<ColorPointCloudPtr> _candidates (candidates.size());
    for (int i = 0; i < candidates.size(); i++)
        _candidates[i] = candidates[i]->getRegisteredCloud();
    
    std::vector<bool> interactionsMask;
    findInteractions(_candidates, interactors, interactionsMask, leafSize);
    
    assert (interactionsMask.size() == _candidates.size());
    
    actors.reserve(candidates.size());
    interactedActors.reserve(candidates.size());
    
    for (int i = 0; i < candidates.size(); i++)
    {
        if (!interactionsMask[i])
            actors.push_back(candidates[i]);
        else
            interactedActors.push_back(candidates[i]);
    }
}

void ReMedi::findInteractions(std::vector<ColorPointCloudPtr> candidates, std::vector<ColorPointCloudPtr> interactors, std::vector<bool>& mask, float leafSize)
{
    mask.clear();
    mask.resize(candidates.size(), false);

    std::vector<ColorPointCloudPtr> interactorsFilt;
    bool bMainInteractorsPresent = false;
    
    for (int i = 0; i < interactors.size(); i++)
    {
        ColorPointCloudPtr pInteractorFilt (new ColorPointCloud);
        downsample(interactors[i], leafSize, *pInteractorFilt);
        
        ColorPointCloudPtr pMainInteractorFilt (new ColorPointCloud);
        biggestEuclideanCluster(pInteractorFilt, 250, 25000, leafSize, *pMainInteractorFilt);
        
        if (pMainInteractorFilt->size() > 0)
            interactorsFilt.push_back(pMainInteractorFilt);
    }

    if (!interactorsFilt.empty())
    {
        std::vector<ColorPointCloudPtr> candidatesCloudsFilt (candidates.size());
        for (int i = 0; i < candidates.size(); i++)
        {
            ColorPointCloudPtr pCandidateCloudFilt (new ColorPointCloud);
            downsample(candidates[i], leafSize, *pCandidateCloudFilt);
            
            candidatesCloudsFilt[i] = pCandidateCloudFilt;
        }
        
        // Check for each cluster that lies on the table
        for (int j = 0; j < interactorsFilt.size(); j++) for (int i = 0; i < candidatesCloudsFilt.size(); i++)
            if (isInteractive(candidatesCloudsFilt[i], interactorsFilt[j], 4.*leafSize))
                mask[i] = true;
    }
}

bool ReMedi::isInteractive(ColorPointCloudPtr tabletopRegionCluster, ColorPointCloudPtr interactionCloud, float tol)
{
    bool interactive = false;
    
    // if some of its points
    for (int k_i = 0; k_i < tabletopRegionCluster->points.size() && !interactive; k_i++)
    {
        if (tabletopRegionCluster->points[k_i].z > 0)
        {
            // some point of the interaction region.
            for (int k_j = 0; k_j < interactionCloud->points.size() && !interactive; k_j++)
            {
                if (interactionCloud->points[k_j].z > 0)
                {
                    ColorPointT p = tabletopRegionCluster->points[k_i];
                    ColorPointT q = interactionCloud->points[k_j];
                    
                    // If they are practically contiguous, they must had been part of the same cloud
                    float pqDist = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2) + pow(p.z - q.z, 2));
                    interactive = (pqDist <= tol); // if tabletop cluster is stick to any cluster
                    // in the interaction region, we say it is interactive
                    // and become an interactor
                }
            }
        }
    }
    
    return interactive;
}

void remedi::MatToColoredPointCloud(cv::Mat depth, cv::Mat color, cv::Mat mask, int pos[2], pcl::PointCloud<pcl::PointXYZRGB>& cloud)
{
    cloud.height = depth.rows;
    cloud.width =  depth.cols;
    cloud.resize(depth.rows * depth.cols);
    cloud.is_dense = true;
    
    float invfocal = (1/285.63f) / (X_RESOLUTION/320.f); // Kinect inverse focal length. If depth map resolution
    
    float z;
    float rwx, rwy, rwz;
    pcl::PointXYZRGB p;
    
    for (unsigned int y = 0; y < cloud.height; y++) for (unsigned int x = 0; x < cloud.width; x++)
    {
        if (mask.at<unsigned char>(y,x) > 0)
        {
            z = (float) depth.at<unsigned short>(y,x) /*z_us*/;
            
            rwx = ((x+pos[0]) - 320.0) * invfocal * z;
            rwy = ((y+pos[1]) - 240.0) * invfocal * z;
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

void remedi::findCorrespondences(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, float tol, std::vector<std::vector<std::pair<int,Cloudject::Ptr> > >& correspondences, std::vector<std::vector<PointT> >& positions)
{
    correspondences.clear();
    
    // Correspondences found using the positions of the clouds' centroids
//    std::vector<std::vector<PointT> > positions;
    _getDetectionPositions(frames, detections, true, positions); // registrate is "true"
    
    std::vector<std::vector<std::pair<std::pair<int,int>,PointT> > > _correspondences;
    _findCorrespondences(positions, tol, _correspondences);
    
    // Bc we want to have chains of clouds (not points) and the views to which they correspond, transform _correspondences -> correspondences
    correspondences.clear();
    for (int i = 0; i < _correspondences.size(); i++)
    {
        std::vector<std::pair<int,Cloudject::Ptr> > chain; // clouds chain
        for (int j = 0; j < _correspondences[i].size(); j++)
        {
            int v   = _correspondences[i][j].first.first;
            int idx = _correspondences[i][j].first.second;
            
            chain += std::pair<int,Cloudject::Ptr>(v, detections[v][idx]);
        }
        correspondences += chain;
    }
}

void remedi::_findCorrespondences(std::vector<std::vector<PointT> > positions, float tol, std::vector<std::vector<std::pair<std::pair<int,int>,PointT> > >& correspondences)
{
    correspondences.clear();
    
    // Keep already made assignations, to cut search paths
    std::vector<std::vector<bool> > assignations (positions.size());
    for (int v = 0; v < positions.size(); v++)
        assignations[v].resize(positions[v].size(), false);
    
    // Cumbersome internal (from this function) structure:
    // Vector of chains
    // A chain is a list of points
    // These points have somehow attached the 'view' and the 'index in the view'.
    for (int v = 0; v < positions.size(); v++)
    {
        for (int i = 0; i < positions[v].size(); i++)
        {
            if (!assignations[v][i])
            {
                std::vector<std::pair<std::pair<int,int>,PointT> > chain; // points chain
                chain += std::pair<std::pair<int,int>,PointT>(std::pair<int,int>(v,i), positions[v][i]);
                assignations[v][i] = true;
                
                if (tol > 0)
                    findNextCorrespondence(positions, assignations, v+1, tol, chain); // recursion
                
                correspondences += chain;
            }
        }
    }
}


void remedi::findNextCorrespondence(std::vector<std::vector<PointT> >& positions, std::vector<std::vector<bool> >& assignations, int v, float tol, std::vector<std::pair<std::pair<int,int>,PointT> >& chain)
{
    if (v == positions.size())
    {
        return;
    }
    else
    {
        int minIdx = -1;
        float minDist = std::numeric_limits<float>::max();
        
        for (int i = 0; i < chain.size(); i++)
        {
            for (int j = 0; j < positions[v].size(); j++)
            {
                if ( !assignations[v][j] )
                {
                    PointT p = chain[i].second;
                    PointT q = positions[v][j];
                    float d = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2) + pow(p.z - q.z, 2));
                    
                    if (d < minDist && d <= tol)
                    {
                        minIdx = j;
                        minDist = d;
                    }
                }
            }
        }
        
        if (minIdx >= 0)
        {
            chain += std::pair<std::pair<int,int>,PointT>(std::pair<int,int>(v,minIdx), positions[v][minIdx]);
            assignations[v][minIdx] = true;
        }
        
        findNextCorrespondence(positions, assignations, v+1, tol, chain);
    }
}

void remedi::_getDetectionPositions(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, bool bRegistrate, std::vector<std::vector<PointT> >& positions)
{
    positions.clear();
    
    positions.resize(detections.size());
    for (int v = 0; v < detections.size(); v++)
    {
        std::vector<PointT> viewPositions (detections[v].size());
        
        for (int i = 0; i < detections[v].size(); i++)
        {
            if (!bRegistrate)
                viewPositions[i] = detections[v][i]->getCloudCentroid();
            else
                viewPositions[i] = detections[v][i]->getRegisteredCloudCentroid();
        }
        
        positions[v] = viewPositions;
    }
}