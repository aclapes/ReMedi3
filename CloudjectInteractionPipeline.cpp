//
//  CloudjectInteractionPipeline.cpp
//  remedi3
//
//  Created by Albert Clapés on 06/04/15.
//
//

#include "CloudjectInteractionPipeline.h"
#include "constants.h"
#include "statistics.h"
#include <pcl/point_types.h>
#include "MultiviewVisualizer.h"

CloudjectInteractionPipeline::CloudjectInteractionPipeline() :
    m_MultiviewStrategy(DETECT_MONOCULAR),
    m_InteractionThresh(3),
    m_ValPerf(0)
{
    
}

CloudjectInteractionPipeline::CloudjectInteractionPipeline(const CloudjectInteractionPipeline& rhs)
{
    *this = rhs;
}

CloudjectInteractionPipeline& CloudjectInteractionPipeline::operator=(const CloudjectInteractionPipeline& rhs)
{
    if (this != &rhs)
    {
        m_Sequences = rhs.m_Sequences;
        m_Categories = rhs.m_Categories;
        
        m_pRegisterer = rhs.m_pRegisterer;
        m_pTableModeler = rhs.m_pTableModeler;
        m_ClassificationPipeline = rhs.m_ClassificationPipeline;
        
        m_Gt = rhs.m_Gt;
        m_IAct = rhs.m_IAct; // [I]nter[Act]tions groundtruth
        m_CorrespCriterion = rhs.m_CorrespCriterion;
        m_DetectionCorrespThresh = rhs.m_DetectionCorrespThresh;
        m_InteractionThresh = rhs.m_InteractionThresh;
        
        m_MultiviewStrategy = rhs.m_MultiviewStrategy;
        m_MultiviewLateFusionStrategy = rhs.m_MultiviewLateFusionStrategy;
        m_MultiviewCorrespThresh = rhs.m_MultiviewCorrespThresh;
        
        m_LeafSize = rhs.m_LeafSize;
        
        m_LateFusionScalings = rhs.m_LateFusionScalings;
        
        m_ValParams = rhs.m_ValParams;
        m_ValPerf = rhs.m_ValPerf;
        
        m_DetectionResults = rhs.m_DetectionResults;
        m_InteractionResults = rhs.m_InteractionResults;
        m_InteractionPredictions = rhs.m_InteractionPredictions;
    }
    return *this;
}

void CloudjectInteractionPipeline::setInputSequences(const std::vector<Sequence<ColorDepthFrame>::Ptr>& sequences)
{
    m_Sequences = sequences;
}

void CloudjectInteractionPipeline::setCategories(const std::vector<const char*>& categories)
{
    m_Categories = categories;
}

void CloudjectInteractionPipeline::setInteractiveRegisterer(InteractiveRegisterer::Ptr ir)
{
    m_pRegisterer = ir;
}

void CloudjectInteractionPipeline::setTableModeler(TableModeler::Ptr pTM)
{
    m_pTableModeler = pTM;
}

void CloudjectInteractionPipeline::setBackgroundSubtractor(BackgroundSubtractor<cv::BackgroundSubtractorMOG2,ColorDepthFrame>::Ptr pBS)
{
    m_pSubtractor = pBS;
}

void CloudjectInteractionPipeline::setGroundtruth(const Groundtruth& gt)
{
    m_Gt = gt;
}

void CloudjectInteractionPipeline::setInteractionGroundtruth(const Interaction& iact)
{
    m_IAct = iact;
}

void CloudjectInteractionPipeline::setCorrespondenceCriterion(int criterion)
{
    m_CorrespCriterion = criterion;
}

void CloudjectInteractionPipeline::setDetectionCorrespondenceThresh(float thresh)
{
    m_DetectionCorrespThresh = thresh;
}

void CloudjectInteractionPipeline::setInteractionThresh(float thresh)
{
    m_InteractionThresh = thresh;
}

void CloudjectInteractionPipeline::setMultiviewStrategy(int strategy)
{
    m_MultiviewStrategy = strategy;
}

void CloudjectInteractionPipeline::setMultiviewLateFusionStrategy(int strategy)
{
    m_MultiviewLateFusionStrategy = strategy;
}

void CloudjectInteractionPipeline::setMultiviewCorrespondenceThresh(float thresh)
{
    m_MultiviewCorrespThresh = thresh;
}

void CloudjectInteractionPipeline::setLeafSize(Eigen::Vector3f leafSize)
{
    m_LeafSize = leafSize;
}

void CloudjectInteractionPipeline::setMultiviewLateFusionNormalization(std::vector<std::vector<float> > scalings)
{
    m_LateFusionScalings = scalings;
}

void CloudjectInteractionPipeline::setClassificationPipeline(CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr pipeline)
{
    m_ClassificationPipeline = pipeline;
}

void CloudjectInteractionPipeline::setValidationParameters(std::vector<float> correspCriteria,
                                                         std::vector<float> detectionThreshs,
                                                         std::vector<float> interactionThreshs,
                                                         std::vector<float> multiviewLateFusionStrategies,
                                                         std::vector<float> multiviewCorrespThreshs)
{
    m_ValParams.clear();
    
    if (!correspCriteria.empty())
        m_ValParams.push_back(correspCriteria);
    if (!detectionThreshs.empty())
        m_ValParams.push_back(detectionThreshs);
    if (!interactionThreshs.empty())
        m_ValParams.push_back(interactionThreshs);
    
    if (!multiviewLateFusionStrategies.empty())
        m_ValParams.push_back(multiviewLateFusionStrategies);
    if (!multiviewCorrespThreshs.empty())
        m_ValParams.push_back(multiviewCorrespThreshs);
}

void CloudjectInteractionPipeline::setValidationInteractionOverlapCriterion(int crit)
{
    m_ValInteractionOvlCriterion = crit;
}

void CloudjectInteractionPipeline::validateDetection()
{
    std::vector<std::vector<float> > combinations;
    expandParameters<float>(m_ValParams, combinations);
    
    float valPerf = 0.f;
    int valIdx = 0;
    for (int i = 0; i < combinations.size(); i++)
    {
        std::cout << cv::Mat(combinations[i]) << std::endl;
        setValidationParametersCombination(combinations[i]);
        
        if (m_MultiviewStrategy == DETECT_MONOCULAR)
            predictDetectionMonocular();
        else
            predictDetectionMultiview();
        
        float fscore = .0f;
        for (int v = 0; v < m_DetectionResults.size(); v++)
        {
            Result result;
            for (int j = 0; j < m_DetectionResults[v].size(); j++)
                result += m_DetectionResults[v][j];
            
            float fscoreView = (2.f*result.tp) / (2.f*result.tp + result.fp + result.fn);
            std::cout << result.toVector() << ", (" << fscoreView << ")" << ((v < m_DetectionResults.size() - 1) ? "," : ";\n");
            
            fscore += (fscoreView / m_DetectionResults.size());
        }
        
        if (fscore > valPerf)
        {
            valPerf = fscore;
            valIdx = i;
        }
    }
    
    // Set the parameters finally
    setValidationParametersCombination(combinations[valIdx]);
    m_ValPerf = valPerf;
}

void CloudjectInteractionPipeline::validateInteraction()
{
    m_InteractionPredictions.clear(); // not used in the validation
    m_InteractionResults.clear();
    
    expandParameters<float>(m_ValParams, m_ValCombs);
    
    if (m_MultiviewStrategy == DETECT_MONOCULAR)
    {
        m_InteractionResults.resize(m_pSubtractor->getNumOfViews());
        for (int v = 0; v < m_pSubtractor->getNumOfViews(); v++)
            m_InteractionResults[v] = cv::Mat(m_ValCombs.size(), m_Categories.size(),
                                              CV_32FC4, cv::Scalar(0,0,0,0));
    }
    else if (m_MultiviewStrategy == DETECT_MULTIVIEW)
    {
        m_InteractionResults.push_back(cv::Mat(m_ValCombs.size(), m_Categories.size(),
                                               CV_32FC4, cv::Scalar(0,0,0,0)));
    }
    
    m_ValPerf = 0.f;
    m_ValIdx = 0;
    for (int i = 0; i < m_ValCombs.size(); i++)
    {
        std::cout << cv::Mat(m_ValCombs[i]) << std::endl;
        setValidationParametersCombination(m_ValCombs[i]);
        
        if (m_MultiviewStrategy == DETECT_MONOCULAR)
            predictInteractionMultiview();
        else if (m_MultiviewStrategy == DETECT_MULTIVIEW)
            predictInteractionMultiview();
        
        float ovl = .0f;
        for (int v = 0; v < m_InteractionCombResults.size(); v++)
        {
            Result result;
            for (int j = 0; j < m_InteractionCombResults[v].size(); j++)
            {
                result += m_InteractionCombResults[v][j];
                m_InteractionResults[v].at<cv::Vec4f>(i,j) = m_InteractionCombResults[v][j].toVector();
            }
            
            float ovlView = ((float)result.tp) / (result.tp + result.fp + result.fn);
            std::cout << result.toVector() << ", (" << ovlView << ")" << ((v < m_InteractionCombResults.size() - 1) ? "," : ";\n");
            
            ovl += (ovlView / m_InteractionCombResults.size());
        }
        
        if (ovl > m_ValPerf)
        {
            m_ValPerf = ovl;
            m_ValIdx = i;
        }
    }
    
    // Set the parameters finally
    setValidationParametersCombination(m_ValCombs[m_ValIdx]);
}

float CloudjectInteractionPipeline::getValidationPerformance()
{
    return m_ValPerf;
}

std::vector<cv::Mat> CloudjectInteractionPipeline::getInteractionResults()
{
    return m_InteractionResults;
}

void CloudjectInteractionPipeline::predictInteraction()
{
    m_InteractionPredictions.clear(); // not used in the validation
    m_InteractionResults.clear();
    
    expandParameters<float>(m_ValParams, m_ValCombs);
    
    if (m_MultiviewStrategy == DETECT_MONOCULAR)
    {
        m_InteractionPredictions.resize(m_pSubtractor->getNumOfViews());
        
        m_InteractionResults.resize(m_pSubtractor->getNumOfViews());
        for (int v = 0; v < m_pSubtractor->getNumOfViews(); v++)
            m_InteractionResults[v] = cv::Mat(1, m_Categories.size(), CV_32FC4, cv::Scalar(0,0,0,0));
        
        predictInteractionMultiview();
    }
    else if (m_MultiviewStrategy == DETECT_MULTIVIEW)
    {
        m_InteractionPredictions.resize(1);
        
        m_InteractionResults.push_back(cv::Mat(1, m_Categories.size(), CV_32FC4, cv::Scalar(0,0,0,0)));
        
        predictInteractionMultiview();
    }
    
    m_Perf = .0f;
    for (int v = 0; v < m_InteractionCombResults.size(); v++)
    {
        Result result;
        for (int j = 0; j < m_InteractionCombResults[v].size(); j++)
        {
            m_InteractionResults[v].at<cv::Vec4f>(0,j) = m_InteractionCombResults[v][j].toVector();
            result += m_InteractionCombResults[v][j];
        }
        
        float ovlView = ((float)result.tp) / (result.tp + result.fp + result.fn);
        std::cout << result.toVector() << ", (" << ovlView << ")" << ((v < m_InteractionCombResults.size() - 1) ? "," : ";\n");
        
        m_Perf += (ovlView / m_InteractionCombResults.size());
    }
}

void CloudjectInteractionPipeline::save(std::string filename, std::string extension)
{
    std::string filenameWithSuffix = filename + extension; // extension is expected to already include the ext dot (.xxx)
    cv::FileStorage fs;
    
    if (extension.compare(".xml") == 0)
        fs.open(filenameWithSuffix, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_XML);
    else // yml
        fs.open(filenameWithSuffix, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    
    fs << "correspCriterion" << m_CorrespCriterion;
    fs << "detectionThresh" << m_DetectionCorrespThresh;
    fs << "interactionThresh" << m_InteractionThresh;
    
    fs << "MultiviewStrategy" << m_MultiviewStrategy;
    fs << "multiviewLateFusionStrategy" << m_MultiviewLateFusionStrategy;
    fs << "leafSizeX" << m_LeafSize.x();
    fs << "leafSizeY" << m_LeafSize.y();
    fs << "leafSizeZ" << m_LeafSize.z();
    
    fs << "valInteractionOvlCriterion" << m_ValInteractionOvlCriterion;

    cv::Mat valCombsMat;
    cvx::convert<float>(m_ValCombs, valCombsMat);
    fs << "valCombsMat" << valCombsMat;

    fs << "numViews" << m_pSubtractor->getNumOfViews();
    fs << "numSequences" << ((int) m_Sequences.size());
    
    for (int v = 0; v < m_LateFusionScalings.size(); v++)
        fs << ("lateFusionScalings-" + boost::lexical_cast<std::string>(v)) << m_LateFusionScalings[v];

    for (int v = 0; v < m_InteractionResults.size(); v++)
        fs << ("interactionResults-" + boost::lexical_cast<std::string>(v)) << m_InteractionResults[v];
    
    for (int s = 0; s < m_InteractionPredictions.size(); s++)
        for (int v = 0; v < m_InteractionPredictions[s].size(); v++)
            fs << ("interactionPredictions-" + boost::lexical_cast<std::string>(s) + "-" + boost::lexical_cast<std::string>(v)) << m_InteractionPredictions[s][v];
    
    fs << "valPerf" << m_ValPerf;
    fs << "valIdx" << m_ValIdx;
    fs << "perf" << m_Perf;
    
    fs.release();
}

bool CloudjectInteractionPipeline::load(std::string filename, std::string extension)
{
    std::string filenameWithSuffix = filename + extension; // extension is expected to already include the ext dot (.xxx)
    cv::FileStorage fs;
    if (extension.compare(".xml") == 0)
        fs.open(filenameWithSuffix, cv::FileStorage::READ | cv::FileStorage::FORMAT_XML);
    else // yml
        fs.open(filenameWithSuffix, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    
    if (!fs.isOpened())
        return false;
    
    fs["correspCriterion"] >> m_CorrespCriterion;
    fs["detectionThresh"] >> m_DetectionCorrespThresh;
    fs["interactionThresh"] >> m_InteractionThresh;
    
    fs["MultiviewStrategy"] >> m_MultiviewStrategy;
    fs["multiviewLateFusionStrategy"] >> m_MultiviewLateFusionStrategy;

    float leafSizeX, leafSizeY, leafSizeZ;
    fs["leafSizeX"] >> leafSizeX;
    fs["leafSizeY"] >> leafSizeY;
    fs["leafSizeZ"] >> leafSizeZ;
    m_LeafSize = Eigen::Vector3f(leafSizeX,leafSizeY,leafSizeZ);
    
    fs["valInteractionOvlCriterion"] >> m_ValInteractionOvlCriterion;
    
    cv::Mat valCombsMat;
    fs["valCombsMat"] >> valCombsMat;
    cvx::convert<float>(valCombsMat, m_ValCombs);
    
    int V = 0;
    fs["numViews"] >> V;
    if (V == 0) V = m_pSubtractor->getNumOfViews();
    
    int S = 0;
    fs["numSequences"] >> S;
    if (S == 0) S = m_Sequences.size();
    
    if (m_MultiviewStrategy == DETECT_MULTIVIEW)
    {
        m_LateFusionScalings.resize(V);
        for (int v = 0; v < V; v++)
            fs["lateFusionScalings-" + boost::lexical_cast<std::string>(v)] >> m_LateFusionScalings[v];
        
        m_InteractionResults.resize(1);        
        fs["interactionResults-0"] >> m_InteractionResults[0];
        
        m_InteractionPredictions.resize(S);
        for (int s = 0; s < S; s++)
        {
            cv::Mat P;
            fs["interactionPredictions-" + boost::lexical_cast<std::string>(s) + "-0"] >> P;
            m_InteractionPredictions[s].resize(1);
            cvx::convert<int>(P, m_InteractionPredictions[s][0]);
        }
    }
    else if (m_MultiviewStrategy == DETECT_MONOCULAR)
    {
        m_InteractionResults.resize(V);
        for (int v = 0; v < V; v++)
            fs["interactionResults-" + boost::lexical_cast<std::string>(v)] >> m_InteractionResults[v];
 
        m_InteractionPredictions.resize(S);
        for (int s = 0; s < S; s++)
        {
            m_InteractionPredictions[s].resize(V);
            for (int v = 0; v < V; v++)
                fs["interactionPredictions-" + boost::lexical_cast<std::string>(s) + "-" + boost::lexical_cast<std::string>(v)] >> m_InteractionPredictions[s][v];
        }
    }
    
    fs["valPerf"] >> m_ValPerf;
    fs["valIdx"] >> m_ValIdx;
    fs["perf"] >> m_Perf;
    
    fs.release();
    
    return true;
}

std::vector<std::vector<Result> > CloudjectInteractionPipeline::getDetectionResults()
{
    return m_DetectionResults; // one per view
}


// =============================================================================
//
//
//
//  Private methods
//
//
//
// =============================================================================


void CloudjectInteractionPipeline::setValidationParametersCombination(std::vector<float> combination)
{
    setCorrespondenceCriterion(combination[0]);
    setDetectionCorrespondenceThresh(combination[1]);
    setInteractionThresh(combination[2]);
    
    if (m_MultiviewStrategy == DETECT_MONOCULAR)
    {
        // nothing particular here
    }
    else if (m_MultiviewStrategy == DETECT_MULTIVIEW)
    {
        setMultiviewLateFusionStrategy(combination[3]);
        setMultiviewCorrespondenceThresh(combination[4]);
    }
}

//
// Detection
//

void CloudjectInteractionPipeline::predictDetectionMonocular()
{
    m_DetectionResults.clear();
    m_DetectionResults.resize(m_pSubtractor->getNumOfViews(), std::vector<Result>(m_Categories.size()));
    
    for (int s = 0; s < m_Sequences.size(); s++)
    {
        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
        + m_Sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        
        int V = m_Sequences[s]->getNumOfViews();
        
#ifdef DEBUG_VISUALIZE_DETECTIONS
        MultiviewVisualizer viz;
        viz.create(V, m_pRegisterer);
#endif
        m_Sequences[s]->restart();
        while (m_Sequences[s]->hasNextFrames())
        {
            vector<ColorDepthFrame::Ptr> frames = m_Sequences[s]->nextFrames();
            m_pRegisterer->setInputFrames(frames);
            m_pRegisterer->registrate(frames);
            
            std::vector<cv::Mat> tabletops;
            std::vector<cv::Mat> interactions;
            m_pTableModeler->getTabletopMask(frames, tabletops);
            m_pTableModeler->getInteractionMask(frames, interactions);
            
            vector<string> fids = m_Sequences[s]->getFramesFilenames();
            
            // Prepare the interactors from the different views
            
            std::vector<ColorPointCloudPtr> interactors (m_Sequences[s]->getNumOfViews());
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
            {
                interactors[v] = ColorPointCloudPtr(new ColorPointCloud);
                cv::Mat aux;
                cvx::open(interactions[v] & ~tabletops[v], 2, aux);
                frames[v]->getRegisteredColoredPointCloud(aux, *(interactors[v]));
            }
            
            // Distinguish actors from interactors
            std::vector<std::vector<Cloudject::Ptr> > nonInteractedActors (m_Sequences[s]->getNumOfViews());
            std::vector<std::vector<Cloudject::Ptr> > interactedActors (m_Sequences[s]->getNumOfViews());
            
            for (int v = 0; v < V; v++)
            {
                std::vector<ForegroundRegion> regionsF;
                remedi::io::readPredictionRegions(resultsParent + string(CLOUDJECTS_DIRNAME)
                                                  + m_Sequences[s]->getViewName(v), fids[v], regionsF, true);
                
                // Read the cloudjects representing the selected regions
                std::vector<Cloudject::Ptr> cloudjectsF;
                remedi::io::readCloudjectsWithDescriptor(resultsParent + string(CLOUDJECTS_DIRNAME) + m_Sequences[s]->getViewName(v), fids[v], regionsF, "pfhrgb250", cloudjectsF, true);
                
                for (int i = 0; i < cloudjectsF.size(); i++)
                {
                    cloudjectsF[i]->setLeafSize(m_LeafSize);
                    cloudjectsF[i]->setRegistrationTransformation(frames[v]->getRegistrationTransformation());
                }
                
//                findActors(cloudjectsF, interactors, nonInteractedActors[v], m_LeafSize, interactedActors[v]);
            }
            
            for (int v = 0; v < V; v++)
            {
                std::vector<std::vector<float> > margins (nonInteractedActors[v].size());
                for (int i = 0; i < nonInteractedActors[v].size(); i++)
                {
                    std::vector<int> predictions;
                    std::vector<float> distsToMargin;
                    predictions = m_ClassificationPipeline->predict(nonInteractedActors[v][i], distsToMargin);
                    nonInteractedActors[v][i]->setLikelihoods(distsToMargin);
                    
                    for (int j = 0; j < predictions.size(); j++)
                        if (predictions[j] > 0)
                            nonInteractedActors[v][i]->addRegionLabel(m_Categories[j]);
                    
                    margins[i] = distsToMargin;
                }
                
                std::vector<int> indices;
                maxPooling(nonInteractedActors[v], indices);
                
                // Evaluation
                
                std::vector<Result> resultsFrame;
                evaluateDetectionInFrame (frames[v],
                                          m_Gt.at(m_Sequences[s]->getName()).at(m_Sequences[s]->getViewName(v)).at(fids[v]),
                                          nonInteractedActors[v],
                                          indices,
                                          m_LeafSize,
                                          resultsFrame);
                
                for (int i = 0; i < resultsFrame.size(); i++)
                    m_DetectionResults[v][i] += resultsFrame[i];
            }
            
#ifdef DEBUG_VISUALIZE_DETECTIONS
//            viz.visualize(interactors, interactedActors, nonInteractedActors);
#endif //DEBUG_VISUALIZE_DETECTIONS
        }
    }
}

void CloudjectInteractionPipeline::predictDetectionMultiview()
{
    m_DetectionResults.clear();
    m_DetectionResults.resize(1, std::vector<Result>(m_Categories.size()));
    
    for (int s = 0; s < m_Sequences.size(); s++)
    {
        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
        + m_Sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        
        int V = m_Sequences[s]->getNumOfViews();
#ifdef DEBUG_VISUALIZE_DETECTIONS
        MultiviewVisualizer viz;
        viz.create(V, m_pRegisterer);
        std::cout << "[" << std::endl;
#endif
        m_Sequences[s]->restart();
        while (m_Sequences[s]->hasNextFrames())
        {
            vector<ColorDepthFrame::Ptr> frames = m_Sequences[s]->nextFrames();
            m_pRegisterer->setInputFrames(frames);
            m_pRegisterer->registrate(frames);
            
            std::vector<cv::Mat> tabletops;
            std::vector<cv::Mat> interactions;
            m_pTableModeler->getTabletopMask(frames, tabletops);
            m_pTableModeler->getInteractionMask(frames, interactions);
            
            vector<string> fids = m_Sequences[s]->getFramesFilenames();
            
            // Prepare the interactors from the different views
            
            std::vector<ColorPointCloudPtr> interactors (m_Sequences[s]->getNumOfViews());
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
            {
                interactors[v] = ColorPointCloudPtr(new ColorPointCloud);
                cv::Mat aux;
                cvx::open(interactions[v] & ~tabletops[v], 2, aux);
                frames[v]->getRegisteredColoredPointCloud(aux, *(interactors[v]));
            }
            
            // Distinguish actors from interactors
            std::vector<std::vector<Cloudject::Ptr> > nonInteractedActors (m_Sequences[s]->getNumOfViews());
            std::vector<std::vector<Cloudject::Ptr> > interactedActors (m_Sequences[s]->getNumOfViews());
            
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
            {
                std::vector<ForegroundRegion> regionsF;
                remedi::io::readPredictionRegions(resultsParent + string(CLOUDJECTS_DIRNAME)
                                                  + m_Sequences[s]->getViewName(v), fids[v], regionsF, true);
                
                // Read the cloudjects representing the selected regions
                std::vector<Cloudject::Ptr> cloudjectsF;
                remedi::io::readCloudjectsWithDescriptor(resultsParent + string(CLOUDJECTS_DIRNAME) + m_Sequences[s]->getViewName(v), fids[v], regionsF, "pfhrgb250", cloudjectsF, true);
                
                for (int i = 0; i < cloudjectsF.size(); i++)
                {
                    cloudjectsF[i]->setLeafSize(m_LeafSize);
                    cloudjectsF[i]->setRegistrationTransformation(frames[v]->getRegistrationTransformation());
                }
                
//                findActors(cloudjectsF, interactors, nonInteractedActors[v], m_LeafSize, interactedActors[v]);
            }
            
            // Find the correspondences among actor clouds
            std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences;
            // Using grids
            std::vector<std::vector<VoxelGridPtr> > grids;
            findCorrespondences(frames, nonInteractedActors, m_MultiviewCorrespThresh, correspondences, grids); // false (not to registrate)
            
            for (int i = 0; i < correspondences.size(); i++)
            {
                for (int j = 0; j < correspondences[i].size(); j++)
                {
                    std::vector<float> distsToMargin;
                    m_ClassificationPipeline->predict(correspondences[i][j].second, distsToMargin);
                    correspondences[i][j].second->setLikelihoods(distsToMargin);
                    
                    for (int k = 0; k < m_Categories.size(); k++)
                        if (distsToMargin[k] < 0) correspondences[i][j].second->addRegionLabel(m_Categories[k]);
                }
            }
            
            fuseCorrespondences(correspondences);
            
            std::vector<int> indices;
            maxPooling(correspondences, indices);
            
            // Evaluation
            
            std::vector<std::map<std::string,std::map<std::string,GroundtruthRegion> > > annotationsF (V);
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
                annotationsF[v] = m_Gt.at(m_Sequences[s]->getName()).at(m_Sequences[s]->getViewName(v)).at(fids[v]);
            
            std::vector<Result> resultsFrame;
            evaluateDetectionInFrame(frames, annotationsF, correspondences, indices, m_LeafSize, resultsFrame);
            
            for (int i = 0; i < resultsFrame.size(); i++)
                m_DetectionResults[0][i] += resultsFrame[i];
            
#ifdef DEBUG_VISUALIZE_DETECTIONS
//            viz.visualize(interactors, interactedActors, correspondences);
#endif //DEBUG_VISUALIZE_DETECTIONS
        }
    }
}

//
// Interaction
//

void CloudjectInteractionPipeline::getActorsFromFrames(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > cloudjects, std::vector<ColorPointCloudPtr>& interactorClouds, std::vector<std::vector<Cloudject::Ptr> >& interactedActorCloudjects, std::vector<std::vector<Cloudject::Ptr> >& freeActorCloudjects)
{
    interactorClouds.clear();
    interactedActorCloudjects.clear();
    freeActorCloudjects.clear();
    
    std::vector<cv::Mat> foregroundMasks;
    m_pSubtractor->setInputFrames(frames);
    m_pSubtractor->subtract(foregroundMasks);
    
    m_pRegisterer->setInputFrames(frames);
    m_pRegisterer->registrate(frames);
    
    std::vector<cv::Mat> tabletopMasks;
    std::vector<cv::Mat> interactionMasks;
    m_pTableModeler->getTabletopMask(frames, tabletopMasks);
    m_pTableModeler->getInteractionMask(frames, interactionMasks);
    
    // Prepare the scenes
    std::vector<ColorPointCloudPtr> outerInteractionScenes (frames.size());
    std::vector<ColorPointCloudPtr> actorScenes (frames.size());
    for (int v = 0; v < frames.size(); v++)
    {
        cv::Mat outerInteractionMask = interactionMasks[v] & ~tabletopMasks[v];
        cv::Mat actorsMask = tabletopMasks[v] & foregroundMasks[v];
        
        outerInteractionScenes[v] = ColorPointCloudPtr(new ColorPointCloud);
        actorScenes[v] = ColorPointCloudPtr(new ColorPointCloud);
        frames[v]->getRegisteredColoredPointCloud(outerInteractionMask, *(outerInteractionScenes[v]));
        frames[v]->getRegisteredColoredPointCloud(actorsMask, *(actorScenes[v]));
    }
    
    // Determine interactors from scenes
    interactorClouds.resize(frames.size());
    for (int v = 0; v < frames.size(); v++)
    {
        ColorPointCloudPtr downsampledInteractionScene (new ColorPointCloud);
        pclx::downsample(outerInteractionScenes[v], Eigen::Vector3f(.01f,.01f,.01f), *downsampledInteractionScene);
        
        interactorClouds[v] = ColorPointCloudPtr(new ColorPointCloud);
        pclx::biggestEuclideanCluster(downsampledInteractionScene, 2 * .01f, 100, *(interactorClouds[v]));
    }
    
    // Distinguish interacted from free actors (using the interactors)
    interactedActorCloudjects.resize(frames.size());
    freeActorCloudjects.resize(frames.size());
    for (int v = 0; v < frames.size(); v++)
    {
        ColorPointCloudPtr downsampledActorScene (new ColorPointCloud);
        pclx::downsample(actorScenes[v], Eigen::Vector3f(.01f,.01f,.01f), *downsampledActorScene);
        
        std::vector<ColorPointCloudPtr> actorClouds;
        pclx::clusterize<pcl::PointXYZRGB>(downsampledActorScene, 2 * .01f, 1, actorClouds);
        
        std::vector<ColorPointCloudPtr> freeActorClouds, interactedActorClouds;
        for (int i = 0; i < actorClouds.size(); i++)
        {
            bool bIsInteractive = isInteractive(actorClouds[i], interactorClouds[v], m_InteractionThresh);
            if (!bIsInteractive)
                freeActorClouds.push_back(actorClouds[i]);
            else
                interactedActorClouds.push_back(actorClouds[i]);
        }
        

        
        for (int i = 0; i < cloudjects.size(); i++)
        {
            cloudjects[v][i]->setLeafSize(m_LeafSize);
            cloudjects[v][i]->setRegistrationTransformation(frames[v]->getRegistrationTransformation());
        }
        
        findInclusions(cloudjects[v], freeActorClouds, m_LeafSize, freeActorCloudjects[v]);
        findInclusions(cloudjects[v], interactedActorClouds, m_LeafSize, interactedActorCloudjects[v]);
    }
}

void CloudjectInteractionPipeline::detectInteractions(std::vector<std::vector<Cloudject::Ptr> > freeActorCloudjects, std::vector<std::vector<int> >& interactions)
{
    interactions.clear();
    
    interactions.resize(freeActorCloudjects.size());
    for (int v = 0; v < freeActorCloudjects.size(); v++)
    {
        std::vector<std::vector<float> > margins (freeActorCloudjects[v].size());
        for (int i = 0; i < freeActorCloudjects[v].size(); i++)
        {
            std::vector<int> predictions;
            std::vector<float> distsToMargin;
            predictions = m_ClassificationPipeline->predict(freeActorCloudjects[v][i], distsToMargin);
            freeActorCloudjects[v][i]->setLikelihoods(distsToMargin);
            
            for (int j = 0; j < predictions.size(); j++)
                if (predictions[j] > 0)
                    freeActorCloudjects[v][i]->addRegionLabel(m_Categories[j]);
            
            margins[i] = distsToMargin;
        }
        
        std::vector<int> indices;
        maxPooling(freeActorCloudjects[v], indices);
        
        std::vector<int> interactionsPr;
        interactionFromNegativeDetections(indices, interactions[v]);
    }
}

void CloudjectInteractionPipeline::findInclusions(std::vector<Cloudject::Ptr> cloudjectsIn, std::vector<ColorPointCloudPtr> cloudsIn, Eigen::Vector3f leafSize, std::vector<Cloudject::Ptr>& cloudjectsOut)
{
    cloudjectsOut.clear();
    cloudjectsOut.reserve(cloudjectsIn.size());
    
    std::vector<VoxelGridPtr> cloudsGrids (cloudsIn.size());
    for (int i = 0; i < cloudsIn.size(); i++)
    {
        ColorPointCloud cloudOut;
        VoxelGridPtr pGrid (new VoxelGrid);
        pclx::voxelize(cloudsIn[i], leafSize, cloudOut, *pGrid);
        cloudsGrids[i] = pGrid;
    }

    for (int i = 0; i < cloudjectsIn.size(); i++)
    {
        ColorPointCloudPtr cloudjectCloudOut (new ColorPointCloud);
        VoxelGridPtr cloudjectGrid (new VoxelGrid);
        
        pclx::voxelize(cloudjectsIn[i]->getRegisteredCloud(), leafSize, *cloudjectCloudOut, *cloudjectGrid);
        
        bool bIncluded = false;
        for (int j = 0; j < cloudsIn.size() && !bIncluded ; j++)
        {
            VoxelGridPtr cloudGrid = cloudsGrids[j];
            
            if ( (m_CorrespCriterion == CORRESP_INC && pclx::computeInclusion3d(*cloudjectGrid, *cloudGrid) > m_MultiviewCorrespThresh)
                || (m_CorrespCriterion == CORRESP_OVL && pclx::computeOverlap3d(*cloudjectGrid, *cloudGrid) > m_MultiviewCorrespThresh) )
                bIncluded = true;
        }
        
        if (bIncluded)
            cloudjectsOut.push_back(cloudjectsIn[i]);
    }
}

void CloudjectInteractionPipeline::predictInteractionMonocular()
{
    m_InteractionCombResults.clear();
    m_InteractionCombResults.resize(m_pSubtractor->getNumOfViews(), std::vector<Result>(m_Categories.size()));
    
    for (int s = 0; s < m_Sequences.size(); s++)
    {
        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
        + m_Sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        
#ifdef DEBUG_VISUALIZE_DETECTIONS
        MultiviewVisualizer viz;
        viz.create(V, m_pRegisterer);
#endif
        // In prediction tiime (not used in validation timem since m_InteractionPredictions is 0)
        std::vector<int> n;
        m_Sequences[s]->getNumOfFramesOfViews(n);
        for (int v = 0; v < m_InteractionPredictions[s].size(); v++)
            m_InteractionPredictions[s][v] = cv::Mat(n[v], m_Categories.size(), cv::DataType<int>::type, cv::Scalar(0));
        
        int f = 0;
        m_Sequences[s]->restart();
        while (m_Sequences[s]->hasNextFrames())
        {
            vector<ColorDepthFrame::Ptr> frames = m_Sequences[s]->nextFrames();

            std::vector<std::vector<Cloudject::Ptr> > cloudjectsF (frames.size());
            for (int v = 0; v < frames.size(); v++)
            {
                // Load cloudjects from disk and set leaf size and registration
                std::vector<ForegroundRegion> regionsF;
                remedi::io::readPredictionRegions(resultsParent + string(CLOUDJECTS_DIRNAME)
                                                  + m_Sequences[s]->getViewName(v), frames[v]->getFilename(), regionsF, true);
                
                remedi::io::readCloudjectsWithDescriptor(resultsParent + string(CLOUDJECTS_DIRNAME) + m_Sequences[s]->getViewName(v), frames[v]->getFilename(), regionsF, "pfhrgb250", cloudjectsF[v], true);
            }
            
            std::vector<ColorPointCloudPtr> interactorClouds;
            std::vector<std::vector<Cloudject::Ptr> > interactedActorCloudjects, freeActorCloudjects;
            getActorsFromFrames(frames, cloudjectsF, interactorClouds, interactedActorCloudjects, freeActorCloudjects);
            
            std::vector<std::vector<int> > interactions;
            detectInteractions(freeActorCloudjects, interactions);
            
            // Output the predictions if m_InteractionPredictions has been initialized outside of this method calling
            // (this only happens when predicting, not validating)
            if (!m_InteractionPredictions.empty())
            {
                std::vector<int> delays = m_Sequences[s]->getDelays();
                
                for (int v = 0; v < m_InteractionPredictions[s].size(); v++)
                    for (int k = 0; k < m_Categories.size(); k++)
                        m_InteractionPredictions[s][v].at<int>(f+delays[v], k) = interactions[v][k];
            }
            
            // Evaluation in separate views
            for (int v = 0; v < frames.size(); v++)
            {
                std::vector<Result> results;
                evaluateInteractionInFrame(interactions[v], m_IAct.at(m_Sequences[s]->getName()).at(m_Sequences[s]->getViewName(v)).at(frames[v]->getFilename()), results);
                for (int i = 0; i < results.size(); i++)
                    m_InteractionCombResults[v][i] += results[i];
            }
#ifdef DEBUG
            for (int v = 0; v < frames.size(); v++)
            {
                std::vector<int> gt = m_IAct.at(m_Sequences[s]->getName()).at(m_Sequences[s]->getViewName(v)).at(fids[v]);
                std::copy(gt[f].begin(), gt[f].end(), std::ostream_iterator<int>(std::cout, "\t"));
                std::cout << ", ";
            } std::cout << endl;
            
            for (int v = 0; v < frames.size(); v++)
            {
                std::copy(interactions[v].begin(), interactions[v].end(), std::ostream_iterator<int>(std::cout, "\t"));
                std::cout << ", ";
            } std::cout << std::endl;
#endif

#ifdef DEBUG_VISUALIZE_DETECTIONS
            viz.visualize(interactorClouds, interactedActorCloudjects, freeActorCloudjects);
#endif //DEBUG_VISUALIZE_DETECTIONS
            
            f++;
        }
    }
}

void CloudjectInteractionPipeline::predictInteractionMultiview()
{
    m_InteractionCombResults.clear();
    m_InteractionCombResults.resize(1, std::vector<Result>(m_Categories.size()));
    
    for (int s = 0; s < m_Sequences.size(); s++)
    {
        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
        + m_Sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        
        int V = m_Sequences[s]->getNumOfViews();
#ifdef DEBUG_VISUALIZE_DETECTIONS
        MultiviewVisualizer viz;
        viz.create(V, m_pRegisterer);
        std::cout << "[" << std::endl;
#endif
        
        std::vector<std::vector<int> > P (m_Sequences[s]->getMinNumOfFrames());
        std::vector<std::vector<int> > G (m_Sequences[s]->getMinNumOfFrames());
        
        std::vector<int> interactionsPrPrev (m_Categories.size(), 0);
        std::vector<int> interactionsGtPrev (m_Categories.size(), 0);
        
        int f = 0;
        m_Sequences[s]->restart();
        while (m_Sequences[s]->hasNextFrames())
        {
            vector<ColorDepthFrame::Ptr> frames = m_Sequences[s]->nextFrames();
            
            std::vector<cv::Mat> foregroundMasks;
            m_pSubtractor->setInputFrames(frames);
            m_pSubtractor->subtract(foregroundMasks);
            
            m_pRegisterer->setInputFrames(frames);
            m_pRegisterer->registrate(frames);
            
            std::vector<cv::Mat> tabletopMasks;
            std::vector<cv::Mat> interactionMasks;
            m_pTableModeler->getTabletopMask(frames, tabletopMasks);
            m_pTableModeler->getInteractionMask(frames, interactionMasks);
            
            vector<string> fids = m_Sequences[s]->getFramesFilenames();
            
            // Prepare the interactors from the different views
            
            std::vector<ColorPointCloudPtr> outerInteractionScenes (m_Sequences[s]->getNumOfViews());
            std::vector<ColorPointCloudPtr> actorScenes (m_Sequences[s]->getNumOfViews());
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
            {
                cv::Mat outerInteractionMask = interactionMasks[v] & ~tabletopMasks[v];
                cv::Mat actorsMask = tabletopMasks[v] & foregroundMasks[v];
                
                outerInteractionScenes[v] = ColorPointCloudPtr(new ColorPointCloud);
                actorScenes[v] = ColorPointCloudPtr(new ColorPointCloud);
                frames[v]->getRegisteredColoredPointCloud(outerInteractionMask, *(outerInteractionScenes[v]));
                frames[v]->getRegisteredColoredPointCloud(actorsMask, *(actorScenes[v]));
            }
            
            // Distinguish actors from interactors
            std::vector<ColorPointCloudPtr> interactorClouds (m_Sequences[s]->getNumOfViews());
            for (int v = 0; v < V; v++)
            {
                ColorPointCloudPtr downsampledInteractionScene (new ColorPointCloud);
                pclx::downsample(outerInteractionScenes[v], Eigen::Vector3f(.01f,.01f,.01f), *downsampledInteractionScene);
                
                interactorClouds[v] = ColorPointCloudPtr(new ColorPointCloud);
                pclx::biggestEuclideanCluster(downsampledInteractionScene, m_InteractionThresh, 100, *(interactorClouds[v]));
            }
            
            std::vector<std::vector<Cloudject::Ptr> > interactedActorCloudjects (m_Sequences[s]->getNumOfViews());
            std::vector<std::vector<Cloudject::Ptr> > freeActorCloudjects (m_Sequences[s]->getNumOfViews());
            for (int v = 0; v < V; v++)
            {
                ColorPointCloudPtr downsampledActorScene (new ColorPointCloud);
                pclx::downsample(actorScenes[v], Eigen::Vector3f(.01f,.01f,.01f), *downsampledActorScene);
                
                std::vector<ColorPointCloudPtr> actorClouds;
                pclx::clusterize<pcl::PointXYZRGB>(downsampledActorScene, 2 * .01f, 1, actorClouds);
                
                std::vector<ColorPointCloudPtr> freeActorClouds, interactedActorClouds;
                for (int i = 0; i < actorClouds.size(); i++)
                {
                    bool bIsInteractive = false;
                    for (int j = 0; j < interactorClouds.size() && !bIsInteractive; j++)
                        bIsInteractive = isInteractive(actorClouds[i], interactorClouds[v], m_InteractionThresh);
                    
                    if (!bIsInteractive)
                        freeActorClouds.push_back(actorClouds[i]);
                    else
                        interactedActorClouds.push_back(actorClouds[i]);
                }
                
                // Load cloudjects from disk and set leaf size and registration
                std::vector<ForegroundRegion> regionsF;
                remedi::io::readPredictionRegions(resultsParent + string(CLOUDJECTS_DIRNAME)
                                                  + m_Sequences[s]->getViewName(v), fids[v], regionsF, true);
                
                std::vector<Cloudject::Ptr> cloudjects;
                remedi::io::readCloudjectsWithDescriptor(resultsParent + string(CLOUDJECTS_DIRNAME) + m_Sequences[s]->getViewName(v), fids[v], regionsF, "pfhrgb250", cloudjects, true);
                
                for (int i = 0; i < cloudjects.size(); i++)
                {
                    cloudjects[i]->setLeafSize(m_LeafSize);
                    cloudjects[i]->setRegistrationTransformation(frames[v]->getRegistrationTransformation());
                }
                
                findInclusions(cloudjects, freeActorClouds, m_LeafSize, freeActorCloudjects[v]);
                findInclusions(cloudjects, interactedActorClouds, m_LeafSize, interactedActorCloudjects[v]);
            }
            
            // Find the correspondences among actor clouds
            std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences;
            // Using grids
            std::vector<std::vector<VoxelGridPtr> > grids;
            findCorrespondences(frames, freeActorCloudjects, m_MultiviewCorrespThresh, correspondences, grids); // false (not to registrate)
            
            for (int i = 0; i < correspondences.size(); i++)
            {
                for (int j = 0; j < correspondences[i].size(); j++)
                {
                    std::vector<float> distsToMargin;
                    m_ClassificationPipeline->predict(correspondences[i][j].second, distsToMargin);
                    correspondences[i][j].second->setLikelihoods(distsToMargin);
                    
                    for (int k = 0; k < m_Categories.size(); k++)
                        if (distsToMargin[k] < 0) correspondences[i][j].second->addRegionLabel(m_Categories[k]);
                }
            }
            
            fuseCorrespondences(correspondences);
            
            std::vector<int> indices;
            maxPooling(correspondences, indices);
            
            std::vector<int> interactionsPr;
            interactionFromNegativeDetections(indices, interactionsPr);
            
            // Evaluation
            
            std::vector<int> interactionsGt (m_Categories.size(), 0);
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
            {
                std::vector<int> interactionsViewGt = m_IAct.at(m_Sequences[s]->getName()).at(m_Sequences[s]->getViewName(v)).at(fids[v]);
                for (int k = 0; k < m_Categories.size(); k++)
                    interactionsGt[k] = interactionsViewGt[k];
            }
            
            std::vector<Result> results;
            evaluateInteractionInFrame(interactionsPr, interactionsGt, results);
            for (int i = 0; i < results.size(); i++)
                m_InteractionCombResults[0][i] += results[i];
            
            cv::Mat interactionsPrBE, interactionsGtBE;
            if (m_ValInteractionOvlCriterion == INTERACTION_OVL_BEGINEND)
            {
                firstDerivative(interactionsPrPrev, interactionsPr, P[f]);
                firstDerivative(interactionsGtPrev, interactionsGt, G[f]);
            }
            else
            {
                P[f] = interactionsPr;
                G[f] = interactionsGt;
            }
            
            // debug
            // -----
//            std::copy(P[f].begin(), P[f].end(), std::ostream_iterator<int>(std::cout, "\t"));
//            std::cout << ";" << std::endl;
//            std::copy(G[f].begin(), G[f].end(), std::ostream_iterator<int>(std::cout, "\t"));
//            std::cout << ";" << std::endl;
//            std::cout << std::endl;
            //
            
#ifdef DEBUG_VISUALIZE_DETECTIONS
            viz.visualize(interactorClouds, interactedActorCloudjects, correspondences);
#endif //DEBUG_VISUALIZE_DETECTIONS
            
            interactionsPrPrev = interactionsPr;
            interactionsGtPrev = interactionsGt;
            f++;
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

void CloudjectInteractionPipeline::findActors(std::vector<ColorPointCloudPtr> actors, std::vector<ColorPointCloudPtr> interactors, std::vector<ColorPointCloudPtr>& nonInteractedActors, Eigen::Vector3f leafSize, std::vector<ColorPointCloudPtr>& interactedActors)
{
    nonInteractedActors.clear();
    interactedActors.clear();
    
    std::vector<bool> interactionsMask;
    findInteractions(actors, interactors, interactionsMask, leafSize);
    
    assert (interactionsMask.size() == actors.size());
    
    nonInteractedActors.reserve(actors.size());
    interactedActors.reserve(actors.size());
    
    for (int i = 0; i < actors.size(); i++)
    {
        if (!interactionsMask[i])
            nonInteractedActors.push_back(actors[i]);
        else
            interactedActors.push_back(actors[i]);
    }
}

void CloudjectInteractionPipeline::findInteractions(std::vector<ColorPointCloudPtr> candidates, std::vector<ColorPointCloudPtr> interactors, std::vector<bool>& mask, Eigen::Vector3f leafSize)
{
    mask.clear();
    mask.resize(candidates.size(), false);
    
    std::vector<ColorPointCloudPtr> interactorsFilt;
    bool bMainInteractorsPresent = false;
    
    for (int i = 0; i < interactors.size(); i++)
    {
        ColorPointCloudPtr pInteractorFilt (new ColorPointCloud);
        downsample(interactors[i], leafSize.x(), *pInteractorFilt);
        
        ColorPointCloudPtr pMainInteractorFilt (new ColorPointCloud);
        pclx::biggestEuclideanCluster(pInteractorFilt, leafSize.x(), 250, *pMainInteractorFilt);
        
        if (pMainInteractorFilt->size() > 0)
            interactorsFilt.push_back(pMainInteractorFilt);
    }
    
    if (!interactorsFilt.empty())
    {
        std::vector<ColorPointCloudPtr> candidatesCloudsFilt (candidates.size());
        for (int i = 0; i < candidates.size(); i++)
        {
            ColorPointCloudPtr pCandidateCloudFilt (new ColorPointCloud);
            downsample(candidates[i], leafSize.x(), *pCandidateCloudFilt);
            
            candidatesCloudsFilt[i] = pCandidateCloudFilt;
        }
        
        // Check for each cluster that lies on the table
        for (int j = 0; j < interactorsFilt.size(); j++) for (int i = 0; i < candidatesCloudsFilt.size(); i++)
            if (isInteractive(candidatesCloudsFilt[i], interactorsFilt[j], m_InteractionThresh/*4.*leafSize.x()*/))
                mask[i] = true;
    }
}

void CloudjectInteractionPipeline::getDetectionGrids(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, bool bRegistrate, std::vector<std::vector<VoxelGridPtr> >& grids)
{
    grids.clear();
    
    grids.resize(detections.size());
    for (int v = 0; v < detections.size(); v++)
    {
        std::vector<VoxelGridPtr> viewGrids (detections[v].size());
        
        for (int i = 0; i < detections[v].size(); i++)
        {
            //            if (!bRegistrate)
            //                viewGrids[i] = detections[v][i]->getCloudCentroid();
            //            else
            //                viewGrids[i] = detections[v][i]->getRegisteredCloudCentroid();
            viewGrids[i] = detections[v][i]->getRegisteredGrid();
        }
        
        grids[v] = viewGrids;
    }
}

bool CloudjectInteractionPipeline::isInteractive(ColorPointCloudPtr tabletopRegionCluster, ColorPointCloudPtr interactionCloud, float tol)
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

void CloudjectInteractionPipeline::findCorrespondences(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, float tol, std::vector<std::vector<std::pair<int,Cloudject::Ptr> > >& correspondences, std::vector<std::vector<VoxelGridPtr> >& grids)
{
    correspondences.clear();
    
    // Correspondences found using the positions of the clouds' centroids
    //    std::vector<std::vector<PointT> > positions;
    getDetectionGrids(frames, detections, true, grids); // registrate is "true"
    
    std::vector<std::vector<std::pair<std::pair<int,int>,VoxelGridPtr> > > _correspondences;
    
    if (m_CorrespCriterion == CloudjectInteractionPipeline::CORRESP_INC)
        pclx::findCorrespondencesBasedOnInclusion(grids, tol, _correspondences);
    else if (m_CorrespCriterion == CloudjectInteractionPipeline::CORRESP_OVL)
        pclx::findCorrespondencesBasedOnOverlap(grids, tol, _correspondences);
    
    // Bc we want to have chains of clouds (not points) and the views to which they correspond, transform _correspondences -> correspondences
    correspondences.clear();
    for (int i = 0; i < _correspondences.size(); i++)
    {
        std::vector<std::pair<int,Cloudject::Ptr> > chain; // clouds chain
        for (int j = 0; j < _correspondences[i].size(); j++)
        {
            int v   = _correspondences[i][j].first.first;
            int idx = _correspondences[i][j].first.second;
            
            chain.push_back( std::pair<int,Cloudject::Ptr>(v, detections[v][idx]) );
        }
        correspondences.push_back( chain );
    }
}

void CloudjectInteractionPipeline::fuseCorrespondences(const std::vector<std::vector<std::pair<int, Cloudject::Ptr> > >& correspondences)
{
    for (int i = 0; i < correspondences.size(); i++)
    {
        // Calculate the fused margins (depending on the normalization criterion)
        std::vector<float>  marginsFused (m_Categories.size());
        if (m_MultiviewLateFusionStrategy == MULTIVIEW_LF_OR)
        {
            for (int k = 0; k < m_Categories.size(); k++)
            {
                marginsFused[k] = 1;
                for (int j = 0; j < correspondences[i].size() && (marginsFused[k] > 0); j++)
                {
                    if (correspondences[i][j].second->getLikelihoods()[k] < 0)
                        marginsFused[k] = -1;
                }
            }
        }
        else if (m_MultiviewLateFusionStrategy == MULTIVIEW_LFSCALE_SUMDIV)
        {
            for (int j = 0; j < correspondences[i].size(); j++)
            {
                std::vector<float> distsToMargin = correspondences[i][j].second->getLikelihoods();
                
                float sum = 0.f;
                for (int k = 0; k < m_Categories.size(); k++)
                    sum += pow(distsToMargin[k],2);
                
                for (int k = 0; k < distsToMargin.size(); k++)
                    distsToMargin[k] /= sum;
                
                correspondences[i][j].second->setLikelihoods(distsToMargin);
            }
            
            for (int k = 0; k < m_Categories.size(); k++)
            {
                float distToMarginFused = .0f;
                float W = .0f;
                for (int j = 0; j < correspondences[i].size(); j++)
                {
                    std::vector<float> distsToMargin = correspondences[i][j].second->getLikelihoods();
                    pcl::PointXYZ c = correspondences[i][j].second->getCloudCentroid();
                    // Inverse distance weighting
                    float wsq = 1.f / sqrt(pow(c.x,2) + pow(c.y,2) + pow(c.z,2));
                    // Scale by the dispersion of either positive or negative margins
                    distToMarginFused += (wsq * (distsToMargin[k]));
                    W += wsq;
                }
                marginsFused[k] = distToMarginFused/W;
            }
        }
        else if (m_MultiviewLateFusionStrategy == MULTIVIEW_LFSCALE_DEV)
        {
            for (int k = 0; k < m_Categories.size(); k++)
            {
                float distToMarginFused = .0f;
                float W = .0f;
                for (int j = 0; j < correspondences[i].size(); j++)
                {
                    float distToMargin = correspondences[i][j].second->getLikelihoods()[k];
                    pcl::PointXYZ c = correspondences[i][j].second->getCloudCentroid();
                    // Inverse distance weighting
                    float wsq = 1.f / sqrt(pow(c.x,2) + pow(c.y,2) + pow(c.z,2));
                    // Scale by the dispersion of either positive or negative margins
                    float s = (distToMargin < 0) ? m_LateFusionScalings[k][1] : m_LateFusionScalings[k][0];
                    distToMarginFused += (wsq * (distToMargin / abs(s)));
                    W += wsq;
                }
                marginsFused[k] = distToMarginFused/W;
            }
        }
        else if (m_MultiviewLateFusionStrategy == MULTIVIEW_LF_FURTHEST)
        {
            for (int k = 0; k < m_Categories.size(); k++)
            {
                marginsFused[k] = correspondences[i][0].second->getLikelihoods()[k];
                for (int j = 1; j < correspondences[i].size() && (marginsFused[k] > 0); j++)
                {
                    if (correspondences[i][j].second->getLikelihoods()[k] < marginsFused[k])
                        marginsFused[k] = correspondences[i][j].second->getLikelihoods()[k];
                }
            }
        }
        
        // Set the fused margins
        for (int j = 0; j < correspondences[i].size(); j++)
        {
            correspondences[i][j].second->setLikelihoods(marginsFused);
            
            correspondences[i][j].second->removeAllRegionLabels();
            for (int k = 0; k < m_Categories.size(); k++)
                if (marginsFused[k] < 0) correspondences[i][j].second->addRegionLabel(m_Categories[k]);
        }
    }
}

void CloudjectInteractionPipeline::maxPooling(std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences, std::vector<int>& indices)
{
    // This functions assumes the likelihoods of correspondences are homongenous
    indices.clear();
    indices.resize(m_Categories.size(), -1);
    
    for (int k = 0; k < m_Categories.size(); k++)
    {
        float lowestMargin = 0;
        for (int i = 0; i < correspondences.size(); i++)
        {
            float margin = correspondences[i][0].second->getLikelihoods()[k]; // assumption (index 0)
            if (margin < lowestMargin)
            {
                lowestMargin = margin;
                indices[k] = i;
            }
        }
    }
}

void CloudjectInteractionPipeline::maxPooling(std::vector<Cloudject::Ptr> detections, std::vector<int>& indices)
{
    indices.clear();
    indices.resize(m_Categories.size(), -1);
    
    for (int k = 0; k < m_Categories.size(); k++)
    {
        float lowestMargin = 0;
        for (int i = 0; i < detections.size(); i++)
        {
            float margin = detections[i]->getLikelihoods()[k]; // assumption (index 0)
            if (margin < lowestMargin)
            {
                lowestMargin = margin;
                indices[k] = i;
            }
        }
    }
}

void CloudjectInteractionPipeline::interactionFromNegativeDetections(std::vector<int> negatives, std::vector<int>& interaction)
{
    interaction.resize(negatives.size());
    for (int i = 0; i < negatives.size(); i++)
        interaction[i] = (negatives[i] < 0);
}

// Evaluate detection performance in a frame per object
void CloudjectInteractionPipeline::evaluateDetectionInFrame(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& gt, std::vector<Cloudject::Ptr> detections, std::vector<int> indices, Eigen::Vector3f leafSize, std::vector<Result>& result)
{
    result.clear();
    result.resize(m_Categories.size());
    
    std::map<std::string,std::map<std::string,VoxelGridPtr> > gtgrids;
    precomputeGrids(frame, gt, leafSize, gtgrids, false);
    
    for (int k = 0; k < m_Categories.size(); k++)
    {
        if (indices[k] < 0)
            continue;
        
        bool bMatched = false;
        VoxelGridPtr pDetectionGrid = detections[indices[k]]->getGrid();
        
        if (gt.count(m_Categories[k]) > 0)
        {
            std::map<std::string,VoxelGridPtr> categoryGrids = gtgrids[m_Categories[k]];
            std::map<std::string,VoxelGridPtr>::iterator it = categoryGrids.begin();
            for ( ; (it != categoryGrids.end()) && !bMatched; ++it)
            {
                VoxelGridPtr pAnnotGrid = it->second;
                float thresh;
                if (m_CorrespCriterion == CORRESP_INC)
                    thresh = pclx::computeInclusion3d(*pDetectionGrid, *pAnnotGrid);
                else if (m_CorrespCriterion == CORRESP_OVL)
                    thresh = pclx::computeOverlap3d(*pDetectionGrid, *pAnnotGrid);
                
                if (thresh > m_DetectionCorrespThresh)
                    bMatched = true;
            }
        }
        
        bMatched ? result[k].tp++ : result[k].fp++;
    }

    // If one of the categoryes undetected, but was annotated in one of the views' gt
    for (int k = 0; k < indices.size(); k++)
    {
        if (indices[k] < 0 && gt.count(m_Categories[k]) > 0) // annotated?
            result[k].fn++;
    }
}

// Evaluate detection performance in a frame per object (multiple views)
void CloudjectInteractionPipeline::evaluateDetectionInFrame(const std::vector<ColorDepthFrame::Ptr>& frames, const std::vector<std::map<std::string,std::map<std::string,GroundtruthRegion> > >& gt, std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences, std::vector<int> indices, Eigen::Vector3f leafSize, std::vector<Result>& result)
{
    result.clear();
    result.resize(m_Categories.size());
    
    std::vector<std::map<std::string,std::map<std::string,VoxelGridPtr> > > gtgrids (gt.size());
    for (int v = 0; v < gt.size(); v++)
        precomputeGrids(frames[v], gt[v], leafSize, gtgrids[v], true);
    
    for (int k = 0; k < m_Categories.size(); k++)
    {
        if (indices[k] < 0)
            continue;
        
        bool bMatched = false;
        for (int j = 0; j < correspondences[indices[k]].size() && !bMatched; j++)
        {
            VoxelGridPtr pDetectionGrid = correspondences[indices[k]][j].second->getRegisteredGrid();
            
            for (int v = 0; (v < gt.size()) && !bMatched; v++)
            {
                if (gt[v].count(m_Categories[k]) > 0)
                {
                    std::map<std::string,VoxelGridPtr> categoryGrids = gtgrids[v][m_Categories[k]];
                    std::map<std::string,VoxelGridPtr>::iterator it = categoryGrids.begin();
                    for ( ; (it != categoryGrids.end()) && !bMatched; ++it)
                    {
                        VoxelGridPtr pAnnotGrid = it->second;
                        float thresh;
                        if (m_CorrespCriterion == CORRESP_INC)
                            thresh = pclx::computeInclusion3d(*pDetectionGrid, *pAnnotGrid);
                        else if (m_CorrespCriterion == CORRESP_OVL)
                            thresh = pclx::computeOverlap3d(*pDetectionGrid, *pAnnotGrid);
                        
                        if (thresh > m_DetectionCorrespThresh)
                            bMatched = true;
                    }
                }
            }
        }
        
        bMatched ? result[k].tp++ : result[k].fp++;
    }
    
    
    // If one of the categoryes undetected, but was annotated in one of the views' gt
    for (int k = 0; k < indices.size(); k++)
    {
        if (indices[k] < 0)
        {
            bool bIsAnnotated = false;
            for (int v = 0; v < gt.size(); v++) // over views' gt
                if (gt[v].count(m_Categories[k]) > 0) // annotated?
                    bIsAnnotated = true;
            
            if (bIsAnnotated) result[k].fn++;
        }
    }
}

void CloudjectInteractionPipeline::evaluateInteractionInFrame(std::vector<int> prediction, std::vector<int> groundtruth, std::vector<Result>& result)
{
    assert (prediction.size() == groundtruth.size());
    
    result.resize(prediction.size());
    for (int k = 0; k < prediction.size(); k++)
    {
        if (prediction[k] == 1 && groundtruth[k] == 1)
            result[k].tp++;
        else if (prediction[k] == 1 && groundtruth[k] == 0)
            result[k].fp++;
        else if (prediction[k] == 0 && groundtruth[k] == 1)
            result[k].fn++;
        else if (prediction[k] == 0 && groundtruth[k] == 0)
            result[k].tn++;
    }
}

template<typename RegionT>
VoxelGridPtr CloudjectInteractionPipeline::computeGridFromRegion(ColorDepthFrame::Ptr frame, RegionT region, Eigen::Vector3f leafSize, bool bRegistrate)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    region.allocateMask();
    int pos[2] = {region.getRect().x, region.getRect().y};
    unsigned short range[2] = {MIN_DEPTH,MAX_DEPTH};
    MatToColoredPointCloud(frame->getDepth(), frame->getColor(), region.getMask(), pos, range, *pCloud);
    region.releaseMask();
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pRegCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    if (bRegistrate)
        pcl::transformPointCloud(*pCloud, *pRegCloud, frame->getRegistrationTransformation());
    
    pcl::PointCloud<pcl::PointXYZRGB> dummy;
    VoxelGridPtr pGrid (new VoxelGrid);
    pclx::voxelize((!bRegistrate ? pCloud : pRegCloud), leafSize, dummy, *pGrid);
    
    return pGrid;
}

void CloudjectInteractionPipeline::precomputeGrids(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& annotations, Eigen::Vector3f leafSize, std::map<std::string,std::map<std::string,VoxelGridPtr> >& grids, bool bRegistrate)
{
    std::map<std::string,std::map<std::string,GroundtruthRegion> >::const_iterator it;
    std::map<std::string,GroundtruthRegion>::const_iterator jt;
    for (it = annotations.begin(); it != annotations.end(); ++it)
        for (jt = it->second.begin(); jt != it->second.end(); ++jt)
            grids[it->first][jt->first] = computeGridFromRegion(frame, annotations.at(it->first).at(jt->first), leafSize, bRegistrate);
}

void CloudjectInteractionPipeline::precomputeGrids(ColorDepthFrame::Ptr frame, const std::vector<ForegroundRegion>& detections, Eigen::Vector3f leafSize, std::vector<VoxelGridPtr>& grids, bool bRegistrate)
{
    grids.resize(detections.size());
    for (int i = 0; i < detections.size(); ++i)
        grids[i] = computeGridFromRegion(frame, detections[i], leafSize, bRegistrate);
}

void CloudjectInteractionPipeline::firstDerivative(std::vector<int> inputPrev, std::vector<int> inputCurr, std::vector<int>& derivative)
{
    assert(inputPrev.size() == inputCurr.size());
    
    derivative.resize(inputCurr.size(), 0);
    for (int i = 0; i < inputCurr.size(); i++)
    {
        if (inputPrev[i] == 0 && inputCurr[i] == 1)
            derivative[i] = 1;
        else if (inputPrev[i] == 1 && inputCurr[i] == 0)
            derivative[i] = -1;
    }
}
