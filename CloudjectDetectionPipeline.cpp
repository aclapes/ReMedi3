//
//  CloudjectDetectionPipeline.cpp
//  remedi3
//
//  Created by Albert Clapés on 06/04/15.
//
//

#include "CloudjectDetectionPipeline.h"
#include "constants.h"
#include "statistics.h"
#include <pcl/point_types.h>

CloudjectDetectionPipeline::CloudjectDetectionPipeline() :
    m_MultiviewStrategy(DETECT_MONOCULAR),
    m_InteractionThresh(3),
    m_ValPerf(0)
{
    
}

CloudjectDetectionPipeline::CloudjectDetectionPipeline(const CloudjectDetectionPipeline& rhs)
{
    *this = rhs;
}

CloudjectDetectionPipeline& CloudjectDetectionPipeline::operator=(const CloudjectDetectionPipeline& rhs)
{
    if (this != &rhs)
    {
        m_Sequences = rhs.m_Sequences;
        m_Categories = rhs.m_Categories;
        
        m_pRegisterer = rhs.m_pRegisterer;
        m_pTableModeler = rhs.m_pTableModeler;
        m_ClassificationPipeline = rhs.m_ClassificationPipeline;

        m_Gt = rhs.m_Gt;
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
    }
    return *this;
}

void CloudjectDetectionPipeline::setInputSequences(const std::vector<Sequence<ColorDepthFrame>::Ptr>& sequences)
{
    m_Sequences = sequences;
}

void CloudjectDetectionPipeline::setCategories(const std::vector<const char*>& categories)
{
    m_Categories = categories;
}

void CloudjectDetectionPipeline::setInteractiveRegisterer(InteractiveRegisterer::Ptr ir)
{
    m_pRegisterer = ir;
}

void CloudjectDetectionPipeline::setTableModeler(TableModeler::Ptr tm)
{
    m_pTableModeler = tm;
}

void CloudjectDetectionPipeline::setDetectionGroundtruth(const Groundtruth& gt)
{
    m_Gt = gt;
}

void CloudjectDetectionPipeline::setCorrespondenceCriterion(int criterion)
{
    m_CorrespCriterion = criterion;
}

void CloudjectDetectionPipeline::setDetectionCorrespondenceThresh(float thresh)
{
    m_DetectionCorrespThresh = thresh;
}

void CloudjectDetectionPipeline::setInteractionThresh(float thresh)
{
    m_InteractionThresh = thresh;
}

void CloudjectDetectionPipeline::setMultiviewStrategy(int strategy)
{
    m_MultiviewStrategy = strategy;
}

void CloudjectDetectionPipeline::setMultiviewLateFusionStrategy(int strategy)
{
    m_MultiviewLateFusionStrategy = strategy;
}

void CloudjectDetectionPipeline::setMultiviewCorrespondenceThresh(float thresh)
{
    m_MultiviewCorrespThresh = thresh;
}

void CloudjectDetectionPipeline::setLeafSize(Eigen::Vector3f leafSize)
{
    m_LeafSize = leafSize;
}

void CloudjectDetectionPipeline::setMultiviewLateFusionNormalization(std::vector<std::vector<float> > scalings)
{
    m_LateFusionScalings = scalings;
}

void CloudjectDetectionPipeline::setClassificationPipeline(CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr pipeline)
{
    m_ClassificationPipeline = pipeline;
}

void CloudjectDetectionPipeline::setValidationParameters(std::vector<float> correspCriteria,
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

void CloudjectDetectionPipeline::validate()
{
    std::vector<std::vector<float> > combinations;
    expandParameters<float>(m_ValParams, combinations);
    
    float valPerf = 0.f;
    int valIdx = 0;
    for (int i = 0; i < combinations.size(); i++)
    {
        std::cout << cv::Mat(combinations[i]) << std::endl;
        setValidationParametersCombination(combinations[i]);
        
        detect();

        float fscore = .0f;
        for (int v = 0; v < m_DetectionResults.size(); v++)
        {
            DetectionResult result;
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

float CloudjectDetectionPipeline::getValidationPerformance()
{
    return m_ValPerf;
}

void CloudjectDetectionPipeline::detect()
{
    if (m_MultiviewStrategy == DETECT_MONOCULAR)
        detectMonocular();
    else if (m_MultiviewStrategy == DETECT_MULTIVIEW)
        detectMultiview();
}

void CloudjectDetectionPipeline::save(std::string filename, std::string extension)
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
    
    fs << "numViews" << ((int) m_LateFusionScalings.size());
    for (int v = 0; v < m_LateFusionScalings.size(); v++)
        fs << ("lateFusionScalings-" + boost::lexical_cast<std::string>(v)) << m_LateFusionScalings[v];
    
    fs << "valPerf" << m_ValPerf;
    
    fs.release();
}

bool CloudjectDetectionPipeline::load(std::string filename, std::string extension)
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
    
    fs << "MultiviewStrategy" << m_MultiviewStrategy;
    fs << "multiviewLateFusionStrategy" << m_MultiviewLateFusionStrategy;
    
    float leafSizeX, leafSizeY, leafSizeZ;
    fs["leafSizeX"] >> leafSizeX;
    fs["leafSizeY"] >> leafSizeY;
    fs["leafSizeZ"] >> leafSizeZ;
    m_LeafSize = Eigen::Vector3f(leafSizeX,leafSizeY,leafSizeZ);
    
    int V;
    fs["numViews"] >> V;
    m_LateFusionScalings.resize(V);
    for (int v = 0; v < V; v++)
        fs["lateFusionScalings-" + boost::lexical_cast<std::string>(v)] >> m_LateFusionScalings[v];
    
    fs["valPerf"] >> m_ValPerf;

    fs.release();
    
    return true;
}

std::vector<std::vector<DetectionResult> > CloudjectDetectionPipeline::getDetectionResults()
{
    return m_DetectionResults; // one per view
}

//
// Private methods
//

void CloudjectDetectionPipeline::setValidationParametersCombination(std::vector<float> combination)
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

void createVisualizationSetup(int V, InteractiveRegisterer::Ptr pRegisterer, pcl::visualization::PCLVisualizer& viz, std::vector<int>& vp)
{
    vp.resize(V + 1);
    for (int v = 0; v < (V + 1); v++)
    {
        viz.createViewPort(v*(1.f/(V+1)), 0, (v+1)*(1.f/(V+1)), 1, vp[v]);
        viz.setBackgroundColor (1, 1, 1, vp[v]);
        //viz.addCoordinateSystem(0.1, 0, 0, 0, "cs" + boost::lexical_cast<string>(v), vp[v]);
        pRegisterer->setDefaultCamera(viz, vp[v]);
    }
}

void visualizeMonocular(pcl::visualization::PCLVisualizer::Ptr pViz, std::vector<int> vp, std::vector<ColorPointCloudPtr> interactors, std::vector<std::vector<Cloudject::Ptr> > interactions, std::vector<std::vector<Cloudject::Ptr> > actors)
{
    // Count the number of views (viewports - 1)
    int V = vp.size() - 1;
    
    for (int v = 0; v < V; v++)
    {
        pViz->removeAllPointClouds(vp[v]);
        pViz->removeAllShapes(vp[v]);
        
        for (int i = 0; i < interactors.size(); i++)
            pViz->addPointCloud(interactors[i], "interactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), vp[v] );
        
        for (int i = 0; i < interactions[v].size(); i++)
        {
            pViz->addPointCloud(interactions[v][i]->getRegisteredCloud(), "nactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), vp[v] );
            pViz->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 0, "nactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), vp[v]);
        }
        
        for (int i = 0; i < actors[v].size(); i++)
        {
            pViz->addPointCloud(actors[v][i]->getRegisteredCloud(), "actor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), vp[v] );
            pViz->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, g_Colors[i%14][0], g_Colors[i%14][1], g_Colors[i%14][2], "actor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), vp[v]);
        }
    }
    
    pViz->spinOnce(50);
}

void CloudjectDetectionPipeline::detectMonocular()
{
    m_DetectionResults.clear();
    m_DetectionResults.resize(m_Sequences[0]->getNumOfViews(), std::vector<DetectionResult>(m_Categories.size()));
    
    for (int s = 0; s < m_Sequences.size(); s++)
    {
        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
        + m_Sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        
        int V = m_Sequences[s]->getNumOfViews();
        
#ifdef DEBUG_VISUALIZE_DETECTIONS
        pcl::visualization::PCLVisualizer::Ptr pViz (new pcl::visualization::PCLVisualizer);
        std::vector<int> vp;
        
        createVisualizationSetup(V, m_pRegisterer, *pViz, vp);
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
                
                findActors(cloudjectsF, interactors, nonInteractedActors[v], m_LeafSize, interactedActors[v]);
            }
            
            for (int v = 0; v < V; v++)
            {
                std::vector<std::vector<float> > margins (nonInteractedActors[v].size());
                for (int i = 0; i < nonInteractedActors[v].size(); i++)
                {
                    std::vector<int> predictions;
                    std::vector<float> distsToMargin;
                    predictions = m_ClassificationPipeline->predict(nonInteractedActors[v][i], distsToMargin);
                    nonInteractedActors[v][i]->setPredictions(distsToMargin);
                    
                    for (int j = 0; j < predictions.size(); j++)
                        if (predictions[j] > 0)
                            nonInteractedActors[v][i]->addRegionLabel(m_Categories[j]);
                    
                    margins[i] = distsToMargin;
                }
                
                std::vector<DetectionResult> objectsResult;
                evaluateFrame2(frames[v],
                               m_Gt.at(m_Sequences[s]->getName()).at(m_Sequences[s]->getViewName(v)).at(fids[v]),
                               nonInteractedActors[v],
                               m_LeafSize,
                               objectsResult);
                
                for (int i = 0; i < objectsResult.size(); i++)
                    m_DetectionResults[v][i] += objectsResult[i];
                
#ifdef DEBUG_VISUALIZE_DETECTIONS
                DetectionResult result;
                for (int i = 0; i < objectsResult.size(); i++)
                    result += objectsResult[i];
                
                std::cout << result.tp << "\t" << result.fp << "\t" << result.fn;
                std::cout << ((v < m_Sequences[s]->getNumOfViews() - 1) ?  "\t" : ";\n");
#endif //DEBUG_VISUALIZE_DETECTIONS
            }
            
#ifdef DEBUG_VISUALIZE_DETECTIONS
            visualizeMonocular(pViz, vp, interactors, interactedActors, nonInteractedActors);
#endif //DEBUG_VISUALIZE_DETECTIONS
        }
#ifdef DEBUG_VISUALIZE_DETECTIONS
        std::cout << "];" << std::endl;
#endif //DEBUG_VISUALIZE_DETECTIONS
    }
}

void visualizeMultiview(pcl::visualization::PCLVisualizer::Ptr pViz, std::vector<int> vp, std::vector<ColorPointCloudPtr> interactors, std::vector<std::vector<Cloudject::Ptr> > interactions, std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences)
{
    // Count the number of views (viewports - 1)
    int V = vp.size() - 1;
    
    for (int v = 0; v < V; v++)
    {
        pViz->removeAllPointClouds(vp[v]);
        pViz->removeAllShapes(vp[v]);
    }
    
    for (int v = 0; v < V; v++)
    {
        for (int i = 0; i < interactors.size(); i++)
            pViz->addPointCloud(interactors[i], "interactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), vp[v] );
        
        for (int i = 0; i < interactions[v].size(); i++)
        {
            pViz->addPointCloud(interactions[v][i]->getRegisteredCloud(), "nactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), vp[v] );
            pViz->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 0, "nactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), vp[v]);
        }
    }
    
    for (int i = 0; i < correspondences.size(); i++)
    {
        std::vector<VoxelGridPtr> grids (correspondences[i].size());
        for (int j = 0; j < correspondences[i].size(); j++)
        {
            int v = correspondences[i][j].first;

            // Visualize in color and boxed in the viewpoirt
            
            pViz->addPointCloud(correspondences[i][j].second->getRegisteredCloud(), "actor" + boost::lexical_cast<string>(i) + "-" + boost::lexical_cast<string>(j), vp[v] );
            
            grids[j] = correspondences[i][j].second->getRegisteredGrid();
            Eigen::Vector3f leafSize = grids[j]->getLeafSize();
            Eigen::Vector3i min = grids[j]->getMinBoxCoordinates();
            Eigen::Vector3i max = grids[j]->getMaxBoxCoordinates();
            
            pViz->addCube(min.x() * leafSize.x(), max.x() * leafSize.x(), min.y() * leafSize.y(), max.y() * leafSize.y(), min.z() * leafSize.z(), max.z() * leafSize.z(), g_Colors[i%14][0], g_Colors[i%14][1], g_Colors[i%14][2], "cube" + boost::lexical_cast<string>(i) + "-" + boost::lexical_cast<string>(j), vp[v]);
        
            // Visualize shaded and with the grid boxes in the fusion viewport (to notice the overlap)
            for (int x = min.x(); x < max.x(); x++) for (int y = min.y(); y < max.y(); y++) for (int z = min.z(); z < max.z(); z++)
            {
                if (grids[j]->getCentroidIndexAt(Eigen::Vector3i(x,y,z)) > 0)
                    pViz->addCube(x * leafSize.x(), (x+1) * leafSize.x(), y * leafSize.y(), (y+1) * leafSize.y(), z * leafSize.z(), (z+1) * leafSize.z(), g_Colors[v%14][0], g_Colors[v%14][1], g_Colors[v%14][2], "cube" + boost::lexical_cast<string>(i) + "-" + boost::lexical_cast<string>(j) + "-" + boost::lexical_cast<string>(x) + "-" + boost::lexical_cast<string>(y) + "-" + boost::lexical_cast<string>(z), vp[V]);
            }
        }
        
        Eigen::Vector3f leafSize = grids[0]->getLeafSize();
        
        for (int j1 = 0; j1 < correspondences[i].size(); j1++)
        {
            for (int j2 = j1 + 1; j2 < correspondences[i].size(); j2++)
            {
                Eigen::Vector3f leafSize = grids[j1]->getLeafSize();
                Eigen::Vector3f leafSizeAux = grids[j2]->getLeafSize();
                // Check they are equal
                
                Eigen::Vector3i min1 = grids[j1]->getMinBoxCoordinates();
                Eigen::Vector3i min2 = grids[j2]->getMinBoxCoordinates();
                Eigen::Vector3i max1 = grids[j1]->getMaxBoxCoordinates();
                Eigen::Vector3i max2 = grids[j2]->getMaxBoxCoordinates();
                
                Eigen::Vector3i minI (min1.x() > min2.x() ? min1.x() : min2.x(),
                                      min1.y() > min2.y() ? min1.y() : min2.y(),
                                      min1.z() > min2.z() ? min1.z() : min2.z());
                Eigen::Vector3i maxI (max1.x() < max2.x() ? max1.x() : max2.x(),
                                      max1.y() < max2.y() ? max1.y() : max2.y(),
                                      max1.z() < max2.z() ? max1.z() : max2.z());
                
                for (int x = minI.x(); x < maxI.x(); x++) for (int y = minI.y(); y < maxI.y(); y++) for (int z = minI.z(); z < maxI.z(); z++)
                {
                    int idx1 = grids[j1]->getCentroidIndexAt(Eigen::Vector3i(x,y,z));
                    int idx2 = grids[j2]->getCentroidIndexAt(Eigen::Vector3i(x,y,z));
                    if (idx1 != 1 && idx2 != -1)
                    {
                        std::string name = "ovl_sphere_" + boost::lexical_cast<string>(i) + "-" + boost::lexical_cast<string>(x) + "-" + boost::lexical_cast<string>(y) + "-" + boost::lexical_cast<string>(z);
                        pViz->addSphere(pcl::PointXYZ(x*leafSize.x()+(leafSize.x()/2.f),y*leafSize.y()+(leafSize.y()/2.f),z*leafSize.z()+(leafSize.z()/2.f)), leafSize.x()/2.f, 0, 0, 1, name, vp[V]);
                    }
                }
            }
        }
    }
    
    pViz->spinOnce(50);
}

void CloudjectDetectionPipeline::fuseCorrespondences(const std::vector<std::vector<std::pair<int, Cloudject::Ptr> > >& correspondences)
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
                    std::vector<float> distsToMargin = correspondences[i][j].second->getPredictions();
                    if (distsToMargin[k] < 0)
                        marginsFused[k] = -1;
                }
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
                    std::vector<float> distsToMargin = correspondences[i][j].second->getPredictions();
                    pcl::PointXYZ c = correspondences[i][j].second->getCloudCentroid();
                    // Inverse distance weighting
                    float wsq = 1.f / sqrt(pow(c.x,2) + pow(c.y,2) + pow(c.z,2));
                    // Scale by the dispersion of either positive or negative margins
                    float s = (distsToMargin[k] < 0) ? m_LateFusionScalings[k][1] : m_LateFusionScalings[k][0];
                    distToMarginFused += (wsq * (distsToMargin[k] / abs(s)));
                    W += wsq;
                }
                marginsFused[k] = distToMarginFused/W;
            }
        }
        else if (m_MultiviewLateFusionStrategy == MULTIVIEW_LFSCALE_SUMDIV)
        {
            for (int j = 0; j < correspondences[i].size(); j++)
            {
                std::vector<float> distsToMargin = correspondences[i][j].second->getPredictions();
                
                float sum = 0.f;
                for (int k = 0; k < m_Categories.size(); k++)
                    sum += pow(distsToMargin[k],2);
                
                for (int k = 0; k < distsToMargin.size(); k++)
                    distsToMargin[k] /= sum;
                
                correspondences[i][j].second->setPredictions(distsToMargin);
            }
            
            for (int k = 0; k < m_Categories.size(); k++)
            {
                float distToMarginFused = .0f;
                float W = .0f;
                for (int j = 0; j < correspondences[i].size(); j++)
                {
                    std::vector<float> distsToMargin = correspondences[i][j].second->getPredictions();
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
        
        // Set the fused margins
        for (int j = 0; j < correspondences[i].size(); j++)
        {
            correspondences[i][j].second->setPredictions(marginsFused);
            
            correspondences[i][j].second->removeAllRegionLabels();
            for (int k = 0; k < m_Categories.size(); k++)
                if (marginsFused[k] < 0) correspondences[i][j].second->addRegionLabel(m_Categories[k]);
        }
    }
}

void printCorrespondences(const std::vector<std::vector<std::pair<int,Cloudject::Ptr> > >& correspondences)
{
    for (int i = 0; i < correspondences.size(); i++)
    {
        for (int j = 0; j < correspondences[i].size(); j++)
            std::cout << cv::Mat(correspondences[i][j].second->getPredictions()) << std::endl;
        std::cout << std::endl;
    }
}

void CloudjectDetectionPipeline::detectMultiview()
{
    m_DetectionResults.clear();
//    m_DetectionResults.resize(1, std::vector<DetectionResult>(m_Categories.size()));
    m_DetectionResults.resize(m_Sequences[0]->getNumOfViews(), std::vector<DetectionResult>(m_Categories.size()));
    
    for (int s = 0; s < m_Sequences.size(); s++)
    {
        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
        + m_Sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        
        int V = m_Sequences[s]->getNumOfViews();
#ifdef DEBUG_VISUALIZE_DETECTIONS
        pcl::visualization::PCLVisualizer::Ptr pViz (new pcl::visualization::PCLVisualizer);
        std::vector<int> vp;
        
        createVisualizationSetup(V, m_pRegisterer, *pViz, vp);
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
                
                findActors(cloudjectsF, interactors, nonInteractedActors[v], m_LeafSize, interactedActors[v]);
            }
            
            // Find the correspondences among actor clouds
            std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences;
            // Using grids
            std::vector<std::vector<VoxelGridPtr> > grids;
            findCorrespondences(frames, nonInteractedActors, m_MultiviewCorrespThresh, correspondences, grids); // false (not to registrate)
            
            for (int i = 0; i < correspondences.size(); i++)
            {
                std::vector<std::vector<float> > distsToMargin (correspondences[i].size());
                std::vector<std::vector<int> > predictions (correspondences[i].size());
                
                for (int j = 0; j < correspondences[i].size(); j++)
                {
                    predictions[j] = m_ClassificationPipeline->predict(correspondences[i][j].second, distsToMargin[j]);
                    
                     // some values set now, but verwritten by the fused margins if fusion
                    correspondences[i][j].second->setPredictions(distsToMargin[j]);
                    for (int k = 0; k < m_Categories.size(); k++)
                        if (distsToMargin[j][k] < 0) correspondences[i][j].second->addRegionLabel(m_Categories[k]);
                }
            }
            
//            std::cout << "Unfused correspondences: " << std::endl;
//            printCorrespondences(correspondences);
            fuseCorrespondences(correspondences);
//            std::cout << "Fused correspondences: " << std::endl;
//            printCorrespondences(correspondences);
            
            // DEBUG: Structure changes to use the monocular evaluation function
            //            std::vector<DetectionResult> objectsResult;
            //            evaluateFrame2(frames, annotationsF, correspondences, m_LeafSize, objectsResult);
            // -----------------------------------------------------------------
            std::vector<std::map<std::string,std::map<std::string,GroundtruthRegion> > > annotationsF (V);
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
                annotationsF[v] = m_Gt.at(m_Sequences[s]->getName()).at(m_Sequences[s]->getViewName(v)).at(fids[v]);
            
            std::vector<std::vector<Cloudject::Ptr> > detections (m_Sequences[s]->getNumOfViews());
            for (int i = 0; i < correspondences.size(); i++) for (int j = 0; j < correspondences[i].size(); j++)
            {
                int v = correspondences[i][j].first;
                detections[v].push_back(correspondences[i][j].second);
            }
            // -----------------------------------------------------------------
            
            
            // DEBUG: Structure changes to use the monocular evaluation function
            //            for (int i = 0; i < objectsResult.size(); i++)
            //                m_DetectionResults[0][i] += objectsResult[i];
            // -----------------------------------------------------------------
#ifdef DEBUG_VISUALIZE_DETECTIONS
            std::vector<DetectionResult> results (m_Sequences[s]->getNumOfViews()); // for printing purpose
#endif //DEBUG_VISUALIZE_DETECTIONS
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
            {
                std::vector<DetectionResult> objectsResult;
                evaluateFrame2(frames[v], annotationsF[v], detections[v], m_LeafSize, objectsResult);
                for (int i = 0; i < objectsResult.size(); i++)
                {
                    m_DetectionResults[v][i] += objectsResult[i];
#ifdef DEBUG_VISUALIZE_DETECTIONS
                    results[v] += objectsResult[i];
#endif //DEBUG_VISUALIZE_DETECTIONS
                }
            }
            // -----------------------------------------------------------------
            
#ifdef DEBUG_VISUALIZE_DETECTIONS
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
                std::cout << results[v].tp << "\t" << results[v].fp << "\t" << results[v].fn << (v < m_DetectionResults.size() - 1 ? ",\t" : ";\n");
//            visualizeMultiview(pViz, vp, interactors, interactedActors, correspondences);
#endif //DEBUG_VISUALIZE_DETECTIONS
        }
#ifdef DEBUG_VISUALIZE_DETECTIONS
        std::cout << "];" << std::endl;
#endif //DEBUG_VISUALIZE_DETECTIONS
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

void CloudjectDetectionPipeline::findActors(std::vector<Cloudject::Ptr> candidates, std::vector<ColorPointCloudPtr> interactors, std::vector<Cloudject::Ptr>& actors, Eigen::Vector3f leafSize, std::vector<Cloudject::Ptr>& interactedActors)
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

void CloudjectDetectionPipeline::findInteractions(std::vector<ColorPointCloudPtr> candidates, std::vector<ColorPointCloudPtr> interactors, std::vector<bool>& mask, Eigen::Vector3f leafSize)
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
        biggestEuclideanCluster(pInteractorFilt, 250, 25000, leafSize.x(), *pMainInteractorFilt);
        
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

void CloudjectDetectionPipeline::getDetectionPositions(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, bool bRegistrate, std::vector<std::vector<PointT> >& positions)
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

void CloudjectDetectionPipeline::getDetectionGrids(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, bool bRegistrate, std::vector<std::vector<VoxelGridPtr> >& grids)
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

bool CloudjectDetectionPipeline::isInteractive(ColorPointCloudPtr tabletopRegionCluster, ColorPointCloudPtr interactionCloud, float tol)
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

void CloudjectDetectionPipeline::findCorrespondences(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, float tol, std::vector<std::vector<std::pair<int,Cloudject::Ptr> > >& correspondences, std::vector<std::vector<PointT> >& positions)
{
    correspondences.clear();
    
    // Correspondences found using the positions of the clouds' centroids
    //    std::vector<std::vector<PointT> > positions;
    getDetectionPositions(frames, detections, true, positions); // registrate is "true"
    
    std::vector<std::vector<std::pair<std::pair<int,int>,PointT> > > _correspondences;
    pclx::findCorrespondences(positions, tol, _correspondences);

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

void CloudjectDetectionPipeline::findCorrespondences(std::vector<ColorDepthFrame::Ptr> frames, std::vector<std::vector<Cloudject::Ptr> > detections, float tol, std::vector<std::vector<std::pair<int,Cloudject::Ptr> > >& correspondences, std::vector<std::vector<VoxelGridPtr> >& grids)
{
    correspondences.clear();
    
    // Correspondences found using the positions of the clouds' centroids
    //    std::vector<std::vector<PointT> > positions;
    getDetectionGrids(frames, detections, true, grids); // registrate is "true"
    
    std::vector<std::vector<std::pair<std::pair<int,int>,VoxelGridPtr> > > _correspondences;
    
    if (m_CorrespCriterion == CloudjectDetectionPipeline::CORRESP_INC)
        pclx::findCorrespondencesBasedOnInclusion(grids, tol, _correspondences);
    else if (m_CorrespCriterion == CloudjectDetectionPipeline::CORRESP_OVL)
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

// Evaluate detection performance in a frame (multiple views)
void CloudjectDetectionPipeline::evaluateFrame(const vector<std::map<std::string,std::map<std::string,GroundtruthRegion> > >& gt, const vector<std::map<std::string,std::map<std::string,pcl::PointXYZ> > >& gtCentroids, const std::vector<std::vector<std::pair<int, Cloudject::Ptr> > >& correspondences, DetectionResult& result)
{
    result.tp = result.fp = result.fn = 0;
    
    // Auxiliary structure to compute FN (after result.result.tp and FP)
    std::map<std::string,int> matches;
    
    std::map<std::string,std::map<std::string,GroundtruthRegion> >::const_iterator itr;
    std::map<std::string,GroundtruthRegion>::const_iterator jtr;
    
    for (int v = 0; v < gt.size(); v++)
        for (itr = gt[v].begin(); itr != gt[v].end(); ++itr)
            if (itr->first.compare("arms") != 0 && itr->first.compare("others") != 0)
                matches[itr->first] = 0;
    
    // Compute TP and FP
    for (int i = 0; i < correspondences.size(); i++)
    {
        pcl::PointXYZ p = correspondences[i][0].second->getCloudCentroid();
        
        const std::set<std::string> labels = correspondences[i][0].second->getRegionLabels();
        std::set<std::string>::iterator it;
        
        int tpAux = 0;
        for (it = labels.begin(); it != labels.end(); ++it)
        {
            for (int v = 0; v < gt.size(); v++)
            {
                if (gt[v].count(*it) > 0)
                {
                    std::map<std::string,GroundtruthRegion> catgt = gt[v].at(*it);
                    std::map<std::string,GroundtruthRegion>::iterator jt;
                    for (jt = catgt.begin(); jt != catgt.end(); ++jt)
                    {
                        pcl::PointXYZ q = gtCentroids[v].at(*it).at(jt->first);
                        float d = sqrt(pow(p.x-q.x,2)+pow(p.y-q.y,2)+pow(p.z-q.z,2));
                        // is the detected in the correct spot?
                        if (d <= 0.15)
                        {
                            //                            std::cout << *it << " found in " << boost::lexical_cast<string>(v) << std::endl;
                            tpAux++;
                            matches[*it] ++;
                        }
                    }
                }
            }
        }
        
        result.tp += tpAux;
        result.fp += labels.size() - tpAux;
    }
    
    //  Compute FN (annotations without at least one match)
    std::map<std::string,int>::iterator it;
    for (it = matches.begin(); it != matches.end(); ++it)
        if (matches[it->first] == 0) result.fn++; // match, not even once
}

// Evaluate detection performance in a frame
void CloudjectDetectionPipeline::evaluateFrame(const std::map<std::string,std::map<std::string,GroundtruthRegion> >& gt, const std::vector<ForegroundRegion>& dt, DetectionResult& result)
{
    // Iterate over the dt vector to determine TP and FP
    result.tp = result.fp = result.fn = 0;
    
    std::map<std::string,int> matches;
    
    std::map<std::string,std::map<std::string,GroundtruthRegion> >::const_iterator itr;
    
    // Auxiliary structure to compute FN (after TP and FP)
    for (itr = gt.begin(); itr != gt.end(); ++itr)
        if (itr->first.compare("arms") != 0 && itr->first.compare("others") != 0)
            matches[itr->first] = false;
    
    // Compute TP and FP first
    for (int i = 0; i < dt.size(); i++)
    {
        const std::set<std::string> labels = dt[i].getLabels();
        
        std::set<std::string>::iterator it;
        for (it = labels.begin(); it != labels.end(); ++it)
        {
            // There is not even groundtruth object with such label
            if (gt.count(*it) == 0)
            {
                result.fp++;
            }
            else // and if there is..
            {
                std::map<std::string,GroundtruthRegion> catgt = gt.at(*it);
                std::map<std::string,GroundtruthRegion>::iterator jt;
                for (jt = catgt.begin(); jt != catgt.end(); ++jt)
                {
                    // is the detected in the correct spot?
                    if (cvx::rectangleOverlap(dt[i].getRect(),jt->second.getRect()) > 0)
                    {
                        result.tp++;
                        // Count matches to this annotation
                        matches[*it] ++;
                    }
                    else
                    {
                        result.fp++;
                    }
                }
            }
        }
    }
    
    //  Compute FN (annotations without at least one match)
    //    for (it = gt.begin(); it != gt.end(); ++it)
    //        if (it->first.compare("arms") != 0 && it->first.compare("others") != 0)
    //            for (jt = it->second.begin(); jt != it->second.end(); ++jt)
    //                if (!matches[it->first][jt->first]) fn++; // match, not even once
    std::map<std::string,int>::iterator it;
    for (it= matches.begin(); it != matches.end(); ++it)
        if (matches[it->first] == 0) result.fn++; // match, not even once
}

// Evaluate detection performance in a frame per object (multiple views)
void CloudjectDetectionPipeline::evaluateFrame2(const std::vector<ColorDepthFrame::Ptr>& frames, const std::vector<std::map<std::string,std::map<std::string,GroundtruthRegion> > >& gt, std::vector<std::vector<std::pair<int, Cloudject::Ptr> > >& correspondences, Eigen::Vector3f leafSize, std::vector<DetectionResult>& result)
{
    result.clear();
    result.resize(m_Categories.size());

    std::vector<float> maxmargins (m_Categories.size(), 0);
    std::vector<int> indices (m_Categories.size(), -1);
    
    std::vector<std::map<std::string,std::map<std::string,VoxelGridPtr> > > gtgrids (gt.size());
    for (int v = 0; v < gt.size(); v++)
        precomputeGrids(frames[v], gt[v], leafSize, gtgrids[v], true);
    
    for (int i = 0; i < correspondences.size(); i++)
    {
        for (int k = 0; k < m_Categories.size(); k++)
        {
            std::vector<float> margins = correspondences[i][0].second->getPredictions(); // 0-th indexed, since they are all the same
            
            // A detectoin gives a positive for m_Categories[k] category
            if (margins[k] < maxmargins[k]) // worth checking?
            {
                bool bMatched = false;
                // #checks = detection views x subannotations
                for (int j = 0; (j < correspondences[i].size()) && !bMatched; j++)
                {
                    VoxelGridPtr pDetectionGrid = correspondences[i][j].second->getRegisteredGrid();
                    
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
                
                if (bMatched)
                {
                    result[k].tp++;
                    maxmargins[k] = margins[k];
                    
                    if (indices[k] >= 0)
                    {
                        result[k].fp++;
                        result[k].tp--;
                    }
                    indices[k] = i;
                }
            }
            else if (margins[k] < 0)
            {
                result[k].fp++;
            }
        }
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

// Evaluate detection performance in a frame per object
void CloudjectDetectionPipeline::evaluateFrame2(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& gt, std::vector<Cloudject::Ptr>& detections, Eigen::Vector3f leafSize, std::vector<DetectionResult>& result)
{
    result.resize(m_Categories.size());
    
    std::vector<float> maxmargins (m_Categories.size(), 0);
    std::vector<int> indices (m_Categories.size(), -1);
    
    std::map<std::string,std::map<std::string,VoxelGridPtr> > gtgrids;
    precomputeGrids(frame, gt, leafSize, gtgrids, true);
    
    for (int i = 0; i < detections.size(); i++)
    {
        for (int k = 0; k < m_Categories.size(); k++)
        {
            std::vector<float> margins = detections[i]->getPredictions();
            
            // A detectoin gives a positive for m_Categories[k] category
            if (margins[k] < maxmargins[k]) // worth checking?
            {
                bool bMatched = false;
                // #checks = subannotations
                VoxelGridPtr pDetectionGrid = detections[i]->getRegisteredGrid();

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
                
                
                if (bMatched)
                {
                    result[k].tp++;
                    maxmargins[k] = margins[k];
                    
                    if (indices[k] >= 0)
                    {
                        result[k].fp++;
                        result[k].tp--;
                    }
                    indices[k] = i;
                }
            }
            else if (margins[k] < 0)
            {
                result[k].fp++;
            }
        }
    }
    
    // If one of the categoryes undetected, but was annotated in one of the views' gt
    for (int k = 0; k < indices.size(); k++)
    {
        if (indices[k] < 0 && gt.count(m_Categories[k]) > 0) // annotated?
            result[k].fn++;
    }
}

template<typename RegionT>
VoxelGridPtr CloudjectDetectionPipeline::computeGridFromRegion(ColorDepthFrame::Ptr frame, RegionT region, Eigen::Vector3f leafSize, bool bRegistrate)
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
    pclx::voxelize((!bRegistrate ? pCloud : pRegCloud), dummy, *pGrid, leafSize);
    
    return pGrid;
}

template<typename RegionT>
pcl::PointXYZ CloudjectDetectionPipeline::computeCentroidFromRegion(ColorDepthFrame::Ptr frame, RegionT region)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    region.allocateMask();
    int pos[2] = {region.getRect().x, region.getRect().y};
    unsigned short range[2] = {MIN_DEPTH,MAX_DEPTH};
    MatToColoredPointCloud(frame->getDepth(), frame->getColor(), region.getMask(), pos, range, *pCloud);
    region.releaseMask();
    
    pcl::PointXYZ centroid = computeCentroid(*pCloud);
    
    return centroid;
}

void CloudjectDetectionPipeline::precomputeGrids(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& annotations, Eigen::Vector3f leafSize, std::map<std::string,std::map<std::string,VoxelGridPtr> >& grids, bool bRegistrate)
{
    std::map<std::string,std::map<std::string,GroundtruthRegion> >::const_iterator it;
    std::map<std::string,GroundtruthRegion>::const_iterator jt;
    for (it = annotations.begin(); it != annotations.end(); ++it)
        for (jt = it->second.begin(); jt != it->second.end(); ++jt)
            grids[it->first][jt->first] = computeGridFromRegion(frame, annotations.at(it->first).at(jt->first), leafSize, bRegistrate);
}

void CloudjectDetectionPipeline::precomputeGrids(ColorDepthFrame::Ptr frame, const std::vector<ForegroundRegion>& detections, Eigen::Vector3f leafSize, std::vector<VoxelGridPtr>& grids, bool bRegistrate)
{
    grids.resize(detections.size());
    for (int i = 0; i < detections.size(); ++i)
        grids[i] = computeGridFromRegion(frame, detections[i], leafSize, bRegistrate);
}

template<typename RegionT>
pclx::Rectangle3D CloudjectDetectionPipeline::computeRectangleFromRegion(ColorDepthFrame::Ptr frame, RegionT region)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    region.allocateMask();
    int pos[2] = {region.getRect().x, region.getRect().y};
    unsigned short range[2] = {MIN_DEPTH,MAX_DEPTH};
    MatToColoredPointCloud(frame->getDepth(), frame->getColor(), region.getMask(), pos, range, *pCloud);
    region.releaseMask();
    
    pclx::Rectangle3D rectangle;
    computeRectangle3D(*pCloud,rectangle.min,rectangle.max);
    
    return rectangle;
}

void CloudjectDetectionPipeline::precomputeRectangles3D(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& annotations, std::map<std::string,std::map<std::string,pclx::Rectangle3D> >& rectangles)
{
    std::map<std::string,std::map<std::string,GroundtruthRegion> >::const_iterator it;
    std::map<std::string,GroundtruthRegion>::const_iterator jt;
    for (it = annotations.begin(); it != annotations.end(); ++it)
        for (jt = it->second.begin(); jt != it->second.end(); ++jt)
            rectangles[it->first][jt->first] = computeRectangleFromRegion(frame, annotations.at(it->first).at(jt->first));
}

void CloudjectDetectionPipeline::precomputeRectangles3D(ColorDepthFrame::Ptr frame, const std::vector<ForegroundRegion>& detections, std::vector<pclx::Rectangle3D>& rectangles)
{
    rectangles.resize(detections.size());
    for (int i = 0; i < detections.size(); ++i)
        rectangles[i] = computeRectangleFromRegion(frame, detections[i]);
}

void CloudjectDetectionPipeline::precomputeCentroids(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& annotations, std::map<std::string,std::map<std::string,pcl::PointXYZ> >& centroids)
{
    std::map<std::string,std::map<std::string,GroundtruthRegion> >::const_iterator it;
    std::map<std::string,GroundtruthRegion>::const_iterator jt;
    for (it = annotations.begin(); it != annotations.end(); ++it)
        for (jt = it->second.begin(); jt != it->second.end(); ++jt)
            centroids[it->first][jt->first] = computeCentroidFromRegion(frame, annotations.at(it->first).at(jt->first));
}

void CloudjectDetectionPipeline::precomputeCentroids(ColorDepthFrame::Ptr frame, const std::vector<ForegroundRegion>& detections, std::vector<pcl::PointXYZ>& centroids)
{
    centroids.resize(detections.size());
    for (int i = 0; i < detections.size(); ++i)
        centroids[i] = computeCentroidFromRegion(frame, detections[i]);
}
