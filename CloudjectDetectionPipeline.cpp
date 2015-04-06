//
//  CloudjectDetectionPipeline.cpp
//  remedi3
//
//  Created by Albert Clapés on 06/04/15.
//
//

#include "CloudjectDetectionPipeline.h"

#include <pcl/point_types.h>

CloudjectDetectionPipeline::CloudjectDetectionPipeline() :
    m_MultiviewDetectionStrategy(0),
    m_MultiviewActorCorrespThresh(3),
    m_InteractionThresh(0.075)
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
        
        m_Gt = rhs.m_Gt;
        
        m_MultiviewDetectionStrategy = rhs.m_MultiviewDetectionStrategy;
        m_MultiviewActorCorrespThresh = rhs.m_MultiviewActorCorrespThresh;
        m_InteractionThresh = rhs.m_InteractionThresh;
        
        m_LateFusionScalings = rhs.m_LateFusionScalings;
        
        m_ClassificationPipeline = rhs.m_ClassificationPipeline;
        
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

void CloudjectDetectionPipeline::setMultiviewDetectionStrategy(int strategy)
{
    m_MultiviewDetectionStrategy = strategy;
}

void CloudjectDetectionPipeline::setMultiviewLateFusionNormalization(std::vector<std::vector<float> > scalings)
{
    m_LateFusionScalings = scalings;
}

void CloudjectDetectionPipeline::setMultiviewActorCorrespondenceThresh(float thresh)
{
    m_MultiviewActorCorrespThresh = thresh;
}

void CloudjectDetectionPipeline::setInteractionThresh(float thresh)
{
    m_InteractionThresh = thresh;
}

void CloudjectDetectionPipeline::setClassificationPipeline(CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr pipeline)
{
    m_ClassificationPipeline = pipeline;
}

void CloudjectDetectionPipeline::detect()
{
    if (m_MultiviewDetectionStrategy == DETECT_MONOCULAR)
        detectMonocular();
    else if (m_MultiviewDetectionStrategy == DETECT_MULTIVIEW)
        detectMultiview();
}

std::vector<DetectionResult> CloudjectDetectionPipeline::getDetectionResults()
{
    return m_DetectionResults; // one per view
}

//
// Private methods
//

void CloudjectDetectionPipeline::detectMonocular()
{
    m_DetectionResults.resize(m_Sequences[0]->getNumOfViews());
    
    for (int s = 0; s < m_Sequences.size(); s++)
    {
        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
        + m_Sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        
        int V = m_Sequences[s]->getNumOfViews();
        
#ifdef DO_VISUALIZE_DETECTIONS
        pcl::visualization::PCLVisualizer::Ptr pVis (new pcl::visualization::PCLVisualizer);
        std::vector<int> vp (V);
        for (int v = 0; v < V; v++)
        {void setInteractiveRegisterer(InteractiveRegisterer::Ptr ir);
            void setTableModeler(TableModeler::Ptr tm);
            pVis->createViewPort(v*(1.f/V), 0, (v+1)*(1.f/V), 1, vp[v]);
            pVis->setBackgroundColor (1, 1, 1, vp[v]);
            m_pRegisterer->setDefaultCamera(pVis, vp[v]);
        }
#endif
        std::cout << "[" << std::endl;
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
                    cloudjectsF[i]->setRegistrationTransformation(frames[v]->getRegistrationTransformation());
                
                findActors(cloudjectsF, interactors, nonInteractedActors[v], 0.02, interactedActors[v]);
            }
            
            for (int v = 0; v < V; v++)
            {
                std::vector<ForegroundRegion> detections (nonInteractedActors[v].size());
                for (int i = 0; i < nonInteractedActors[v].size(); i++)
                {
                    std::vector<int> predictions;
                    std::vector<float> distsToMargin;
                    predictions = m_ClassificationPipeline->predict(nonInteractedActors[v][i], distsToMargin);
                    
                    ForegroundRegion r = nonInteractedActors[v][i]->getRegion();
                    for (int i = 0; i < predictions.size(); i++)
                        if (predictions[i] > 0) r.addLabel(m_Categories[i]);
                    
                    detections[i] = r;
                }
                
                std::vector<DetectionResult> objResult;
                evaluateFrame2(frames[v],
                               m_Gt.at(m_Sequences[s]->getName()).at(m_Sequences[s]->getViewName(v)).at(fids[v]),
                               detections,
                               objResult);
                
                DetectionResult result;
                for (int i = 0; i < objResult.size(); i++)
                    result += objResult[i];
                
                std::cout << std::to_string(result.tp) << "\t" << std::to_string(result.fp) << "\t" << std::to_string(result.fn);
                std::cout << ((v < m_Sequences[s]->getNumOfViews() - 1) ?  "\t" : ";\n");
                
                m_DetectionResults[v] += result;
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
                
                for (int i = 0; i < nonInteractedActors[v].size(); i++)
                {
                    pVis->addPointCloud(nonInteractedActors[v][i]->getRegisteredCloud(), "actor" + std::to_string(v) + "-" + std::to_string(i), vp[v] );
                    pVis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, g_Colors[i%14][0], g_Colors[i%14][1], g_Colors[i%14][2], "actor" + std::to_string(v) + "-" + std::to_string(i), vp[v]);
                }
            }
            
            pVis->spinOnce();
#endif //DO_VISUALIZE_DETECTIONS
        }
        std::cout << "];" << std::endl;
    }
}

void CloudjectDetectionPipeline::detectMultiview()
{
    m_DetectionResults.resize(1);
    
    for (int s = 0; s < m_Sequences.size(); s++)
    {
        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
        + m_Sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
        
        int V = m_Sequences[s]->getNumOfViews();
#ifdef DO_VISUALIZE_DETECTIONS
        pcl::visualization::PCLVisualizer::Ptr pVis (new pcl::visualization::PCLVisualizer);
        
        std::vector<int> vp (V);
        for (int v = 0; v < V; v++)
        {
            pVis->createViewPort(v*(1.f/V), 0, (v+1)*(1.f/V), 1, vp[v]);
            pVis->setBackgroundColor (1, 1, 1, vp[v]);
            m_pRegisterer->setDefaultCamera(pVis, vp[v]);
        }
#endif
        std::cout << "[" << std::endl;
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
                    cloudjectsF[i]->setRegistrationTransformation(frames[v]->getRegistrationTransformation());
                
                findActors(cloudjectsF, interactors, nonInteractedActors[v], 0.02, interactedActors[v]);
            }
            
            // Find the correspondences among actor clouds
            std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences;
            std::vector<std::vector<PointT> > positions;
            findCorrespondences(frames, nonInteractedActors, OR_CORRESPONDENCE_TOLERANCE, correspondences, positions); // false (not to registrate)
            
            for (int i = 0; i < correspondences.size(); i++)
            {
                std::vector<std::vector<float> > distsToMargin (correspondences[i].size());
                std::vector<std::vector<int> > predictions (correspondences[i].size());
                
                for (int v = 0; v < correspondences[i].size(); v++)
                    predictions[v] = m_ClassificationPipeline->predict(correspondences[i][v].second, distsToMargin[v]);
                
                std::vector<int> predictionsFused (m_Categories.size());
                for (int j = 0; j < m_Categories.size(); j++)
                {
                    float distToMarginFused = .0f;
                    float W = .0f;
                    for (int v = 0; v < correspondences[i].size(); v++)
                    {
                        pcl::PointXYZ c = correspondences[i][v].second->getCloudCentroid();
                        // Inverse distance weighting
                        float wsq = 1.f / pow(sqrt(pow(c.x,2) + pow(c.y,2) + pow(c.z,2)), 2);
                        float s = (predictions[v][j] < 0) ? m_LateFusionScalings[j][0] : m_LateFusionScalings[j][1];
                        distToMarginFused += ((wsq * distsToMargin[v][j]) / s);
                        W += wsq;
                    }
                    predictionsFused[j] = ((distToMarginFused/W) < 0 ? 1 : -1);
                }
                
                for (int v = 0; v < correspondences[i].size(); v++)
                {
                    for (int j = 0; j < m_Categories.size(); j++)
                        if (predictionsFused[j] > 0) correspondences[i][v].second->addRegionLabel(m_Categories[j]);
//                        if (predictions[v][j] > 0) correspondences[i][v].second->addRegionLabel(objectsLabels[j]);
                }
            }
            
            std::vector<std::map<std::string,std::map<std::string,GroundtruthRegion> > > annotationsF (V);
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
                annotationsF[v] = m_Gt.at(m_Sequences[s]->getName()).at(m_Sequences[s]->getViewName(v)).at(fids[v]);
            
            std::vector<std::vector<DetectionResult> > results;
            evaluateFrame2(frames, annotationsF, correspondences, results);
            
            for (int v = 0; v < m_Sequences[s]->getNumOfViews(); v++)
            {
                DetectionResult result;
                for (int i = 0; i < results[v].size(); i++)
                    result += results[v][i];
                
                std::cout << std::to_string(result.tp) << "\t" << std::to_string(result.fp) << "\t" << std::to_string(result.fn) << (v < m_Sequences[s]->getNumOfViews() - 1 ? "\t" : ";\n");
            }
//            m_DetectionResults[0] += result;
            
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
        std::cout << "];" << std::endl;
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

void CloudjectDetectionPipeline::findActors(std::vector<Cloudject::Ptr> candidates, std::vector<ColorPointCloudPtr> interactors, std::vector<Cloudject::Ptr>& actors, float leafSize, std::vector<Cloudject::Ptr>& interactedActors)
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

void CloudjectDetectionPipeline::findInteractions(std::vector<ColorPointCloudPtr> candidates, std::vector<ColorPointCloudPtr> interactors, std::vector<bool>& mask, float leafSize)
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
                            //                            std::cout << *it << " found in " << std::to_string(v) << std::endl;
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
void CloudjectDetectionPipeline::evaluateFrame2(const std::vector<ColorDepthFrame::Ptr>& frames, const std::vector<std::map<std::string,std::map<std::string,GroundtruthRegion> > >& gt, std::vector<std::vector<std::pair<int, Cloudject::Ptr> > >& correspondences, std::vector<std::vector<DetectionResult> >& result)
{
    result.clear();
    result.resize(gt.size(), std::vector<DetectionResult>(m_Categories.size()));
    
    std::vector<std::map<std::string,int> > matches (correspondences.size()); // correspondences x labels
    
    std::vector<std::map<std::string,std::map<std::string,pclx::Rectangle3D> > > gtrects (gt.size());
    for (int v = 0; v < gt.size(); v++)
    {
        precomputeRectangles3D(frames[v], gt[v], gtrects[v]);
        
        for (int k = 0; k < m_Categories.size(); k++)
        {
            std::string name = m_Categories[k];
            
            // TP and FN
            if (gt[v].count(name) > 0) // any object of this kind annotated in frame?
            {
                // An object can consist of several annotations (a book in two parts because of an occlusion)
                // Try two match any of the parts with a detection
                bool bDetectedAnnotation = false;
                
                std::map<std::string,GroundtruthRegion>::const_iterator it = gt[v].at(name).begin();
                std::map<std::string,pclx::Rectangle3D>::const_iterator rt = gtrects[v].at(name).begin();
                for ( ; (it != gt[v].at(name).end()) && !bDetectedAnnotation; ++it, ++rt)
                {
                    for (int i = 0; i < correspondences.size(); i++)
                    {
                        pclx::Rectangle3D a = rt->second;
                        pclx::Rectangle3D b;
                        for (int j = 0; j < correspondences[i].size(); j++)
                        {
                            if (correspondences[i][j].first == v)
                            {
                                Cloudject::Ptr pCj = correspondences[i][j].second;
                                pCj->getCloudRectangle(b.min, b.max);
                                if (pCj->isRegionLabel(name) && pclx::rectangleInclusion(a,b) > 0.25)
                                {
                                    //if (matches[j][name] < pCj->getRegionLabels().size())
                                    //{
                                    matches[i][name]++;
                                    bDetectedAnnotation = true;
                                    //}
                                }
                            }
                        }
                    }
                }
                
                bDetectedAnnotation ? result[v][k].tp++ : result[v][k].fn++;
            }
            
            // FPs
            for (int i = 0; i < correspondences.size(); i++)
            {
                Cloudject::Ptr pCj = correspondences[i][0].second;
                if (pCj->isRegionLabel(name))
                {
                    int nonmatched = pCj->getRegionLabels().size() - matches[i][name];
                    if (nonmatched > 0)
                    {
                        result[v][k].fp += nonmatched;
                    }
                }
            }
        }
        
    }
}

// Evaluate detection performance in a frame per object
void CloudjectDetectionPipeline::evaluateFrame2(ColorDepthFrame::Ptr frame, const std::map<std::string,std::map<std::string,GroundtruthRegion> >& gt, std::vector<ForegroundRegion>& dt, std::vector<DetectionResult>& result)
{
    result.resize(m_Categories.size());
    
    // Precompute rectangles
    std::map<std::string,std::map<std::string,pclx::Rectangle3D> > gtrects;
    std::vector<pclx::Rectangle3D> dtrects;
    
    precomputeRectangles3D(frame, gt, gtrects);
    precomputeRectangles3D(frame, dt, dtrects);
    
    // First TP and FN, but this will help to compute the FPs later (keeps track of matched or TPs)
    std::vector<std::map<std::string,int> > matches (dt.size()); // detections x labels
    
    for (int k = 0; k < m_Categories.size(); k++)
    {
        std::string name = m_Categories[k];
        
        // TP and FN
        if (gt.count(name) > 0) // any object of this kind annotated in frame?
        {
            // An object can consist of several annotations (a book in two parts because of an occlusion)
            // Try two match any of the parts with a detection
            bool bDetectedAnnotation = false;
            
            std::map<std::string,GroundtruthRegion>::const_iterator it = gt.at(name).begin();
            std::map<std::string,pclx::Rectangle3D>::const_iterator rt = gtrects.at(name).begin();
            for ( ; (it != gt.at(name).end()) && !bDetectedAnnotation; ++it, ++rt)
            {
                for (int j = 0; j < dt.size(); j++)
                {
                    pclx::Rectangle3D a = rt->second;
                    pclx::Rectangle3D b = dtrects[j];
                    if (dt[j].isLabel(name) && pclx::rectangleInclusion(a,b) > 0.5)
                    {
                        if (matches[j][name] < dt[j].getLabels().size())
                        {
                            matches[j].at(name) ++;
                            bDetectedAnnotation = true;
                        }
                    }
                }
            }
            
            bDetectedAnnotation ? result[k].tp++ : result[k].fn++;
        }
        
        // FPs
        for (int i = 0; i < dt.size(); i++)
        {
            if (dt[i].isLabel(name))
            {
                int nonmatched = dt[i].getLabels().size() - matches[i][name];
                if (nonmatched > 0)
                {
                    result[k].fp += nonmatched;
                }
            }
        }
    }
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
