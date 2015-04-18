#include <iostream>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/timer.hpp>

#include "Remedi.h"
#include "constants.h"

#include "KinectReader.h"
#include "Sequence.h"
#include "ColorDepthFrame.h"

#include "statistics.h"
#include "cvxtended.h"

#include "io.h"
#include "features.h"
#include "CloudjectSVMClassificationPipeline.h"
#include "CloudjectInteractionPipeline.h"
#include "InteractionFilteringPipeline.h"

#include <pcl/console/parse.h>

using namespace std;
using namespace boost::assign;

namespace fs = boost::filesystem;

int run(ReMedi::Ptr pSys, std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> sequencesSids, int beginFold = 0, int endFold = 0)
{
// *---------------------------------------------------------------------------*
// | Perform table top segmentation
// *---------------------------------------------------------------------------*
//    
//    for (int s = 0; s < sequences.size(); s++)
//    {
//        sequences[s]->restart();
//        while (sequences[s]->hasNextFrames())
//        {
//            std::vector<ColorDepthFrame::Ptr> testFrames = sequences[s]->nextFrames();
//            pSys->getRegisterer()->setInputFrames(testFrames);
//            pSys->getRegisterer()->registrate(testFrames);
//            
//            // focus only on tabletop's region for the BS
//            std::vector<cv::Mat> tabletopMasks;
//            pSys->getTableModeler()->getTabletopMask(testFrames, tabletopMasks);
//            
//            std::vector<string> fids = sequences[s]->getFramesFilenames();
//            for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
//            {
//                std::string path = string(PARENT_PATH) + string(RESULTS_SUBDIR)
//                  + sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR)
//                  + string(TABLETOP_MASKS_DIRNAME) + sequences[s]->getViewName(v);
//                cv::imwrite(path + "/" + filenames[v] + ".png", tabletopMasks[v]);
//            }
//        }
//    }
    
// *---------------------------------------------------------------------------*
// | Background subraction and blob extraction
// *---------------------------------------------------------------------------*
//    
//    pSys->getBackgroundSubtractor()->model(); // train the background model using the bgSeq
//    
//    BoundingBoxesGenerator bbGenerator (BB_MIN_PIXELS, BB_DEPTH_THRESH); // from Oscar
//    
//    for (int s = 0; s < sequences.size(); s++)
//    {
//        std::string parentPath = string(PARENT_PATH) + string(RESULTS_SUBDIR)
//        + sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
//
//        sequences[s]->restart();
//        while (sequences[s]->hasNextFrames())
//        {
//            std::vector<ColorDepthFrame::Ptr> testFrames = sequences[s]->nextFrames();
//
//            std::vector<cv::Mat> foregroundMasks;
//            pSys->getBackgroundSubtractor()->setInputFrames(testFrames);
//            pSys->getBackgroundSubtractor()->subtract(foregroundMasks);
//            
//            std::vector<string> fids = sequences[s]->getFramesFilenames();
//            for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
//            {
//                cv::Mat tabletopMask = cv::imread(parentPath + std::string(TABLETOP_MASKS_DIRNAME)
//                    + sequences[s]->getViewName(v) + "/" + fids[v] + ".png",
//                    CV_LOAD_IMAGE_GRAYSCALE);
//                
//                // Re-using some Oscar's code in here
//                // -------------------------------------------------------------
//                // Generate foreground image (annotated pixels) and bounding boxes
//                bbData bbdata = bbGenerator.generate(foregroundMasks[v] & tabletopMask,
//                                                     testFrames[v]->getDepth());
//                // Save to disk foreground image
//                cv::imwrite(parentPath + std::string(FOREGROUND_MASKS_DIRNAME)
//                            + sequences[s]->getViewName(v) + "/" + fids[v] + ".png",
//                            bbdata.pixelsMask);
//                // Save to disk bounding boxes
//                std::ofstream ofs;
//                ofs.open(parentPath + std::string(FOREGROUND_MASKS_DIRNAME)
//                         + sequences[s]->getViewName(v) + "/" + fids[v] + ".csv");
//                for (int i = 0; i < bbdata.boundingBoxes.size(); i++)
//                {
//                    ofs << (i+1) << "," << bbdata.boundingBoxes[i].xMin << ","
//                                        << bbdata.boundingBoxes[i].yMin << ","
//                                        << bbdata.boundingBoxes[i].xMax << ","
//                                        << bbdata.boundingBoxes[i].yMax << std::endl;
//                }
//                ofs.close();
//                // -------------------------------------------------------------
//            }
//        }
//    }
    
// *---------------------------------------------------------------------------*
// | Cloudject construction and description
// *---------------------------------------------------------------------------*
    
//     Convert blobs to point clouds and store them in PCDs (w/ a file of metainfo for each)
//
//    std::cout << "Generating cloudjects ..." << std::endl;
//
//    for (int s = 0; s < sequences.size(); s++)
//    {
//        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
//        + sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
//        
//        for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
//        {
//            std::string cloudjectsPath = resultsParent + string(CLOUDJECTS_DIRNAME) + sequences[s]->getViewName(v);
//            if (fs::is_directory(cloudjectsPath))
//                fs::remove_all(cloudjectsPath);
//        }
//    }
//    
//    for (int s = 0; s < 1/*sequences.size()*/; s++)
//    {
//        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
//        + sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
//        
//        std::cout << resultsParent << std::endl;
//
//        sequences[s]->restart();
//        while (sequences[s]->hasNextFrames())
//        {
//            std::vector<ColorDepthFrame::Ptr> frames = sequences[s]->nextFrames();
//            std::vector<string> fids = sequences[s]->getFramesFilenames();
//            
//            for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
//            {
//                // Get the predictions (boxes and mask)
//                std::vector<Region> regionsF;
//                remedi::io::readPredictionRegions(resultsParent + std::string(FOREGROUND_MASKS_DIRNAME) + sequences[s]->getViewName(v), fids[v], regionsF, false);
//                
//                cv::Mat fgMaskF = cv::imread(resultsParent + string(FOREGROUND_MASKS_DIRNAME) + sequences[s]->getViewName(v) + "/" + fids[v] + ".png", CV_LOAD_IMAGE_GRAYSCALE); // mask
//                
//                // Create cloudjects
//                std::vector<Cloudject::Ptr> cloudjectsF;
//                remedi::segmentCloudjects(frames[v], fgMaskF, regionsF, cloudjectsF);
//                
//                // Save these as training cloudjects
//                std::string cloudjectsPath = resultsParent + string(CLOUDJECTS_DIRNAME) + sequences[s]->getViewName(v);
//                if (!fs::is_directory(cloudjectsPath)) fs::create_directory(cloudjectsPath);
//                
//                remedi::io::writeCloudjects(cloudjectsPath, fids[v], cloudjectsF, true, true); // binary, header
//            }
//        }
//    }
    
    
//     Describe the point clouds got from the blobs and store the descriptors in PCDs.
//     ...
//
//    std::cout << "Describing training instances ..."  << std::endl;
//
////    for (int s = 0; s < 8; s++)
////    for (int s = 8; s < 16; s++)
////    for (int s = 16; s < 24; s++)
////    for (int s = 24; s < sequences.size(); s++)
//    for (int s = 0; s < sequences.size(); s++)
//    {
//        std::string resultsParent = string(PARENT_PATH) + string(RESULTS_SUBDIR)
//            + sequences[s]->getName() + "/" + string(KINECT_SUBSUBDIR);
//
//        std::cout << resultsParent << std::endl;
//        
//        sequences[s]->restart();
//        while (sequences[s]->hasNextFrames())
//        {
//            sequences[s]->next();
//            std::vector<string> fids = sequences[s]->getFramesFilenames();
//
//            for (int v = 0; v < sequences[s]->getNumOfViews(); v++)
//            {
//                std::vector<Cloudject::Ptr> cloudjectsF;
//                remedi::io::readCloudjects(resultsParent + string(CLOUDJECTS_DIRNAME) + sequences[s]->getViewName(v), fids[v], cloudjectsF, true); // header = true
//                
//                // Feature extraction
//                remedi::features::describeCloudjectsWithPFHRGB(cloudjectsF, OR_PFHDESC_LEAFSIZE, OR_PFHDESC_NORMAL_RADIUS, OR_PFHDESC_PFH_RADIUS); // true verbose
//                
//                std::string cloudjectsPath = resultsParent + string(CLOUDJECTS_DIRNAME) + sequences[s]->getViewName(v);
//                if (!fs::is_directory(cloudjectsPath)) fs::create_directory(cloudjectsPath);
//        
//                remedi::io::writeCloudjectsDescription(cloudjectsPath, fids[v], cloudjectsF, true); // true binary
//            }
//        }
//    
//        std::cout << std::endl;
//    }
    
    return 0;
}

// *---------------------------------------------------------------------------*
// | Learning and prediction
// *---------------------------------------------------------------------------*

int runClassificationTrain(ReMedi::Ptr pSys, std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> sequencesSids, const Groundtruth& gt, int beginFold = 0, int endFold = 0, int beginRep = 0, int endRep = 0)
{
    int numOfAnnotations = sizeof(g_AnnotationLabels)/sizeof(g_AnnotationLabels[0]);
    std::vector<const char*> annotationLabels (g_AnnotationLabels, g_AnnotationLabels + numOfAnnotations);
    
    int numOfObjects = sizeof(g_ObjectsLabels)/sizeof(g_ObjectsLabels[0]);
    std::vector<const char*> objectsLabels (g_ObjectsLabels, g_ObjectsLabels + numOfObjects);

    std::vector<int> quantValParams;
    quantValParams += 50, 100, 200, 400;
    
    std::vector<std::vector<float> > svmrbfValParams;
    std::vector<float> gammaValParams, reglValParams;
    gammaValParams += 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 0.5, 1, 5, 10; // rbf kernel's gamma
    reglValParams += 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 5, 10, 50, 100, 1000; // regularization CValue
    svmrbfValParams += gammaValParams, reglValParams;
    
    // Train classifiers and measure recognition accuracy
    for (int t = beginFold; t < endFold; t++)
    {
        std::cout << "Object classification train in fold " << t << " (LOSOCV) .. " << std::endl;

        std::vector<CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr> pipelines (endRep - beginRep);
        bool bSuccess = true;
        for (int r = beginRep; r < endRep; r++)
        {
            pipelines[r-beginRep] = CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr(new CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>);
            bSuccess &= pipelines[r-beginRep]->load("models/training_" + boost::lexical_cast<string>(t) + "-" + boost::lexical_cast<string>(r));
        }
        
        if (bSuccess)
            std::cout << "Loaded from a previous V+T procedure." << std::endl;
        else
        {
            CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr pPipeline (new CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>);
            
            std::cout << "Creating the sample of training cloudjects .." << std::endl;
            boost::shared_ptr<std::list<MockCloudject::Ptr> > pCloudjectsTr (new std::list<MockCloudject::Ptr>);
            remedi::getTrainingCloudjectsWithDescriptor(sequences, sequencesSids, t, annotationLabels, gt, "pfhrgb250", *pCloudjectsTr);
            std::cout << "Created a training sample of " << pCloudjectsTr->size() << " cloudjects." << std::endl;

            pPipeline->setInputCloudjects(pCloudjectsTr);
            pPipeline->setCategories(objectsLabels);

            pPipeline->setNormalization(CCPIPELINE_NORM_L2);
            pPipeline->setDimRedVariance(0.9);
            //  pPipeline->setDimRedNumFeatures(15);
            pPipeline->setClassifierType("svmrbf");
            
            pPipeline->setGlobalQuantization();
            //  pPipeline->setCentersStratification();
            //  pPipeline->setPerCategoryReduction();
            
            pPipeline->setValidation(10);
            pPipeline->setQuantizationValidationParameters(quantValParams);
            pPipeline->setClassifierValidationParameters(svmrbfValParams);
            
            std::cout << "Validating .." << std::endl;
            
            boost::timer valTimer;
            pPipeline->validate(); // model selection
            float valTime = valTimer.elapsed();
            
            // get the performances of the different quantizations
            std::vector<float> quantValPerfs = (pPipeline->getQuantizationValidationPerformances());
            std::cout << "Performances quantizations: " << cv::Mat(quantValParams) << ": " << cv::Mat(quantValPerfs) << std::endl;
            std::vector<std::vector<float> > classfierParameters =  pPipeline->getClassifierParameters();
            std::cout << cvx::convert<float>(classfierParameters) << std::endl;

            std::cout << "Generating final models .." << std::endl;

            // Training eventually
            // Repeat it several times (because of the quantization stochasticity)
            boost::timer trTimer;
            for (int r = beginRep; r < endRep; r++)
            {
                pPipeline->train(); // use the best parameters found in the model selection
                pPipeline->save("models/training_" + boost::lexical_cast<string>(t) + "-" + boost::lexical_cast<string>(r));
            }
            float trTime = trTimer.elapsed();
            
            std::cout << "It took " << (valTime + trTime) << " (" << valTime << " + " << trTime << ") secs." << std::endl;
        }
    }
    
    return 0;
}

int runClassificationPrediction(ReMedi::Ptr pSys, std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> sequencesSids, const Groundtruth& gt, int beginFold = 0, int endFold = 0, int beginRep = 0, int endRep = 0)
{
    // Test recognition accuracy
    for (int t = beginFold; t < endFold; t++)
    {
        std::cout << "Object classification prediction in fold " << t << " (LOSOCV) .." << std::endl;

        std::vector<CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr> pipelines (endRep - beginRep);
        bool bSuccess = true;
        for (int r = beginRep; r < endRep; r++)
        {
            pipelines[r-beginRep] = CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr(new CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>);
            bSuccess &= pipelines[r-beginRep]->load("models/training_" + boost::lexical_cast<string>(t) + "-" + boost::lexical_cast<string>(r));
        }
        
        if (bSuccess)
        {
            std::cout << "Creating the sample of test cloudjects .." << std::endl;
            boost::shared_ptr<std::list<MockCloudject::Ptr> > pCloudjectsTe (new std::list<MockCloudject::Ptr>);
            remedi::getTestCloudjectsWithDescriptor(sequences, sequencesSids, t, gt, "pfhrgb250", *pCloudjectsTe);
            std::cout << "Created a test sample of " << pCloudjectsTe->size() << " cloudjects." << std::endl;

            for (int r = beginRep; r < endRep; r++)
            {
                // Compute
                std::list<std::vector<int> > predictions;
                std::list<std::vector<float> > distsToMargin;
                std::vector<float> osacc = pipelines[r-beginRep]->predict(pCloudjectsTe, predictions, distsToMargin);
                
                // Output to console
                std::copy(osacc.begin(), osacc.end(), std::ostream_iterator<float>(std::cout, " "));
                std::cout << std::endl;
                
                // to disk
                cv::FileStorage fs ("models/testing_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r) + ".yml", cv::FileStorage::WRITE);
                fs << "predictions" << cvx::convert<int>(predictions);
                fs << "distsToMargin" << cvx::convert<float>(distsToMargin);
                fs.release();
            }
        }
    }
    
    return 0;
}

int runDetectionValidation(ReMedi::Ptr pSys, std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> sequencesSids, const Groundtruth& gt, int beginFold = 0, int endFold = 0, int beginRep = 0, int endRep = 0, std::string multiviewStrategy = "all")
{
    int numOfObjects = sizeof(g_ObjectsLabels)/sizeof(g_ObjectsLabels[0]);
    std::vector<const char*> objectsLabels (g_ObjectsLabels, g_ObjectsLabels + numOfObjects);
    
    // Monocular and multiview parameters
    std::vector<float> detectionThreshs, correspCriteria, interactionThreshs;
    correspCriteria += CloudjectInteractionPipeline::CORRESP_INC, CloudjectInteractionPipeline::CORRESP_OVL;
    detectionThreshs += 1.f/objectsLabels.size(); //0, 1.f/objectsLabels.size(), 2.f/objectsLabels.size();
    interactionThreshs += 0.02, 0.04, 0.06, 0.12;
    // Multiview parameters
    std::vector<float> mvLateFusionStrategies, mvCorrespThreshs;
    mvLateFusionStrategies += ((int)CloudjectInteractionPipeline::MULTIVIEW_LF_OR), ((int)CloudjectInteractionPipeline::MULTIVIEW_LFSCALE_DEV), ((int)CloudjectInteractionPipeline::MULTIVIEW_LF_FURTHEST);
    mvCorrespThreshs += 1.f/objectsLabels.size(); //0.05, 1.f/objectsLabels.size(), 2.f/objectsLabels.size();
    
    std::string monocularValidationFilePrefixStr = "detection_monocular_validation";
    std::string multiviewValidationFilePrefixStr = "detection_multiview_validation";
    
    boost::timer timer;
    
    for (int t = beginFold; t < endFold; t++)
    {
        std::cout << "Detection validation in fold " << t << " (LOSOCV) .." << std::endl;

        std::vector<CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr> classificationPipelines (endRep - beginRep);
        bool bSuccess = true;
        for (int r = beginRep; r < endRep; r++)
        {
            classificationPipelines[r-beginRep] = CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr(new CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>);
            bSuccess &= classificationPipelines[r-beginRep]->load("models/training_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r));
        }
        
        if (bSuccess)
        {
            std::cout << "Creating the sample of test cloudjects .." << std::endl;
            boost::shared_ptr<std::list<MockCloudject::Ptr> > pCloudjectsTe (new std::list<MockCloudject::Ptr>);
            remedi::getTestCloudjectsWithDescriptor(sequences, sequencesSids, t, gt, "pfhrgb250", *pCloudjectsTe);
            std::cout << "Created a training sample of " << pCloudjectsTe->size() << " cloudjects." << std::endl;
            
            std::vector<Sequence<ColorDepthFrame>::Ptr> sequencesTe;
            remedi::getTestSequences(sequences, sequencesSids, t, sequencesTe);
            
            CloudjectInteractionPipeline::Ptr pCjInteractionPipeline (new CloudjectInteractionPipeline);
            pCjInteractionPipeline->setInputSequences(sequencesTe);
            pCjInteractionPipeline->setCategories(objectsLabels);
            
            pCjInteractionPipeline->setLeafSize(Eigen::Vector3f(.02f,.02f,.02f));
            pCjInteractionPipeline->setGroundtruth(gt);
            
            pCjInteractionPipeline->setInteractiveRegisterer(pSys->getRegisterer());
            pCjInteractionPipeline->setTableModeler(pSys->getTableModeler());
            
            if (multiviewStrategy.compare("all") == 0 || multiviewStrategy.compare("v") == 0)
            {
                for (int r = beginRep; r < endRep; r++)
                {
                    pCjInteractionPipeline->setClassificationPipeline(classificationPipelines[r-beginRep]);

                    // MONOCULAR
                    std::cout << "Validating MONOCULAR .. " << std::endl;
                    pCjInteractionPipeline->setMultiviewStrategy(CloudjectInteractionPipeline::DETECT_MONOCULAR);
                    pCjInteractionPipeline->setValidationParameters(correspCriteria, detectionThreshs, interactionThreshs, std::vector<float>(), std::vector<float>());
                    
                    timer.restart();
                    pCjInteractionPipeline->validateDetection();
                    std::cout << pCjInteractionPipeline->getValidationPerformance() << std::endl;
                    std::cout << "It took " << timer.elapsed() << " secs." << std::endl;
                    
                    pCjInteractionPipeline->save(monocularValidationFilePrefixStr + "_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r));
                }
            }
            
            if (multiviewStrategy.compare("all") == 0 || multiviewStrategy.compare("w") == 0)
            {
                for (int r = beginRep; r < endRep; r++)
                {
                    pCjInteractionPipeline->setClassificationPipeline(classificationPipelines[r-beginRep]);

                    // MULTIVIEW
                    std::cout << "Validating MULTIVIEW .. " << std::endl;
                    pCjInteractionPipeline->setMultiviewStrategy(CloudjectInteractionPipeline::DETECT_MULTIVIEW);
                    pCjInteractionPipeline->setValidationParameters(correspCriteria, detectionThreshs, interactionThreshs, mvLateFusionStrategies, mvCorrespThreshs);

                    // Load predictions and dists to margin
                    std::vector<cv::Mat> predictionsTr, distsToMarginTr;
                    for (int tt = 0; tt < NUM_OF_SUBJECTS; tt++)
                    {
                        if (tt != t)
                        {
                            cv::FileStorage fs;
                            fs.open("models/testing_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r) + ".yml", cv::FileStorage::READ);
                            if (fs.isOpened())
                            {
                                cv::Mat aux;
                                fs["predictions"] >> aux;
                                predictionsTr.push_back(aux);
                                fs["distsToMargin"] >> aux;
                                distsToMarginTr.push_back(aux);
                            }
                        }
                    }
                    cv::Mat predictions, distsToMargin;
                    vconcat(predictionsTr, predictions);
                    vconcat(distsToMarginTr, distsToMargin);

                    // Set multiview-related parameters
                    cv::Mat scalingsMat;
                    remedi::computeScalingFactorsOfCategories(predictions, distsToMargin, scalingsMat);
                    pCjInteractionPipeline->setMultiviewLateFusionNormalization(cvx::convert<float>(scalingsMat)); // floats' Mat to vv<float>
                
                    timer.restart();
                    pCjInteractionPipeline->validateDetection();
                    std::cout << pCjInteractionPipeline->getValidationPerformance() << std::endl;
                    std::cout << "It took " << timer.elapsed() << " secs." << std::endl;
                    
                    pCjInteractionPipeline->save(multiviewValidationFilePrefixStr + "_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r));
                }
            }
        }
    }
    
    return 0;
}

int runDetectionPrediction(ReMedi::Ptr pSys, std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> sequencesSids, const Groundtruth& gt, int beginFold = 0, int endFold = 0, int beginRep = 0, int endRep = 0, std::string multiviewStrategy = "all")
{
    // TODO: implement
    
    return 0;
}


int runInteractionValidation(ReMedi::Ptr pSys, std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> sequencesSids, const Groundtruth& gt, const Interaction& iact, int beginFold = 0, int endFold = 0, int beginRep = 0, int endRep = 0, std::string multiviewStrategy = "all")
{
    int numOfObjects = sizeof(g_ObjectsLabels)/sizeof(g_ObjectsLabels[0]);
    std::vector<const char*> objectsLabels (g_ObjectsLabels, g_ObjectsLabels + numOfObjects);
    
    // Monocular and multiview parameters
    std::vector<float> detectionThreshs, correspCriteria, interactionThreshs;
    correspCriteria += CloudjectInteractionPipeline::CORRESP_INC, CloudjectInteractionPipeline::CORRESP_OVL;
    detectionThreshs += 1.f/objectsLabels.size(); //0, 1.f/objectsLabels.size(), 2.f/objectsLabels.size();
    interactionThreshs += 0.025, 0.05, 0.10;
    // Multiview parameters
    std::vector<float> mvLateFusionStrategies, mvCorrespThreshs;
    mvLateFusionStrategies += ((int)CloudjectInteractionPipeline::MULTIVIEW_LF_OR), ((int)CloudjectInteractionPipeline::MULTIVIEW_LFSCALE_DEV), ((int)CloudjectInteractionPipeline::MULTIVIEW_LF_FURTHEST);
    mvCorrespThreshs += 1.f/objectsLabels.size(); //0.05, 1.f/objectsLabels.size(), 2.f/objectsLabels.size();
    
    std::string monocularValidationFilePrefixStr = "interaction_validations/interaction_monocular_validation";
    std::string multiviewValidationFilePrefixStr = "interaction_validations/interaction_multiview_validation";
    
    boost::timer timer;
    
    for (int t = beginFold; t < endFold; t++)
    {
        std::cout << "Interaction validation in fold " << t << " (LOSOCV) .." << std::endl;
        
        std::vector<CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr> classificationPipelines (endRep - beginRep);
        bool bSuccess = true;
        for (int r = beginRep; r < endRep; r++)
        {
            classificationPipelines[r-beginRep] = CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr(new CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>);
            bSuccess &= classificationPipelines[r-beginRep]->load("models/training_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r));
        }
        
        if (bSuccess)
        {
            std::cout << "Creating the sample of test cloudjects .." << std::endl;
            boost::shared_ptr<std::list<MockCloudject::Ptr> > pCloudjectsTe (new std::list<MockCloudject::Ptr>);
            remedi::getTestCloudjectsWithDescriptor(sequences, sequencesSids, t, gt, "pfhrgb250", *pCloudjectsTe);
            std::cout << "Created a training sample of " << pCloudjectsTe->size() << " cloudjects." << std::endl;
            
            std::vector<Sequence<ColorDepthFrame>::Ptr> sequencesTe;
            remedi::getTestSequences(sequences, sequencesSids, t, sequencesTe);
            
            CloudjectInteractionPipeline::Ptr pCjInteractionPipeline (new CloudjectInteractionPipeline);
            pCjInteractionPipeline->setInputSequences(sequencesTe);
            pCjInteractionPipeline->setCategories(objectsLabels);
            
            pCjInteractionPipeline->setLeafSize(Eigen::Vector3f(.02f,.02f,.02f));
            pCjInteractionPipeline->setGroundtruth(gt);
            pCjInteractionPipeline->setInteractionGroundtruth(iact);
            pCjInteractionPipeline->setValidationInteractionOverlapCriterion(CloudjectInteractionPipeline::INTERACTION_OVL_OVERALL); // evaluate the goodness of detection begin-end of interaction
            
            pCjInteractionPipeline->setBackgroundSubtractor(pSys->getBackgroundSubtractor());
            pCjInteractionPipeline->setInteractiveRegisterer(pSys->getRegisterer());
            pCjInteractionPipeline->setTableModeler(pSys->getTableModeler());
            
            pSys->getBackgroundSubtractor()->model();
            
            if (multiviewStrategy.compare("all") == 0 || multiviewStrategy.compare("v") == 0)
            {
                for (int r = beginRep; r < endRep; r++)
                {
                    pCjInteractionPipeline->setClassificationPipeline(classificationPipelines[r-beginRep]);
                    
                    // MONOCULAR
                    std::cout << "Validating MONOCULAR .. " << std::endl;
                    pCjInteractionPipeline->setMultiviewStrategy(CloudjectInteractionPipeline::DETECT_MONOCULAR);
                    pCjInteractionPipeline->setValidationParameters(correspCriteria, detectionThreshs, interactionThreshs, std::vector<float>(), std::vector<float>());

                    timer.restart();
                    pCjInteractionPipeline->validate();
                    std::cout << pCjInteractionPipeline->getValidationPerformance() << std::endl;
                    std::cout << "It took " << timer.elapsed() << " secs." << std::endl;
                    
                    pCjInteractionPipeline->save(monocularValidationFilePrefixStr + "_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r));
                }
            }
            
            if (multiviewStrategy.compare("all") == 0 || multiviewStrategy.compare("w") == 0)
            {
                for (int r = beginRep; r < endRep; r++)
                {
                    pCjInteractionPipeline->setClassificationPipeline(classificationPipelines[r-beginRep]);

                    // MULTIVIEW
                    std::cout << "Validating MULTIVIEW .. " << std::endl;
                    pCjInteractionPipeline->setMultiviewStrategy(CloudjectInteractionPipeline::DETECT_MULTIVIEW);
                    pCjInteractionPipeline->setValidationParameters(correspCriteria, detectionThreshs, interactionThreshs, mvLateFusionStrategies, mvCorrespThreshs);
                    
                    // Load predictions and dists to margin
                    std::vector<cv::Mat> predictionsTr, distsToMarginTr;
                    for (int tt = 0; tt < NUM_OF_SUBJECTS; tt++)
                    {
                        if (tt != t)
                        {
                            cv::FileStorage fs;
                            fs.open("models/testing_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r) + ".yml", cv::FileStorage::READ);
                            if (fs.isOpened())
                            {
                                cv::Mat aux;
                                fs["predictions"] >> aux;
                                predictionsTr.push_back(aux);
                                fs["distsToMargin"] >> aux;
                                distsToMarginTr.push_back(aux);
                            }
                        }
                    }
                    cv::Mat predictions, distsToMargin;
                    vconcat(predictionsTr, predictions);
                    vconcat(distsToMarginTr, distsToMargin);
                    
                    // Set multiview-related parameters
                    cv::Mat scalingsMat;
                    remedi::computeScalingFactorsOfCategories(predictions, distsToMargin, scalingsMat);
                    pCjInteractionPipeline->setMultiviewLateFusionNormalization(cvx::convert<float>(scalingsMat)); // floats' Mat to vv<float>
                    
                    timer.restart();
                    pCjInteractionPipeline->validate();
                    std::cout << pCjInteractionPipeline->getValidationPerformance() << std::endl;
                    std::cout << "It took " << timer.elapsed() << " secs." << std::endl;
                    
                    pCjInteractionPipeline->save(multiviewValidationFilePrefixStr + " " + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r));
                }
            }
        }
    }
    
    return 0;
}

cv::Mat fscore(cv::Mat errors)
{
    std::vector<cv::Mat> channels;
    cv::split(errors, channels);
    
    cv::Mat F = 2*channels[0] / (2*channels[0] + channels[1] + channels[2]);
    cv::Mat f;
    cv::reduce(F, f, 1, CV_REDUCE_AVG);
    
    return f;
}

cv::Mat overlap(cv::Mat errors)
{
    std::vector<cv::Mat> channels;
    cv::split(errors, channels);
    
    cv::Mat F = channels[0] / (channels[0] + channels[1] + channels[2]);
    cv::Mat f;
    cv::reduce(F, f, 1, CV_REDUCE_AVG);
    
    return f;
}

int getInteractionBestValidationCombination(std::string prefix, int t, int beginFold = 0, int endFold = 0, int beginRep = 0, int endRep = 0)
{
    bool bSuccess;
    
    int numOfViews = 0;
    std::vector<cv::Mat> resultsTr;
    for (int r = beginRep; r < endRep; r++)
    {
        for (int tt = 0; tt < NUM_OF_SUBJECTS; tt++)
        {
            if (tt != t)
            {
                CloudjectInteractionPipeline::Ptr pCjInteractionPipeline (new CloudjectInteractionPipeline);
                bSuccess = pCjInteractionPipeline->load(prefix + "_" + boost::lexical_cast<std::string>(tt) + "-" + boost::lexical_cast<std::string>(r));
                
                if (bSuccess)
                {
                    std::vector<cv::Mat> results = pCjInteractionPipeline->getInteractionResults();
                    
                    if (numOfViews < 1) numOfViews = results.size();
                    else assert(numOfViews == results.size());
                    
                    for (int v = 0; v < results.size(); v++)
                        resultsTr.push_back( overlap(results[v]) );
                }
            }
        }
    }
    
    cv::Point bestCombIdx;
    for (int v = 0; v < numOfViews; v++)
    {
        cv::Mat R;
        cv::hconcat(resultsTr, R);
        
        cv::Mat f;
        cv::reduce(R, f, 1, CV_REDUCE_AVG);
        
        double minVal, maxVal;
        cv::Point minPt, maxPt;
        cv::minMaxLoc(f, &minVal, &maxVal, &minPt, &bestCombIdx);
    }
    
    return bestCombIdx.y;
}

int runInteractionPrediction(ReMedi::Ptr pSys, std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> sequencesSids, const Groundtruth& gt, const Interaction& iact, int beginFold = 0, int endFold = 0, int beginRep = 0, int endRep = 0, std::string multiviewStrategy = "all")
{
    int numOfObjects = sizeof(g_ObjectsLabels)/sizeof(g_ObjectsLabels[0]);
    std::vector<const char*> objectsLabels (g_ObjectsLabels, g_ObjectsLabels + numOfObjects);
    
    // Monocular and multiview parameters
    std::vector<float> detectionThreshs, correspCriteria, interactionThreshs;
    correspCriteria += CloudjectInteractionPipeline::CORRESP_INC, CloudjectInteractionPipeline::CORRESP_OVL;
    detectionThreshs += 1.f/objectsLabels.size(); //0, 1.f/objectsLabels.size(), 2.f/objectsLabels.size();
    interactionThreshs += 0.025, 0.05, 0.10;
    // Multiview parameters
    std::vector<float> mvLateFusionStrategies, mvCorrespThreshs;
    mvLateFusionStrategies += ((int)CloudjectInteractionPipeline::MULTIVIEW_LF_OR), ((int)CloudjectInteractionPipeline::MULTIVIEW_LFSCALE_DEV), ((int)CloudjectInteractionPipeline::MULTIVIEW_LF_FURTHEST);
    mvCorrespThreshs += 1.f/objectsLabels.size(); //0.05, 1.f/objectsLabels.size(), 2.f/objectsLabels.size();
    
    std::vector<std::vector<float> > monocularParameters;
    monocularParameters += detectionThreshs, correspCriteria, interactionThreshs;
    std::vector<std::vector<float> > multiviewParameters;
    multiviewParameters += detectionThreshs, correspCriteria, interactionThreshs, mvLateFusionStrategies, mvCorrespThreshs;
    
    std::string monocularPredictionFilePrefixStr = "interaction_predictions/interaction_monocular_prediction";
    std::string multiviewPredictionFilePrefixStr = "interaction_predictions/interaction_multiview_prediction";
    
    bool bSuccess;
    for (int t = beginFold; t < endFold; t++)
    {
        std::cout << "Interaction validation in fold " << t << " (LOSOCV) .." << std::endl;
        
        bSuccess = true;

        std::vector<CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr> classificationPipelines (endRep - beginRep);
        for (int r = beginRep; r < endRep; r++)
        {
            classificationPipelines[r-beginRep] = CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr(new CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>);
            bSuccess &= classificationPipelines[r-beginRep]->load("models/training_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r));
        }
        
        if (bSuccess)
        {
            //
            // Monocular prediction
            //
            
            std::cout << "Creating the sample of test cloudjects .." << std::endl;
            boost::shared_ptr<std::list<MockCloudject::Ptr> > pCloudjectsTe (new std::list<MockCloudject::Ptr>);
            remedi::getTestCloudjectsWithDescriptor(sequences, sequencesSids, t, gt, "pfhrgb250", *pCloudjectsTe);
            std::cout << "Created a training sample of " << pCloudjectsTe->size() << " cloudjects." << std::endl;

            std::vector<Sequence<ColorDepthFrame>::Ptr> sequencesTe;
            remedi::getTestSequences(sequences, sequencesSids, t, sequencesTe);

            CloudjectInteractionPipeline::Ptr pCjInteractionPipeline (new CloudjectInteractionPipeline);
            pCjInteractionPipeline->setInputSequences(sequencesTe);
            pCjInteractionPipeline->setCategories(objectsLabels);

            pCjInteractionPipeline->setLeafSize(Eigen::Vector3f(.02f,.02f,.02f));
            pCjInteractionPipeline->setGroundtruth(gt);
            pCjInteractionPipeline->setInteractionGroundtruth(iact);
            
            pCjInteractionPipeline->setBackgroundSubtractor(pSys->getBackgroundSubtractor());
            pCjInteractionPipeline->setInteractiveRegisterer(pSys->getRegisterer());
            pCjInteractionPipeline->setTableModeler(pSys->getTableModeler());
            
            pSys->getBackgroundSubtractor()->model();
            
            int bestCombIdx = 0;
            
            if (multiviewStrategy.compare("all") == 0 || multiviewStrategy.compare("v") == 0)
            {
                bestCombIdx = getInteractionBestValidationCombination(monocularPredictionFilePrefixStr, t, beginFold, endFold, beginRep, endRep);

                pCjInteractionPipeline->setMultiviewStrategy(CloudjectInteractionPipeline::DETECT_MONOCULAR);

                // Set the best combination of parameters

                std::vector<std::vector<float> > monocularCombinations;
                expandParameters<float>(monocularParameters, monocularCombinations);
                pCjInteractionPipeline->setCorrespondenceCriterion(monocularCombinations[bestCombIdx][0]);
                pCjInteractionPipeline->setDetectionCorrespondenceThresh(monocularCombinations[bestCombIdx][1]);
                pCjInteractionPipeline->setInteractionThresh(monocularCombinations[bestCombIdx][2]);
                
                for (int r = beginRep; r < endRep; r++)
                {
                    pCjInteractionPipeline->setClassificationPipeline(classificationPipelines[r-beginRep]);
                    
                    pCjInteractionPipeline->predict();
                    pCjInteractionPipeline->save(monocularPredictionFilePrefixStr + "_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r));
                }
            }
            
            //
            // Multiview prediction
            //
            
            if (multiviewStrategy.compare("all") == 0 || multiviewStrategy.compare("w") == 0)
            {
                bestCombIdx = getInteractionBestValidationCombination(multiviewPredictionFilePrefixStr, t, beginFold, endFold, beginRep, endRep);
                
                // Proceed to prediction
                
                pCjInteractionPipeline->setMultiviewStrategy(CloudjectInteractionPipeline::DETECT_MULTIVIEW);

                // Set the best combination of parameters
                
                std::vector<std::vector<float> > multiviewCombinations;
                expandParameters<float>(multiviewParameters, multiviewCombinations);
                pCjInteractionPipeline->setCorrespondenceCriterion(multiviewCombinations[bestCombIdx][0]);
                pCjInteractionPipeline->setDetectionCorrespondenceThresh(multiviewCombinations[bestCombIdx][1]);
                pCjInteractionPipeline->setInteractionThresh(multiviewCombinations[bestCombIdx][2]);
                pCjInteractionPipeline->setMultiviewLateFusionStrategy(multiviewCombinations[bestCombIdx][3]);
                pCjInteractionPipeline->setMultiviewCorrespondenceThresh(multiviewCombinations[bestCombIdx][4]);
                
                for (int r = beginRep; r < endRep; r++)
                {
                    pCjInteractionPipeline->setClassificationPipeline(classificationPipelines[r-beginRep]);
                    
                    pCjInteractionPipeline->predict();
                    pCjInteractionPipeline->save(multiviewPredictionFilePrefixStr + "_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r));
                }
            }
        }
    }
    return 0;
}

int runFilteringValidation(ReMedi::Ptr pSys, std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> sequencesSids, const Groundtruth& gt, const Interaction& iact)
{
    std::vector<float> bridges, cleans, medianSizes, openCloseSizes;
    bridges += 0, 1;
    cleans += 0, 1;
    medianSizes += 0, 1, 2, 3;
    openCloseSizes += 0, -2, -1, 1, 2; // negative is closing, and positive for opening sizes
    
    std::string monocularPredictionFilePrefixStr = "interaction_predictions/interaction_monocular_prediction";
    std::string multiviewPredictionFilePrefixStr = "interaction_predictions/interaction_multiview_prediction";
    
    cv::FileStorage fs;

    for (int t = 0; t < NUM_OF_SUBJECTS; t++)
    {
        std::cout << "Interaction filtering in fold " << t << " (LOSOCV) .." << std::endl;
        
        // Monocular
        
        int bestCombIdx;
        
        std::vector<std::vector<cv::Mat> > predictionsFoldRepTr (pSys->getBackgroundSubtractor()->getNumOfViews());
        std::vector<std::vector<cv::Mat> > groundtruthFoldRepTr (pSys->getBackgroundSubtractor()->getNumOfViews());
        
        for (int tt = 0; tt < NUM_OF_SUBJECTS; tt++)
        {
            if (tt != t)
            {
                for (int r = 0; r < NUM_REPETITIONS; r++)
                {
                    std::string path = monocularPredictionFilePrefixStr + "_" + boost::lexical_cast<std::string>(tt) + "-" + boost::lexical_cast<std::string>(r) + ".yml";
                    fs.open(path, cv::FileStorage::READ);

                    int numOfSequencesInFold;
                    fs["numSequences"] >> numOfSequencesInFold;
                    
                    if (!fs.isOpened())
                    {
                        std::cerr << path << " not found" << std::endl;
                    }
                    else
                    {
                        for (int s = 0; s < numOfSequencesInFold; s++)
                        {
                            for (int v = 0; v < pSys->getBackgroundSubtractor()->getNumOfViews(); v++)
                            {
                                cv::Mat p, g;
                                fs[ "interactionPredictions-" + boost::lexical_cast<std::string>(s) + "-"
                                   + boost::lexical_cast<std::string>(v) ] >> p;
                                fs[ "interactionGroundtruth-" + boost::lexical_cast<std::string>(s) + "-"
                                   + boost::lexical_cast<std::string>(v) ] >> g;
                                
                                predictionsFoldRepTr[v].push_back(p);
                                groundtruthFoldRepTr[v].push_back(g);
                            }
                        }
                    }
                    
                    fs.release();
                }
            }
        }
        
        InteractionFilteringPipeline::Ptr pFilteringPipeline (new InteractionFilteringPipeline);
        pFilteringPipeline->setInputData(predictionsFoldRepTr);
        pFilteringPipeline->setInputGroundtruth(groundtruthFoldRepTr);
        pFilteringPipeline->setEvaluationMetric(InteractionFilteringPipeline::EVALUATE_OVERLAP);
        pFilteringPipeline->setValidationParameters(bridges, cleans, medianSizes, openCloseSizes);
        pFilteringPipeline->validate();
        
        // Results
        std::vector<std::vector<float> > combinations;
        std::vector<float> performances;
        pFilteringPipeline->getValidationResults(combinations, performances);
        
        std::cout << cv::Mat(performances) << std::endl;
        
        double minVal, maxVal;
        cv::Point minPt, maxPt;
        cv::minMaxLoc(performances, &minVal, &maxVal, &minPt, &maxPt);
        
        bestCombIdx = maxPt.y;

        
        // Multiview
    }
    
    return 0;
}

int runFilteringPrediction(ReMedi::Ptr pSys, std::vector<Sequence<ColorDepthFrame>::Ptr> sequences, std::vector<int> sequencesSids, const Groundtruth& gt, const Interaction& iact)
{
}

int main(int argc, char** argv)
{
    std::cout.precision(3);

    // *-----------------------------------------------------------------------*
    // | Read parameters
    // *-----------------------------------------------------------------------*
    
    bool bClassificationTrain, bClassificationPrediction,
        bDetectionValidation, bDetectionPrediction,
        bInteractionValidation, bInteractionPrediction,
        bFilteringValidation, bFilteringPrediction;
    
    bClassificationTrain      = (pcl::console::find_argument(argc, argv, "-Ct") >= 0);
    bClassificationPrediction = (pcl::console::find_argument(argc, argv, "-Cp") >= 0);
    bDetectionValidation      = (pcl::console::find_argument(argc, argv, "-Dv") >= 0);
    bDetectionPrediction      = (pcl::console::find_argument(argc, argv, "-Dp") >= 0);
    bInteractionValidation    = (pcl::console::find_argument(argc, argv, "-Iv") >= 0);
    bInteractionPrediction    = (pcl::console::find_argument(argc, argv, "-Ip") >= 0);
    bFilteringValidation      = (pcl::console::find_argument(argc, argv, "-Lv") >= 0);
    bFilteringPrediction      = (pcl::console::find_argument(argc, argv, "-Lp") >= 0);
    
    int beginFold = 0, endFold = 0;
    pcl::console::parse_2x_arguments(argc, argv, "-F", beginFold, endFold);
    
    int beginRep = 0, endRep = 0;
    pcl::console::parse_2x_arguments(argc, argv, "-R", beginRep, endRep);
    
    std::string multiviewStrategy;
    pcl::console::parse_argument(argc, argv, "-V", multiviewStrategy);
    
    // *-----------------------------------------------------------------------*
    // | Read & list the paths of the seqs from PARENT_PATH/SEQUENCES_SUBDIR/
    // *-----------------------------------------------------------------------*
    
    std::vector<std::string> sequencesDirnames; // sequence directory paths
    std::vector<int> sequencesSids; // sequence's subject identifier (for validation purposes)
    remedi::loadSequences(string(PARENT_PATH) + string(SEQUENCES_SUBDIR), sequencesDirnames, sequencesSids);
    
    std::vector<const char*> viewsNames (g_Views, g_Views + sizeof(g_Views)/sizeof(g_Views[0]));
    
    std::vector<std::vector<std::string> > colorFramesPaths, depthFramesPaths;
    remedi::loadFramesPaths(string(PARENT_PATH) + string(SEQUENCES_SUBDIR), sequencesDirnames, KINECT_SUBSUBDIR, FRAMES_COLOR_DIRNAME, viewsNames, colorFramesPaths);
    remedi::loadFramesPaths(string(PARENT_PATH) + string(SEQUENCES_SUBDIR), sequencesDirnames, KINECT_SUBSUBDIR, FRAMES_DEPTH_DIRNAME, viewsNames, depthFramesPaths);
    
    assert(colorFramesPaths.size() == depthFramesPaths.size());
    
    // *-----------------------------------------------------------------------*
    // | Read the frames from sequences
    // *-----------------------------------------------------------------------*
    
    KinectReader reader;
    
    // Create and allocate the frames of the background sequence
    Sequence<ColorDepthFrame>::Ptr pBgSeq (new Sequence<ColorDepthFrame>(viewsNames));
    //    reader.setPreallocation();
    reader.read(colorFramesPaths[0], depthFramesPaths[0], *pBgSeq); // 0-th seq is the bg seq
    //    reader.setPreallocation(false);
    
    // Create and allocate the frames of the rest of the sequences (NOT background sequences)
    std::vector<Sequence<ColorDepthFrame>::Ptr> sequences;
    
    // Load from a file that contains the delays among each sequence's streams (manually specified)
    std::vector<vector<int> > delays;
    remedi::loadDelaysFile(string(PARENT_PATH) + string(SEQUENCES_SUBDIR), string(DELAYS_FILENAME), delays);
    
    for (int s = 1; s < colorFramesPaths.size(); s++) // start from 1 (0 is the bg sequence)
    {
        Sequence<ColorDepthFrame>::Ptr pSeq (new Sequence<ColorDepthFrame>(viewsNames));
        reader.read(colorFramesPaths[s], depthFramesPaths[s], *pSeq);
        pSeq->setPath(string(PARENT_PATH) + string(SEQUENCES_SUBDIR)
                      + sequencesDirnames[s] + "/" + string(KINECT_SUBSUBDIR));
        pSeq->setName(sequencesDirnames[s]);
        pSeq->setDelays(delays[s]);
        
        sequences.push_back(pSeq);
    }
    
    // *-----------------------------------------------------------------------*
    // | Create and parametrize the general system parameters
    // *-----------------------------------------------------------------------*
    
    ReMedi::Ptr pSys (new ReMedi); // the system
    
    pSys->setFramesResolution(X_RESOLUTION, Y_RESOLUTION); // resolution of frames in pixels
    pSys->setDefaultFrame(DEFAULT_FRAME); // an arbitrary frame (for registration, table modeling, etc)
    pSys->setBackgroundSequence(pBgSeq); // Set the background sequence
    
    pSys->setRegistererParameters(IR_NUM_OF_POINTS, IR_VIS_WND_HEIGHT, IR_VIS_WND_WIDTH, IR_VIS_VP, IR_VIS_HP, IR_VIS_DIST, IR_VIS_MARKER_RADIUS);
    pSys->setTableModelerParameters(TM_LEAF_SIZE, TM_NORMAL_RADIUS, TM_SAC_ITERS, TM_SAC_DIST_THRESH, TM_TT_Y_OFFSET, TM_INTERACTIVE_BORDER_DIST, TM_CONF_LEVEL);
    pSys->setSubtractorParameters(BS_NUM_OF_SAMPLES, BS_MODALITY, BS_K, BS_LRATE, BS_BGRATIO, BS_VARGEN, BS_MORPHOLOGY);
    
    pSys->initialize();
    
    // Run
    Groundtruth gt;
    remedi::loadGroundtruth(sequences, gt);
    
    if (bClassificationTrain)
        runClassificationTrain(pSys, sequences, sequencesSids, gt,
                               (beginFold < endFold) ? beginFold : 0,
                               (beginFold < endFold) ? endFold : NUM_OF_SUBJECTS,
                               (beginRep < endRep) ? beginRep : 0,
                               (beginRep < endRep) ? endRep : NUM_REPETITIONS);
    if (bClassificationPrediction)
        runClassificationPrediction(pSys, sequences, sequencesSids, gt,
                                    (beginFold < endFold) ? beginFold : 0,
                                    (beginFold < endFold) ? endFold : NUM_OF_SUBJECTS,
                                    (beginRep < endRep) ? beginRep : 0,
                                    (beginRep < endRep) ? endRep : NUM_REPETITIONS);
    
    int numOfObjects = sizeof(g_ObjectsLabels)/sizeof(g_ObjectsLabels[0]);
    std::vector<const char*> objectsLabels (g_ObjectsLabels, g_ObjectsLabels + numOfObjects);
    
    if (bDetectionValidation)
        runDetectionValidation(pSys, sequences, sequencesSids, gt,
                               (beginFold < endFold) ? beginFold : 0,
                               (beginFold < endFold) ? endFold : NUM_OF_SUBJECTS,
                               (beginRep < endRep) ? beginRep : 0,
                               (beginRep < endRep) ? endRep : NUM_REPETITIONS,
                               multiviewStrategy.empty() ? "all" : multiviewStrategy);
    if (bDetectionPrediction)
        runDetectionPrediction(pSys, sequences, sequencesSids, gt,
                               (beginFold < endFold) ? beginFold : 0,
                               (beginFold < endFold) ? endFold : NUM_OF_SUBJECTS,
                               (beginRep < endRep) ? beginRep : 0,
                               (beginRep < endRep) ? endRep : NUM_REPETITIONS,
                               multiviewStrategy.empty() ? "all" : multiviewStrategy);
    
    Interaction iact;
    remedi::loadInteraction(sequences, objectsLabels, iact); // interaction groundtruth indicating just the begin-end of interactions
    
    if (bInteractionValidation)
        runInteractionValidation(pSys, sequences, sequencesSids, gt, iact,
                                 (beginFold < endFold) ? beginFold : 0,
                                 (beginFold < endFold) ? endFold : NUM_OF_SUBJECTS,
                                 (beginRep < endRep) ? beginRep : 0,
                                 (beginRep < endRep) ? endRep : NUM_REPETITIONS,
                                 multiviewStrategy.empty() ? "all" : multiviewStrategy);
    if (bInteractionPrediction)
        runInteractionPrediction(pSys, sequences, sequencesSids, gt, iact,
                                 (beginFold < endFold) ? beginFold : 0,
                                 (beginFold < endFold) ? endFold : NUM_OF_SUBJECTS,
                                 (beginRep < endRep) ? beginRep : 0,
                                 (beginRep < endRep) ? endRep : NUM_REPETITIONS,
                                 multiviewStrategy.empty() ? "all" : multiviewStrategy);
    
    if (bFilteringValidation)
        runFilteringValidation(pSys, sequences, sequencesSids, gt, iact);

    if (bFilteringPrediction)
        runFilteringPrediction(pSys, sequences, sequencesSids, gt, iact);
    
    return 0;
}
