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

using namespace std;
using namespace boost::assign;

namespace fs = boost::filesystem;

int run()
{
// *---------------------------------------------------------------------------*
// | Read & list the paths of the seqs from PARENT_PATH/SEQUENCES_SUBDIR/
// *---------------------------------------------------------------------------*
    
    std::vector<std::string> sequencesDirnames; // sequence directory paths
    std::vector<int> sequencesSids; // sequence's subject identifier (for validation purposes)
    remedi::loadSequences(string(PARENT_PATH) + string(SEQUENCES_SUBDIR), sequencesDirnames, sequencesSids);
    
    std::vector<const char*> viewsNames (g_Views, g_Views + sizeof(g_Views)/sizeof(g_Views[0]));
    
    std::vector<std::vector<std::string> > colorFramesPaths, depthFramesPaths;
    remedi::loadFramesPaths(string(PARENT_PATH) + string(SEQUENCES_SUBDIR), sequencesDirnames, KINECT_SUBSUBDIR, FRAMES_COLOR_DIRNAME, viewsNames, colorFramesPaths);
    remedi::loadFramesPaths(string(PARENT_PATH) + string(SEQUENCES_SUBDIR), sequencesDirnames, KINECT_SUBSUBDIR, FRAMES_DEPTH_DIRNAME, viewsNames, depthFramesPaths);
    
    assert(colorFramesPaths.size() == depthFramesPaths.size());
    
// *---------------------------------------------------------------------------*
// | Read the frames from sequences
// *---------------------------------------------------------------------------*
    
    KinectReader reader;
    
    // Create and allocate the frames of the background sequence
    Sequence<ColorDepthFrame>::Ptr pBgSeq (new Sequence<ColorDepthFrame>(viewsNames));
//    reader.setPreallocation();
    reader.read(colorFramesPaths[0], depthFramesPaths[0], *pBgSeq); // 0-th seq is the bg seq
//    reader.setPreallocation(false);
    
    // Create and allocate the frames of the rest of the sequences (NOT background sequences)
    vector<Sequence<ColorDepthFrame>::Ptr> sequences;
    
    // Load from a file that contains the delays among each sequence's streams (manually specified)
    vector<vector<int> > delays;
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
    
// *---------------------------------------------------------------------------*
// | Create and parametrize the general system parameters
// *---------------------------------------------------------------------------*
    
    ReMedi sys; // the system
    
    sys.setFramesResolution(X_RESOLUTION, Y_RESOLUTION); // resolution of frames in pixels
    sys.setDefaultFrame(DEFAULT_FRAME); // an arbitrary frame (for registration, table modeling, etc)
    sys.setBackgroundSequence(pBgSeq); // Set the background sequence

    sys.setRegistererParameters(IR_NUM_OF_POINTS, IR_VIS_WND_HEIGHT, IR_VIS_WND_WIDTH, IR_VIS_VP, IR_VIS_HP, IR_VIS_DIST, IR_VIS_MARKER_RADIUS);
    sys.setTableModelerParameters(TM_LEAF_SIZE, TM_NORMAL_RADIUS, TM_SAC_ITERS, TM_SAC_DIST_THRESH, TM_TT_Y_OFFSET, TM_INTERACTIVE_BORDER_DIST, TM_CONF_LEVEL);
    sys.setSubtractorParameters(BS_NUM_OF_SAMPLES, BS_MODALITY, BS_K, BS_LRATE, BS_BGRATIO, BS_VARGEN, BS_MORPHOLOGY);

    sys.initialize();
    
// *---------------------------------------------------------------------------*
// | Perform table top segmentation
// *---------------------------------------------------------------------------*
//    
//    for (int s = 0; s < sequences.size(); s++)
//    {
//        sequences[s]->restart();
//        while (sequences[s]->hasNextFrames())
//        {
//            vector<ColorDepthFrame::Ptr> testFrames = sequences[s]->nextFrames();
//            sys.getRegisterer()->setInputFrames(testFrames);
//            sys.getRegisterer()->registrate(testFrames);
//            
//            // focus only on tabletop's region for the BS
//            vector<cv::Mat> tabletopMasks;
//            sys.getTableModeler()->getTabletopMask(testFrames, tabletopMasks);
//            
//            vector<string> fids = sequences[s]->getFramesFilenames();
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
//    sys.getBackgroundSubtractor()->model(); // train the background model using the bgSeq
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
//            sys.getBackgroundSubtractor()->setInputFrames(testFrames);
//            sys.getBackgroundSubtractor()->subtract(foregroundMasks);
//            
//            vector<string> fids = sequences[s]->getFramesFilenames();
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
//            vector<string> fids = sequences[s]->getFramesFilenames();
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
//            vector<string> fids = sequences[s]->getFramesFilenames();
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
    
    
// *---------------------------------------------------------------------------*
// | Learning and prediction
// *---------------------------------------------------------------------------*

    sequencesDirnames.erase(sequencesDirnames.begin());

    int numOfAnnotations = sizeof(g_AnnotationLabels)/sizeof(g_AnnotationLabels[0]);
    std::vector<const char*> annotationLabels (g_AnnotationLabels, g_AnnotationLabels + numOfAnnotations);
    
    Groundtruth gt;
    remedi::loadGroundtruth(sequences, gt);
    
    Detection dt;
    
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
    for (int t = 0; t < NUM_OF_SUBJECTS; t++)
    {
        std::cout << "Training in fold " << t << " (LOSOCV) .. " << std::endl;

        std::vector<CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr> pipelines (NUM_REPETITIONS);
        bool bSuccess = true;
        for (int r = 0; r < NUM_REPETITIONS; r++)
        {
            pipelines[r] = CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr(new CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>);
            bSuccess &= pipelines[r]->load("training_" + std::to_string(t) + "-" + std::to_string(r));
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

            std::cout << "Generating final models (" << NUM_REPETITIONS << "x) .." << std::endl;

            // Training eventually
            // Repeat it several times (because of the quantization stochasticity)
            boost::timer trTimer;
            for (int r = 0; r < NUM_REPETITIONS; r++)
            {
                pPipeline->train(); // use the best parameters found in the model selection
                pPipeline->save("training_" + std::to_string(t) + "-" + std::to_string(r));
            }
            float trTime = trTimer.elapsed();
            
            std::cout << "It took " << (valTime + trTime) << " (" << valTime << " + " << trTime << ") secs." << std::endl;
        }
    }
    
//    // Test recognition accuracy
//    for (int t = 0; t < NUM_OF_SUBJECTS; t++)
//    {
//        std::cout << "Testing (object recognition) in fold " << t << " (LOSOCV) .." << std::endl;
//
//        std::vector<CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr> pipelines (NUM_REPETITIONS);
//        bool bSuccess = true;
//        for (int r = 0; r < NUM_REPETITIONS; r++)
//        {
//            pipelines[r] = CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr(new CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>);
//            bSuccess &= pipelines[r]->load("training_" + std::to_string(t) + "-" + std::to_string(r));
//        }
//        
//        if (bSuccess)
//        {
//            std::cout << "Creating the sample of test cloudjects .." << std::endl;
//            boost::shared_ptr<std::list<MockCloudject::Ptr> > pCloudjectsTe (new std::list<MockCloudject::Ptr>);
//            remedi::getTestCloudjectsWithDescriptor(sequences, sequencesSids, t, gt, "pfhrgb250", *pCloudjectsTe);
//            std::cout << "Created a test sample of " << pCloudjectsTe->size() << " cloudjects." << std::endl;
//
//            for (int r = 0; r < NUM_REPETITIONS; r++)
//            {
//                // Compute
//                std::list<std::vector<int> > predictions; // (not used)
//                std::list<std::vector<float> > distsToMargin; // (not used)
//                
//                std::vector<float> osacc = pipelines[r]->predict(pCloudjectsTe, predictions, distsToMargin);
//                // Output
//                std::copy(osacc.begin(), osacc.end(), std::ostream_iterator<float>(std::cout, " "));
//                std::cout << std::endl;
//            }
//        }
//    }
    
    std::vector<std::vector<float> > detectionValParams;
    std::vector<float> interactionThresh, mvActorCorrespThresh;
    mvActorCorrespThresh += 5, 7.5, 10;
    interactionThresh += 1, 2, 3, 6, 8;
    detectionValParams += mvActorCorrespThresh, interactionThresh;
    
    std::vector<std::vector<float> > detectionValCombs;
    expandParameters<float>(detectionValParams, detectionValCombs);
    
    std::vector<std::vector<cv::Mat> > dtMonocularResults (NUM_OF_SUBJECTS, std::vector<cv::Mat>(NUM_REPETITIONS,cv::Mat(detectionValCombs.size(),pBgSeq->getNumOfViews(),CV_32FC3)));
    std::vector<std::vector<cv::Mat> > dtMultiviewResults (NUM_OF_SUBJECTS, std::vector<cv::Mat>(NUM_REPETITIONS,cv::Mat(detectionValCombs.size(),1,CV_32FC3)));
    
    cv::FileStorage fs; // store detection in disk to process them later
    
    for (int t = 0; t < NUM_OF_SUBJECTS; t++)
    {
        std::cout << "Testing (object detection) in fold " << t << " (LOSOCV) .." << std::endl;

        std::vector<CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr> pipelines (NUM_REPETITIONS);
        bool bSuccess = true;
        for (int r = 0; r < NUM_REPETITIONS; r++)
        {
            pipelines[r] = CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::Ptr(new CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>);
            bSuccess &= pipelines[r]->load("training_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r));
        }
        
        if (bSuccess)
        {
            std::cout << "Creating the sample of test cloudjects .." << std::endl;
            boost::shared_ptr<std::list<MockCloudject::Ptr> > pCloudjectsTe (new std::list<MockCloudject::Ptr>);
            remedi::getTestCloudjectsWithDescriptor(sequences, sequencesSids, t, gt, "pfhrgb250", *pCloudjectsTe);
            std::cout << "Created a training sample of " << pCloudjectsTe->size() << " cloudjects." << std::endl;
            
            std::vector<Sequence<ColorDepthFrame>::Ptr> sequencesTe;
            remedi::getTestSequences(sequences, sequencesSids, t, sequencesTe);
            
            // MONOCULAR
            for (int r = 0; r < NUM_REPETITIONS; r++)
            {
                for (int i = 0; i < detectionValCombs.size(); i++)
                {
                    sys.setMultiviewDetectionStrategy(DETECT_MONOCULAR);
                    sys.setMultiviewActorCorrespondenceThresh(detectionValCombs[i][0]);
                    sys.setInteractionThresh(detectionValCombs[i][1]);
                    sys.detect(sequencesTe, objectsLabels, gt, dt, pipelines[t]);
                    
                    std::vector<DetectionResult> dtResultsFold = sys.getDetectionResults();
                    for (int v = 0; v < dtResultsFold.size(); v++)
                        dtMonocularResults[t][r].at<cv::Vec3f>(i,v) = dtResultsFold[v].toVector();
                }
                
                fs.open("detection-monocular_" + boost::lexical_cast<std::string>(t) + "-" + boost::lexical_cast<std::string>(r) + ".yml", cv::FileStorage::WRITE);
                fs << "detections" << dtMonocularResults[t][r];
                fs.release();
            }
                
//                // MULTIVIEW
//                // TODO: proper way of normalising prior to late fusion
//                for (int i = 0; i < detectionValCombs.size(); i++)
//                {
//                    sys.setMultiviewDetectionStrategy(DETECT_MULTIVIEW);
//                    sys.setMultiviewLateFusionNormalization(true);
//                    sys.setMultiviewActorCorrespondenceThresh(detectionValCombs[i][0]);
//                    sys.setInteractionThresh(detectionValCombs[i][1]);
//                    sys.detect(sequencesTe, objectsLabels, gt, dt, pipelines[t]);
//                    
//                    int tp, fp, fn;
//                    sys.evaluate(sequencesTe, objectsLabels, gt, dt, tp, fp, fn);
//                }
            
        }
    }

    
    // Use the already trained classifiers and measure TP, FP, and FN over the sequences
//    
//    for (int t = 0; t < NUM_OF_SUBJECTS; t++)
//    {
////        ReMedi::
//        std::vector<Sequence<ColorDepthFrame>::Ptr> sequencesTe;
//        remedi::getTestSequences(sequences, sequencesSids, t, sequencesTe);
//        
//        sys.detect(sequencesTe, objectsLabels, gt, pipelines[t]);
//    }
    
	return 0;
}

int main()
{
    return run();
}
