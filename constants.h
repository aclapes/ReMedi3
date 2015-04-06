//
//  constants.h
//  remedi2
//
//  Created by Albert Clapés on 30/08/14.
//
//

#ifndef remedi2_constants_h
#define remedi2_constants_h

//
// Constants
//

#ifdef __APPLE__
#define PARENT_PATH                         "../../../Data/"
#elif _WIN32 || _WIN64
#define PARENT_PATH                         "../../Data/"
#endif

#define SEQUENCES_SUBDIR                    "Sequences/"
#define RESULTS_SUBDIR                      "Results/"
#define KINECT_SUBSUBDIR                    "Kinects/"

#define FRAMES_COLOR_DIRNAME                "Color"
#define FRAMES_DEPTH_DIRNAME                "Depth"
#define FOREGROUND_GROUNDTRUTH_DIRNAME      "ForegroundGt"
static const char* g_Views[] = {"1","2"};

#define OBJECTLABELS_SUBDIR                 "ObjectLabels/"
#define OBJECTMODELS_SUBDIR                 "ObjectModels/"
static const char* g_ObjectsLabels[]    = {"dish","pillbox","book","tetrabrick","cup"};
static const char* g_AnnotationLabels[] = {"dish","pillbox","book","tetrabrick","cup","arms","others"};

#define NUM_OF_SUBJECTS                     14 // LOSOCV

#define DELAYS_FILENAME                     "delays.txt"

#define DEFAULT_FRAME                       2

#define Y_RESOLUTION                        480
#define X_RESOLUTION                        640

#define MIN_DEPTH                           700
#define MAX_DEPTH                           2300

// Interactive registerer-related constants
#define IR_VIS_WND_HEIGHT                   480
#define IR_VIS_WND_WIDTH                    640
#define IR_VIS_VP                           1
#define IR_VIS_HP                           2
#define IR_VIS_DIST                         -1.25 // -2 meters
#define IR_VIS_MARKER_RADIUS                0.015
#define IR_NUM_OF_POINTS                    -1

// Table modeler-related constants

#define TABLETOP_MASKS_DIRNAME              "TabletopMasks"

#define TM_LEAF_SIZE                        0.01
#define TM_NORMAL_RADIUS                    0.05
#define TM_SAC_ITERS                        200
#define TM_SAC_DIST_THRESH                  0.03
#define TM_TT_Y_OFFSET                      0.4 // tabletop y-dim offset
#define TM_INTERACTIVE_BORDER_DIST          0.7
#define TM_CONF_LEVEL                       99

// Background subtraction-related constants

#define FOREGROUND_MASKS_DIRNAME            "ForegroundMasks"

#define BS_NUM_OF_SAMPLES                   400
#define BS_MODALITY                         3
#define BS_LRATE                            -1
#define BS_MORPHOLOGY                       0

#define BS_K                                30
#define BS_BGRATIO                          0.99999
#define BS_VARGEN                           25

#define BB_MIN_PIXELS                       150
#define BB_DEPTH_THRESH                     30 //mm

// ... recognition-related ones (computed in an independent dataset)

#define CLOUDJECTS_DIRNAME                  "Cloudjects"
#define DESCRIPTIONS_DIRNAME                "Descriptions"

#define TRAINING_BLOB_SAMPLING_RATE         1 //0.01 //0.1

#define OR_PFHDESC_LEAFSIZE                 0.01
#define OR_PFHDESC_NORMAL_RADIUS            0.04
#define OR_PFHDESC_PFH_RADIUS               0.09

#define OR_CORRESPONDENCE_TOLERANCE         0.1

#define NUM_REPETITIONS                     3

static float g_Colors[][3] = {
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 1, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, .5, 0},
    {1, 0, .5},
    {.5, 1, 0},
    {0, 1, .5},
    {.5, 0, 1},
    {0, .5, 1},
    {.5, 1, 0},
    {.25, .5, 0},
    {0, .5, .25},
    {.5, .25, 0},
    {.5, 0, .25},
    {.25, 0, .5},
    {0, .25, .5}
};

//
// Enums
//

enum { COLOR = 0, DEPTH = 1, COLOR_WITH_SHADOWS = 2, COLORDEPTH = 3 };
enum { DESCRIPTION_FPFH, DESCRIPTION_PFHRGB };
enum { DETECT_MONOCULAR, DETECT_MULTIVIEW };

//#define DEBUG_TRAINING_CONSTRUCTION_OVERLAPS
//#define DEBUG_TRAINING_CONSTRUCTION_SELECTION
#define DEBUG_VISUALIZE_DETECTIONS

#endif
