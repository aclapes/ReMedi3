//
//  BoundingBoxesGenerator.h
//  bgs
//
//  Created by Oscar Amoros Huguet on 27/2/15.
//
//
#include <list>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#ifndef __bgs__BoundingBoxesGenerator__
typedef struct boundingBox{
    int xMin;
    int yMin;
    int xMax;
    int yMax;
}boundingBox;
typedef struct bbData{
    std::vector<boundingBox> boundingBoxes;
    cv::Mat mask;
    cv::Mat pixelsMask;
}bbData;
#define __bgs__BoundingBoxesGenerator__
#endif /* defined(__bgs__BoundingBoxesGenerator__) */

class BoundingBoxesGenerator {
    
private:
    int minNumPixels;
    int depthThreshold;
    int numCalls; //DEBUG
    
public:
    
    BoundingBoxesGenerator(int minPix, int depthThresh);
    
    bbData generate( cv::Mat sourceImg, cv::Mat depthImg );
    void setMinNumPixels(int minPix);
    int getMinNumPixels();
    
private:
    
    cv::Mat checkType(cv::Mat image);
    void expandObject(boundingBox& bbPoints,cv::Mat& modificableImg, cv::Mat depthImg, int x, int y, int& numPixels, cv::Mat& tempMask, int bbID);
    bool doOneBB(cv::Mat& modificableImg, cv::Mat depthImg, bbData& output);
    void paintBox(cv::Mat *mask, boundingBox bb);
};