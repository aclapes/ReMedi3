//
//  BoundingBoxesGenerator.cpp
//  bgs
//
//  Created by Oscar Amoros Huguet on 27/2/15.
//
//

#include "BoundingBoxesGenerator.h"

BoundingBoxesGenerator::BoundingBoxesGenerator(int minPix, int depthThresh){
    minNumPixels = minPix;
    depthThreshold = depthThresh;
    numCalls=0;
}

cv::Mat BoundingBoxesGenerator::checkType(cv::Mat image){
    
    // We only consider the case of GrayScale and RGB C2 and C4 cases are not considered
    if (image.type()<=6) {
        image.convertTo(image, CV_8UC1);
    }
    else {
        std::cout << "\nChangin the image to C1";
        cv::cvtColor(image, image, CV_RGB2GRAY);
        image.convertTo(image, CV_8UC1);
    }
    
    return image;
    
}

void BoundingBoxesGenerator::paintBox(cv::Mat *mask, boundingBox bb){
    
    for (int y=0; y+bb.yMin<bb.yMax; y++) {
        
        cv::Vec3b columnPix1 = mask->at<cv::Vec3b>(bb.yMin+y,bb.xMin);
        cv::Vec3b columnPix11 = mask->at<cv::Vec3b>(bb.yMin+y,bb.xMin+1);
        
        cv::Vec3b columnPix2 = mask->at<cv::Vec3b>(bb.yMin+y,bb.xMax);
        cv::Vec3b columnPix22 = mask->at<cv::Vec3b>(bb.yMin+y,bb.xMax-1);
        
        columnPix1.val[0] = 0;
        columnPix1.val[1] = 0;
        columnPix1.val[2] = 255;
        
        columnPix11.val[0] = 0;
        columnPix11.val[1] = 0;
        columnPix11.val[2] = 255;
        
        columnPix2.val[0] = 0;
        columnPix2.val[1] = 0;
        columnPix2.val[2] = 255;
        
        columnPix22.val[0] = 0;
        columnPix22.val[1] = 0;
        columnPix22.val[2] = 255;
        
        //std::cout << "\nyMin+y=" << bb.xMin+y << " xMin=" << bb.xMin;
        mask->at<cv::Vec3b>(bb.yMin+y,bb.xMin) = columnPix1;
        mask->at<cv::Vec3b>(bb.yMin+y,bb.xMin+1) = columnPix11;
        
        mask->at<cv::Vec3b>(bb.yMin+y,bb.xMax) = columnPix2;
        mask->at<cv::Vec3b>(bb.yMin+y,bb.xMax-1) = columnPix22;
        
    }
    for (int x=0; x+bb.xMin<bb.xMax; x++) {
        
        cv::Vec3b rowPix1 = mask->at<cv::Vec3b>(bb.yMin,bb.xMin+x);
        cv::Vec3b rowPix11 = mask->at<cv::Vec3b>(bb.yMin+1,bb.xMin+x);
        
        cv::Vec3b rowPix2 = mask->at<cv::Vec3b>(bb.yMax,bb.xMin+x);
        cv::Vec3b rowPix22 = mask->at<cv::Vec3b>(bb.yMax-1,bb.xMin+x);
        
        rowPix1.val[0] = 0;
        rowPix1.val[1] = 0;
        rowPix1.val[2] = 255;
        
        rowPix11.val[0] = 0;
        rowPix11.val[1] = 0;
        rowPix11.val[2] = 255;
        
        rowPix2.val[0] = 0;
        rowPix2.val[1] = 0;
        rowPix2.val[2] = 255;
        
        rowPix22.val[0] = 0;
        rowPix22.val[1] = 0;
        rowPix22.val[2] = 255;
        
        mask->at<cv::Vec3b>(bb.yMin, bb.xMin+x) = rowPix1;
        mask->at<cv::Vec3b>(bb.yMin+1, bb.xMin+x) = rowPix11;
        
        mask->at<cv::Vec3b>(bb.yMax, bb.xMin+x) = rowPix2;
        mask->at<cv::Vec3b>(bb.yMax-1, bb.xMin+x) = rowPix22;
    }
    
    
}

void BoundingBoxesGenerator::expandObject(boundingBox& bbPoints, cv::Mat& modificableImg, cv::Mat depthImg, int x, int y, int &numPixels, cv::Mat& tempMask, int bbID){
    
    // We set the point to 0 so next time we create one bounding box, this pixel will not be longuer used
    modificableImg.at<unsigned char>(y,x)=0;
    tempMask.at<unsigned char>(y,x)=bbID;
    
    for (int i=-1; i<=1; i++) for (int j=-1; j<=1; j++) {
        
        
        if (modificableImg.at<unsigned char>(y+i,x+j)!=0){
            int diff,depth1,depth2;
            
            // Now check the depth difference
            depth1=(depthImg.at<unsigned short>(y,x) /*>> 3*/);
            depth2=(depthImg.at<unsigned short>(y+i,x+j) /*>> 3*/);
            diff = (int)std::abs((float)depth1 - (float)depth2);
            //std::cout << "\nDEPTH 1=" << (int)(depthImg.at<unsigned short>(y,x) >> 3) << "DEPTH 2=" <<(int)(depthImg.at<unsigned short>(y+i,x+j) >> 3);
            // If the points are not too far in depth
            if ( diff<=depthThreshold ) {
                // Expand from the neighboor
                numPixels++;
                if (bbPoints.yMax < y+i) {
                    bbPoints.yMax = y+i;
                }
                if (bbPoints.yMin > y+i) {
                    bbPoints.yMin = y+i;
                }
                if (bbPoints.xMax < x+j) {
                    bbPoints.xMax = x+j;
                }
                if (bbPoints.xMin > x+j) {
                    bbPoints.xMin = x+j;
                }
                expandObject(bbPoints, modificableImg, depthImg, x+j, y+i, numPixels, tempMask, bbID);
            }
            
        }
        
    }
    
}

bool BoundingBoxesGenerator::doOneBB(cv::Mat& modificableImg, cv::Mat depthImg, bbData& output){
    
    boundingBox bbPoints;
    
    int ySize = modificableImg.rows;
    int xSize = modificableImg.cols;
    int x,y;
    int numPixels;
    bool end=false, found=true;
    cv::Mat tempMask(480,640,CV_8U);
    
    //numCalls++;
    
    //std::cout << "\n Generating a BB. Number of calls for this frame " << numCalls << "\n" ;
    
    numPixels=0;
    x=0; y=0;
    
    while (y<ySize && !end){
        while (x<xSize && !end) {
            
            unsigned char *pixel = modificableImg.ptr<unsigned char>(y,x);
            unsigned short *dpixel = depthImg.ptr<unsigned short>(y,x);
            
            if ( (int)(*pixel) != 0 && (int)(*dpixel) > 0){
                
                bbPoints.xMin = x;
                bbPoints.yMin = y;
                bbPoints.xMax = x;
                bbPoints.yMax = y;
                
                //EXPANDING OBJECT
                
                expandObject(bbPoints, modificableImg, depthImg, x, y, numPixels, tempMask, (output.boundingBoxes.size()+1));
                end=true;
            }
            x++;
        }
        y++;
        if (x==xSize && y==ySize){
            //std::cout << "\n End of x and y \n";
            found= false;
        }
        x=0;
        
    }
    
    // Here we check wether the boundig box has enough pixels in order to take it into account
    if (numPixels>=minNumPixels){
        
        // Here we sould modify the output
        output.boundingBoxes.push_back(bbPoints);
        /*cv::imshow("OneBlob", tempMask);
        cv::waitKey(1);*/
        output.pixelsMask = output.pixelsMask | tempMask;
        
    }
    tempMask.setTo(0);
    
    return found;
}


bbData BoundingBoxesGenerator::generate(cv::Mat sourceImg, cv::Mat depthImg){
    
    bbData output;
    
    cv::Mat modificableImg;
    
    // Initialize the bounding boxes vector
    
    // We want to erase the points for a given objetc so we can repeat the process without complex code for each bb.
    // But we don't want to modify the original image, so we copy it to modificableImg
    
    sourceImg = checkType(sourceImg);
    sourceImg.copyTo(modificableImg);
    cv::cvtColor(sourceImg, output.mask, CV_GRAY2BGR);
    
    // We are assuming that depthImg is C1 and 16bits
    
    //std::cout << "\nPixels mask size=" << output.pixelsMask.size();
    
    output.pixelsMask = cv::Mat(480, 640, CV_8U);
    output.pixelsMask.setTo(0);
    
    while (doOneBB(modificableImg, depthImg, output));
    
    numCalls=0;
    
    for (int i=0; i<(int)output.boundingBoxes.size(); i++) {
        
        paintBox(&(output.mask), output.boundingBoxes.at(i) );
        /*cv::imshow("BoundingBoxes", output.mask);
        cv::waitKey(10);*/
    }
    /*cv::imshow("BlobsAccepted", output.pixelsMask);
    cv::waitKey(10);*/
    
    return output;
    
}

void BoundingBoxesGenerator::setMinNumPixels(int minPix){
    
    minNumPixels=minPix;
    
}

int BoundingBoxesGenerator::getMinNumPixels(){
    return minNumPixels;
}