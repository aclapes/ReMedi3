//
//  io.h
//  remedi3
//
//  Created by Albert Clapés on 10/03/15.
//
//

#ifndef __remedi3__io__
#define __remedi3__io__

#include <stdio.h>

#include "GroundtruthRegion.hpp"
#include "ForegroundRegion.hpp"
#include "Cloudject.hpp"
#include <map>

namespace remedi
{
    namespace io
    {
        // Frame-level input/output functions
        void readAnnotationRegions(std::string parent, std::string fid,  std::map<std::string,std::map<std::string,GroundtruthRegion> >& annotations, bool bHeader = false);
        void readPredictionRegions(std::string parent, std::string fid, std::vector<ForegroundRegion>& predictions, bool bHeader = false);
        
        template<typename PointT>
        void writePointCloud(std::string filepath, pcl::PointCloud<PointT>& pCloud, bool bBinary = false);
        
        void writeCloudjects(std::string parent, std::string fid, std::vector<Cloudject::Ptr> cloudjects, bool bBinary = false, bool bHeader = false);

        void readCloudjects(std::string parent, std::string fid, std::vector<ForegroundRegion> predictions, std::vector<Cloudject::Ptr>& cloudjects, bool bHeader = false);
        void mockreadCloudjects(std::string parent, std::string fid, std::vector<ForegroundRegion> predictions, std::vector<MockCloudject::Ptr>& cloudjects, bool bHeader = false);
        
        void readCloudjects(std::string parent, std::string fid, std::vector<Cloudject::Ptr>& cloudjects, bool bHeader = false);
        void mockreadCloudjects(std::string parent, std::string fid, std::vector<MockCloudject::Ptr>& cloudjects, bool bHeader = false);
        
        void readCloudjectsWithDescriptor(std::string parent, std::string fid,  std::vector<ForegroundRegion> predictions, std::string descriptorType, std::vector<Cloudject::Ptr>& cloudjects, bool bHeader = false);
        void mockreadCloudjectsWithDescriptor(std::string parent, std::string fid,  std::vector<ForegroundRegion> predictions, std::string descriptorType, std::vector<MockCloudject::Ptr>& cloudjects, bool bHeader = false);
        
        void writeCloudjectsDescription(std::string parent, std::string fid, std::vector<Cloudject::Ptr>& cloudjects, bool bBinary = false);
        
//        static void extractDescriptions(std::list<Cloudject> cloudjects, float leafSize, float normalRadius, float pfhRadius, std::vector<Description>& descriptions, bool bVerbose = false);
//        static void writeDescriptions(std::string path, std::vector<Description> descriptions, bool bBinary = false);
//        static void readDescriptions(std::string path, std::list<Description>& descriptions);
//        void readMockCloudjects(std::string path, std::vector<Cloudject>& cloudjects);
    }
}

#endif /* defined(__remedi3__io__) */
