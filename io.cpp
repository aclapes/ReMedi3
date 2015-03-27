//
//  io.cpp
//  remedi3
//
//  Created by Albert Clapés on 10/03/15.
//
//

#include "io.h"
#include <fstream>

#include <pcl/io/pcd_io.h>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

//
// Predictions
//

namespace fs = boost::filesystem;

GroundtruthRegion lineToAnnotationRegion(std::string line)
{
    std::vector<std::string> values;
    
    std::istringstream   linestream(line);
    for (std::string value; std::getline(linestream, value, ','); )
        values.push_back(value);
    
    // Blob id in the csv, to be able to filter by pixel value later
    // Blob bounding box as id and 4 coordinates: minx, miny, maxx, maxy
    int minX = std::stoi(values[2]) - 1;
    int minY = std::stoi(values[3]) - 1;
    int maxX = std::stoi(values[4]) - 1;
    int maxY = std::stoi(values[5]) - 1;
    
    GroundtruthRegion r (std::stoi(values[0]), cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1));
    r.setCategory(values[1]);
    return r;
}

ForegroundRegion lineToPredictionRegion(std::string line)
{
    std::vector<std::string> values;
    
    std::istringstream   linestream(line);
    for (std::string value; std::getline(linestream, value, ','); )
        values.push_back(value);
    
    // Blob id in the csv, to be able to filter by pixel value later
    // Blob bounding box as id and 4 coordinates: minx, miny, maxx, maxy
    int minX = std::stoi(values[1]) - 1;
    int minY = std::stoi(values[2]) - 1;
    int maxX = std::stoi(values[3]) - 1;
    int maxY = std::stoi(values[4]) - 1;
    
    ForegroundRegion r (std::stoi(values[0]), cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1));
    return r;
}

void remedi::io::readAnnotationRegions(std::string parent, std::string fid, std::map<std::string,std::map<std::string,GroundtruthRegion> >& annotations, bool bHeader)
{
    annotations.clear();
    
    std::string filepathPrefix = parent + "/" + fid;
    std::ifstream file (filepathPrefix + ".png.object.csv");
    
    // If it is supposed to be a header, discard it
    std::string line;
    if (bHeader) std::getline(file, line);
    
    // Each line is an annotation
    for ( ; std::getline(file, line); )
    {
        GroundtruthRegion r = lineToAnnotationRegion(line);
        
        std::string filepath = filepathPrefix + ".png." + r.getCategory() + ".png";
        if (fs::exists(fs::path(filepath)))
            r.setFilepath(filepath);
        else
            r.setFilepath(filepathPrefix + ".png." + r.getCategory() + std::to_string(r.getID()) + ".png");
        
        annotations[r.getCategory()][std::to_string(r.getID())] = r;
    }
    file.close();
}

void remedi::io::readPredictionRegions(std::string parent, std::string fid, std::vector<ForegroundRegion>& predictions, bool bHeader)
{
    predictions.clear();
    
    std::string filepath = parent + "/" + fid;
    std::ifstream file (filepath + ".csv");
    
    // If it is supposed to be a header, discard it
    std::string line;
    if (bHeader) std::getline(file, line);
    
    // Each line is a prediction
    int i = 1;
    for ( ; std::getline(file, line); )
    {
        ForegroundRegion r = lineToPredictionRegion(line);
        r.setFilepath(filepath + "-" + std::to_string(i) + ".png");
        
        predictions.push_back(r);
        i++;
    }
    file.close();
}

//
// Cloudjects
//

template<typename PointT>
void remedi::io::writePointCloud(std::string filepath, pcl::PointCloud<PointT>& pCloud, bool bBinary)
{
    pcl::PCDWriter writer;
    
    if (bBinary)
        writer.writeBinary(filepath, pCloud);
    else
        writer.writeASCII(filepath, pCloud);
}

void remedi::io::writeCloudjects(std::string parent, std::string fid, std::vector<Cloudject::Ptr> cloudjects, bool bBinary, bool bHeader)
{   
    std::ofstream ofs;
    ofs.open(parent + "/" + fid + ".csv", std::ofstream::out);
    
    if (bHeader)
        ofs << "object_id,minx,miny,maxx,maxy" << std::endl;
    
    for (int i = 0; i < cloudjects.size(); i++)
    {
        // Write pcd file, with filename composed by fname and region ID
        std::string filename = fid  + "-" + std::to_string(cloudjects[i]->getRegionID());
        
        writePointCloud(parent + "/" + filename + ".pcd", *(cloudjects[i]->getCloud()), bBinary);
        cv::imwrite(parent + "/" + filename + ".png", cloudjects[i]->getRegionMask());
        
        // perhaps no need to replicate
        ofs << cloudjects[i]->getForegroundRegion().toString(",") << std::endl;
    }
    ofs.close();
}


void remedi::io::readCloudjects(std::string parent, std::string fid, std::vector<ForegroundRegion> predictions, std::vector<Cloudject::Ptr>& cloudjects, bool bHeader)
{
    cloudjects.clear();
    
    pcl::PCDReader reader;
    for (int i = 0; i < predictions.size(); i++)
    {
        Cloudject::Ptr pCj (new Cloudject(predictions[i]));
        pCj->setRegionMask(cv::imread(parent + "/" + fid + "-" + std::to_string(predictions[i].getID()) + ".png", CV_LOAD_IMAGE_GRAYSCALE));
        reader.read(parent + "/" + fid + "-" + std::to_string(predictions[i].getID()) + ".pcd", *(pCj->getCloud()));
        
        cloudjects.push_back(pCj);
    }
}

void remedi::io::mockreadCloudjects(std::string parent, std::string fid, std::vector<ForegroundRegion> predictions, std::vector<MockCloudject::Ptr>& cloudjects, bool bHeader)
{
    cloudjects.clear();
    
    for (int i = 0; i < predictions.size(); i++)
    {
        MockCloudject::Ptr pMCj (new MockCloudject(predictions[i]));
        pMCj->setCloudPath(parent + "/" + fid + "-" + std::to_string(predictions[i].getID()) + ".pcd");
        
        cloudjects.push_back(pMCj);
    }
}

void remedi::io::readCloudjects(std::string parent, std::string fid, std::vector<Cloudject::Ptr>& cloudjects, bool bHeader)
{
    cloudjects.clear();
    
    std::vector<ForegroundRegion> predictions;
    readPredictionRegions(parent, fid, predictions, bHeader);
    
    readCloudjects(parent, fid, predictions, cloudjects, bHeader);
}

void remedi::io::mockreadCloudjects(std::string parent, std::string fid, std::vector<MockCloudject::Ptr>& cloudjects, bool bHeader)
{
    cloudjects.clear();
    
    std::vector<ForegroundRegion> predictions;
    readPredictionRegions(parent, fid, predictions, bHeader);
    
    mockreadCloudjects(parent, fid, predictions, cloudjects, bHeader);
}


void remedi::io::readCloudjectsWithDescriptor(std::string parent, std::string fid, std::vector<ForegroundRegion> predictions, std::string descriptorType, std::vector<Cloudject::Ptr>& cloudjects, bool bHeader)
{
    cloudjects.clear();
    
    readCloudjects(parent, fid, predictions, cloudjects, bHeader);
    
    assert (predictions.size() == cloudjects.size());
    
    pcl::PCDReader reader;
    for (int i = 0; i < predictions.size(); i++)
    {
        cloudjects[i]->setDescriptorType(descriptorType);
        
        if (descriptorType.compare("fpfh33") == 0)
        {
            pcl::PointCloud<pcl::FPFHSignature33>* pDescriptor = new pcl::PointCloud<pcl::FPFHSignature33>();
            reader.read(parent + "/" + fid + "-" + std::to_string(predictions[i].getID()) + "_fpfh33.pcd", *pDescriptor);
            cloudjects[i]->setDescriptor((void*) pDescriptor);
        }
        else if (descriptorType.compare("pfhrgb250") == 0)
        {
            pcl::PointCloud<pcl::PFHRGBSignature250>* pDescriptor = new pcl::PointCloud<pcl::PFHRGBSignature250>();
            reader.read(parent + "/" + fid + "-" + std::to_string(predictions[i].getID()) + "_pfhrgb250.pcd", *pDescriptor);
            cloudjects[i]->setDescriptor((void*) pDescriptor);
        }
    }
}

void remedi::io::mockreadCloudjectsWithDescriptor(std::string parent, std::string fid, std::vector<ForegroundRegion> predictions, std::string descriptorType, std::vector<MockCloudject::Ptr>& cloudjects, bool bHeader)
{
    cloudjects.clear();
    
    mockreadCloudjects(parent, fid, predictions, cloudjects, bHeader);
    
    assert (predictions.size() == cloudjects.size());
    
//    pcl::PCDReader reader;
    for (int i = 0; i < predictions.size(); i++)
    {
        cloudjects[i]->setDescriptorType(descriptorType);
        
        if (descriptorType.compare("fpfh33") == 0)
        {
            cloudjects[i]->setDescriptorPath(parent + "/" + fid + "-" + std::to_string(predictions[i].getID()) + "_fpfh33.pcd");
//            pcl::PointCloud<pcl::FPFHSignature33>* pDescriptor = new pcl::PointCloud<pcl::FPFHSignature33>();
//            reader.read(parent + "/" + fid + "-" + std::to_string(predictions[i].getID()) + "_fpfh33.pcd", *pDescriptor);
//            cloudjects[i]->describe("fpfh33", pDescriptor);
        }
        else if (descriptorType.compare("pfhrgb250") == 0)
        {
            cloudjects[i]->setDescriptorPath(parent + "/" + fid + "-" + std::to_string(predictions[i].getID()) + "_pfhrgb250.pcd");
//            pcl::PointCloud<pcl::PFHRGBSignature250>* pDescriptor = new pcl::PointCloud<pcl::PFHRGBSignature250>();
//            reader.read(parent + "/" + fid + "-" + std::to_string(predictions[i].getID()) + "_pfhrgb250.pcd", *pDescriptor);
//            cloudjects[i]->describe("pfhrgb250", pDescriptor);
        }
    }
}

//void remedi::io::mockreadCloudjects(std::string parent, std::string fid, std::vector<Cloudject>& cloudjects, bool bHeader)
//{
//    cloudjects.clear();
//    
//    std::vector<ForegroundRegion> predictions;
//    readPredictionRegions(parent, fid, predictions, bHeader);
//    
//    pcl::PCDReader reader;
//    for (int i = 0; i < predictions.size(); i++)
//    {
//        Cloudject cj (predictions[i]);
////        cj.setRegionMask(cv::imread(parent + "/" + fid + "-" + std::to_string(predictions[i].getID()) + ".png", CV_LOAD_IMAGE_GRAYSCALE));
//        // (the mock part is not to read the .pcd)
//        
//        cloudjects.push_back(cj);
//    }
//}

void remedi::io::writeCloudjectsDescription(std::string parent, std::string fid, std::vector<Cloudject::Ptr>& cloudjects, bool bBinary)
{
    std::vector<Cloudject::Ptr>::iterator it;
    for (it = cloudjects.begin(); it != cloudjects.end(); ++it)
    {
        // Write pcd file, with filename composed by fname and region ID
        std::string descriptorType = (*it)->getDescriptorType();
        std::string filename = fid  + "-" + std::to_string((*it)->getRegionID());
        
        if (descriptorType == "pfhrgb250")
        {
            writePointCloud(parent + "/" + filename + "_pfhrgb250.pcd", *((pcl::PointCloud<pcl::PFHRGBSignature250>*) (*it)->getDescriptor()), bBinary);
        }
        else if (descriptorType == "fpfh33")
        {
            writePointCloud(parent + "/" + filename + "_fpfh33.pcd", *((pcl::PointCloud<pcl::FPFHSignature33>*) (*it)->getDescriptor()), bBinary);
        }
    }
}

//
//void ReMedi::writeDescriptions(std::string parent, std::vector<Description> descriptions, bool bBinary)
//{
//    pcl::PCDWriter writer;
//    std::ofstream ofs;
//    
//    for (int i = 0; i < descriptions.size(); i++)
//    {
//        fs::path path (parent + "/");
//        
//        // Write pcd file
//        std::string filename = descriptions[i].fname  + "-" + std::to_string(i);
//        if (bBinary)
//            writer.writeBinary(path.string() + filename + ".pfhrgb250.pcd", *(descriptions[i].pDescriptor));
//        else
//            writer.writeASCII(path.string() + filename + ".pfhrgb250.pcd", *(descriptions[i].pDescriptor));
//        
//        // (Open) and write csv file
//        ofs.open(path.string() + filename + ".csv", std::ofstream::out);
//        
//        ofs << "sname,vname,fname,posx,posy,width,height,id" << std::endl;
//        ofs << descriptions[i].sname << "," << descriptions[i].vname << "," << descriptions[i].fname << ","
//        << descriptions[i].pos[0] << "," << descriptions[i].pos[1] << ","
//        << descriptions[i].size[0] << "," << descriptions[i].size[1] << ","
//        << descriptions[i].id << ",";
//        ofs << std::endl;
//        
//        // Iterate over the set of labels (at least 1)
//        std::set<std::pair<std::string,float> >::iterator it;
//        for (it = descriptions[i].labels.begin(); it != descriptions[i].labels.end(); it++)
//            ofs << (it == descriptions[i].labels.begin() ? "" : ",") << (*it).first;
//        ofs << std::endl;
//        
//        for (it = descriptions[i].labels.begin(); it != descriptions[i].labels.end(); it++)
//            ofs << (it == descriptions[i].labels.begin() ? "" : ",") << (*it).second;
//        ofs << std::endl;
//        
//        ofs.close(); // (close) csv written file
//    }
//}
//
//void ReMedi::readDescriptions(std::string parent, std::list<Description>& descriptions)
//{
//    descriptions.clear();
//    
//    pcl::PCDReader reader;
//    std::ifstream ifs;
//    
//    fs::directory_iterator it(parent);
//    fs::directory_iterator end;
//    for( ; it != end ; ++it )
//    {
//        if ( !fs::is_directory( *it ) && it->path().extension().string().compare(".csv") == 0)
//        {
//            Description d;
//            
//            // Read cloudject metadata (csv)
//            std::string headerLine, basicsLine, labelsLine, contentionsLine;
//            
//            ifs.open(it->path().string());
//            std::getline(ifs, headerLine); // discard header
//            std::getline(ifs, basicsLine); // actual first metadata line
//            std::getline(ifs, labelsLine);
//            std::getline(ifs, contentionsLine);
//            ifs.close();
//            
//            std::istringstream basicsLineStream(basicsLine);
//            std::istringstream labelsLineStream(labelsLine);
//            std::istringstream contentionsLineStream(contentionsLine);
//            
//            std::vector<std::string> basics, labels, contentions;
//            for (std::string v; std::getline(basicsLineStream, v, ','); )
//                basics.push_back(v);
//            for (std::string v; std::getline(labelsLineStream, v, ','); )
//                labels.push_back(v);
//            for (std::string v; std::getline(contentionsLineStream, v, ','); )
//                contentions.push_back(v);
//            
//            assert(labels.size() == contentions.size());
//            
//            // Format should be ( from writeDescriptions(...) ):
//            // "fid,bid,posx,posy,width,height,label1,[label2,...]"
//            d.sname = basics[0];
//            d.vname = basics[1];
//            d.fname = basics[2];
//            d.pos[0] = std::stoi(basics[3]);
//            d.pos[1] = std::stoi(basics[4]);
//            d.size[0] = std::stoi(basics[5]);
//            d.size[1] = std::stoi(basics[6]);
//            d.id = (unsigned short) std::stoi(basics[7]);
//            for (int i = 0; i < labels.size(); i++)
//                d.labels.insert( std::pair<std::string,float>(labels[i], std::stof(contentions[i])) );
//            
//            // Read description point cloud
//            d.pDescriptor = pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr(new pcl::PointCloud<pcl::PFHRGBSignature250>);
//            
//            std::string pfhrgbpath = it->path().parent_path().string() + "/" + it->path().stem().string() + ".pfhrgb250.pcd";
//            reader.read(pfhrgbpath, *(d.pDescriptor));
//            
//            // Return this descriptor after all
//            descriptions.push_back(d);
//        }
//    }
//}
//
//void ReMedi::readMockCloudjects(std::string parent, std::list<Cloudject>& cloudjects)
//{
//    cloudjects.clear();
//    
//    pcl::PCDReader reader;
//    std::ifstream ifs;
//    
//    fs::directory_iterator it(parent);
//    fs::directory_iterator end;
//    for( ; it != end ; ++it )
//    {
//        if ( !fs::is_directory( *it ) && it->path().extension().string().compare(".pcd") == 0)
//        {
//            // Read cloudject metadata (csv)
//            std::string csvpath = it->path().parent_path().string() + "/" + it->path().stem().string() + ".csv";
//            
//            // Process first line: fid,bid,posx,posy,label1,[label2,...]
//            std::string headerLine, basicsLine, labelsLine, contentionsLine;
//            
//            ifs.open(csvpath);
//            std::getline(ifs, headerLine); // discard header
//            std::getline(ifs, basicsLine); // actual first metadata line
//            std::getline(ifs, labelsLine);
//            std::getline(ifs, contentionsLine);
//            ifs.close();
//            
//            std::vector<std::string> basics, labels, contentions;
//            
//            std::istringstream basicLineStream(basicsLine);
//            std::istringstream labelsLineStream(labelsLine);
//            std::istringstream contentionsLineStream(contentionsLine);
//            
//            for (std::string v; std::getline(basicLineStream, v, ','); )
//                basics.push_back(v);
//            for (std::string v; std::getline(labelsLineStream, v, ','); )
//                labels.push_back(v);
//            for (std::string v; std::getline(contentionsLineStream, v, ','); )
//                contentions.push_back(v);
//            
//            assert (labels.size() == contentions.size());
//            
//            Cloudject cj;
//            cj.fname = basics[0];
//            cj.id = (unsigned short) std::stoi(basics[1]);
//            cj.pos[0] = std::stoi(basics[2]);
//            cj.pos[1] = std::stoi(basics[3]);
//            for (int i = 0; i < labels.size(); i++)
//                cj.labels.insert( std::pair<std::string,float>(labels[i], std::stof(contentions[i])) );
//            
//            // Read the mask
//            std::string pngpath = it->path().parent_path().string() + "/" + it->path().stem().string() + ".png";
//            cj.mask = cv::imread(pngpath, CV_LOAD_IMAGE_UNCHANGED);
//            
//            // Return this cj after all
//            cloudjects.push_back(cj);
//        }
//    }
//}
