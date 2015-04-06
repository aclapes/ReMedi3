//
//  Cloudject.h
//  remedi3
//
//  Created by Albert Clapés on 10/03/15.
//
//

#ifndef __remedi3__Cloudject__
#define __remedi3__Cloudject__

#include <stdio.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfhrgb.h>

#include "ForegroundRegion.hpp"

#include <set>
#include <string>
#include <list>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <pcl/io/pcd_io.h>

#include "conversion.h"
#include "constants.h"
#include "pclxtended.h"


class Cloudject
{
public:
    
    // Constructors
    
    Cloudject()
    : m_pCloud(new pcl::PointCloud<pcl::PointXYZRGB>),
      m_pDescriptor(NULL)
    {}
    
    Cloudject(ForegroundRegion r)
    : m_Region(r),
      m_pCloud(new pcl::PointCloud<pcl::PointXYZRGB>),
      m_pDescriptor(NULL)
    {
    
    }
    
    Cloudject(const Cloudject& rhs)
    {
        *this = rhs;
    }
    
    Cloudject& operator=(const Cloudject& rhs)
    {
        if (this != &rhs)
        {
            m_Region = rhs.m_Region;
            
            m_pCloud = rhs.m_pCloud;
            m_DescriptorType = rhs.m_DescriptorType;
            m_pDescriptor = rhs.m_pDescriptor;
            
            m_T = m_T;
        }
        return *this;
    }
    
    ~Cloudject()
    {
        release(); // cause this procedure can be executed from somewhere else
    }
    
    // Public methods
    
    ForegroundRegion getForegroundRegion()
    {
        return m_Region;
    }
    
    cv::Mat getRegionMask()
    {
        return m_Region.getMask();
    }
    
    void setRegionMask(cv::Mat mask)
    {
        m_Region.setMask(mask);
    }
    
    bool isRegionLabel(std::string label)
    {
        return m_Region.isLabel(label);
    }
    
    void addRegionLabel(std::string label)
    {
        m_Region.addLabel(label);
    }
    
    void setRegionLabels(std::set<std::string> labels)
    {
        m_Region.setLabels(labels);
    }
    
    std::set<std::string> getRegionLabels() const
    {
        return m_Region.getLabels();
    }
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& getCloud()
    {
        return m_pCloud;
    }
    
    ForegroundRegion getRegion()
    {
        return m_Region;
    }
    
    int getRegionID()
    {
        return m_Region.getID();
    }
    
    cv::Rect getRegionRect()
    {
        return m_Region.getRect();
    }
    
    std::string getDescriptorType()
    {
        return m_DescriptorType;
    }
    
    void setDescriptorType(std::string type)
    {
        m_DescriptorType = type;
    }
    
    void setCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud)
    {
        m_pCloud = pCloud;
    }
    
    void* getDescriptor()
    {
        return m_pDescriptor;
    }

    void setDescriptor(void* pDescriptor)
    {
        m_pDescriptor = pDescriptor;
    }
    
    int getNumOfDescriptorPoints()
    {
        if (m_DescriptorType.compare("fpfh33") == 0)
        {
            pcl::PointCloud<pcl::FPFHSignature33>* pDescriptor = (pcl::PointCloud<pcl::FPFHSignature33>*) m_pDescriptor;
            return pDescriptor->size();
        }
        else if (m_DescriptorType.compare("pfhrgb250") == 0)
        {
            pcl::PointCloud<pcl::PFHRGBSignature250>* pDescriptor = (pcl::PointCloud<pcl::PFHRGBSignature250>*) m_pDescriptor;
            return pDescriptor->size();
        }

    }
    
    void setRegistrationTransformation(Eigen::Matrix4f T)
    {
        m_T = T;
    }
    
    pcl::PointXYZ getCloudCentroid()
    {
        return computeCentroid(*m_pCloud);
//        int n = m_pCloud->points.size();
//        
//        double x = .0;
//        double y = .0;
//        double z = .0;
//        
//        int c = 0;
//        for (int i = 0; i < n; ++i)
//        {
//            if (m_pCloud->points[i].z > .0)
//            {
//                x += m_pCloud->points[i].x;
//                y += m_pCloud->points[i].y;
//                z += m_pCloud->points[i].z;
//                c++;
//            }
//        }
//        
//        return pcl::PointXYZ(x/c, y/c, z/c);
    }
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getRegisteredCloud()
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pCloud->height = m_pCloud->height;
        pCloud->width = m_pCloud->width;
        pCloud->resize(m_pCloud->height * m_pCloud->width);
        pCloud->is_dense = true;
        
        pcl::transformPointCloud(*m_pCloud, *pCloud, m_T);
        
        // Solving a issue: after transforming, (0,0,0,1) have changed. Since this
        // is undesired - e.g. for a centroid computation - re-set them to 0.
        for (int y = 0; y < m_pCloud->height; ++y) for (int x = 0; x < m_pCloud->width; ++x)
        {
            if ( m_Region.getMask().at<unsigned char>(y,x) == 0 )
                pCloud->at(x,y).x = pCloud->at(x,y).y = pCloud->at(x,y).z = 0;
        }
        
        return pCloud;
    }
    
    cv::Mat getRegisteredRegionMask()
    {
        cv::Mat registeredCloud;
        ColoredPointCloudToMat(getRegisteredCloud(), Y_RESOLUTION, X_RESOLUTION, m_Region.getRect().x, m_Region.getRect().y, registeredCloud);
        
        cv::Mat registeredCloudMask = registeredCloud > 0;
        return registeredCloudMask;
    }
    
    pcl::PointXYZ getRegisteredCloudCentroid()
    {
        Eigen::Vector4f _centroid = m_T * getCloudCentroid().getVector4fMap();
        
        pcl::PointXYZ centroid;
        centroid.getVector4fMap() = _centroid;

        return centroid;
    }
    
    boost::shared_ptr<pcl::VoxelGrid<pcl::PointXYZRGB> > getRegisteredGrid()
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloudTmp (new pcl::PointCloud<pcl::PointXYZRGB>);
        boost::shared_ptr<pcl::VoxelGrid<pcl::PointXYZRGB> > pGrid (new pcl::VoxelGrid<pcl::PointXYZRGB>);
        
        pclx::voxelize(getRegisteredCloud(), *pCloudTmp, *pGrid, Eigen::Vector3f(.01f,.01f,.01f));
        
        return pGrid;
    }
    
    void getCloudRectangle(pcl::PointXYZ& min, pcl::PointXYZ& max)
    {
        int n = m_pCloud->points.size();
        
        float minVal = std::numeric_limits<float>::min();
        float maxVal = std::numeric_limits<float>::max();
        min = pcl::PointXYZ(maxVal, maxVal, maxVal);
        max = pcl::PointXYZ(minVal, minVal, minVal);
        
        for (int i = 0; i < n; ++i)
        {
            if (m_pCloud->points[i].z > .0 && m_pCloud->points[i].z < 4096.f)
            {
                if (m_pCloud->points[i].x < min.x) min.x = m_pCloud->points[i].x;
                if (m_pCloud->points[i].y < min.y) min.y = m_pCloud->points[i].y;
                if (m_pCloud->points[i].z < min.z) min.z = m_pCloud->points[i].z;

                if (m_pCloud->points[i].x > max.x) max.x = m_pCloud->points[i].x;
                if (m_pCloud->points[i].y > max.y) max.y = m_pCloud->points[i].y;
                if (m_pCloud->points[i].z > max.z) max.z = m_pCloud->points[i].z;
            }
        }
    }
    
    void getRegisteredCloudRectangle(pcl::PointXYZ& min, pcl::PointXYZ& max)
    {
        pcl::PointXYZ minTmp, maxTmp;
        getCloudRectangle(minTmp, maxTmp);
        
        Eigen::Vector4f _minT = m_T * minTmp.getVector4fMap();
        Eigen::Vector4f _maxT = m_T * maxTmp.getVector4fMap();

        min.getVector4fMap() = _minT;
        max.getVector4fMap() = _maxT;
        
//        std::cout << min << " " << max << std::endl;
        float aux;
        if (min.x > max.x)
        {
            aux = min.x;
            min.x = max.x;
            max.x = aux;
        }
        if (min.y > max.y)
        {
            aux = min.y;
            min.y = max.y;
            max.y = aux;
        }
        if (min.z > max.z)
        {
            aux = min.z;
            min.z = max.z;
            max.z = aux;
        }
//        std::cout << min << " " << max << std::endl;
    }
    
    void describe(std::string descriptorType, std::vector<float> params)
    {
        // delete previous computed descriptor if any
        release();
        
        // keep record of the type
        setDescriptorType(descriptorType);
        
        // compute the new one
        if (descriptorType.compare("fpfh33") == 0)
        {
            pcl::PointCloud<pcl::FPFHSignature33>* pDescriptor = new pcl::PointCloud<pcl::FPFHSignature33>;
            describeFPFH(m_pCloud, params[0], params[1], params[2], *pDescriptor);
            setDescriptor((void*) pDescriptor);
        }
        else if (descriptorType.compare("pfhrgb250") == 0)
        {
            pcl::PointCloud<pcl::PFHRGBSignature250>* pDescriptor = new pcl::PointCloud<pcl::PFHRGBSignature250>;
            describePFHRGB(m_pCloud, params[0], params[1], params[2], *pDescriptor);
            setDescriptor((void*) pDescriptor);
        }
    }
    
    void describe(std::vector<float> params)
    {
        describe(m_DescriptorType, params);
    }

    
    typedef boost::shared_ptr<Cloudject> Ptr;
    

protected:
    
    // Protected methods
    
    void release()
    {
        if (!m_pCloud->empty())
            m_pCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        if (m_pDescriptor != NULL)
        {
            if (m_DescriptorType.compare("fpfh33") == 0)
            {
                ((pcl::PointCloud<pcl::FPFHSignature33>*) m_pDescriptor)->clear();
                delete ((pcl::PointCloud<pcl::FPFHSignature33>*) m_pDescriptor);
            }
            else if (m_DescriptorType.compare("pfhrgb250") == 0)
            {
                ((pcl::PointCloud<pcl::PFHRGBSignature250>*) m_pDescriptor)->clear();
                delete ((pcl::PointCloud<pcl::PFHRGBSignature250>*) m_pDescriptor);
            }
        }
        
        m_pDescriptor = NULL;
    }
    
private:
    
    ForegroundRegion                                m_Region;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr          m_pCloud;
    void*                                           m_pDescriptor;
    std::string                                     m_DescriptorType;
    
    Eigen::Matrix4f                                 m_T;

    // Private methods
    
    void describeFPFH(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud,
                        float leafSize, float normalRadius, float pfhrgbRadius,
                        pcl::PointCloud<pcl::FPFHSignature33>& descriptor)
    {
        //
        // Downsampling
        //
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pFiltCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        
        pcl::VoxelGrid<pcl::PointXYZRGB> vox;
        vox.setInputCloud(pCloud);
        vox.setLeafSize(leafSize, leafSize, leafSize);
        vox.filter(*pFiltCloud);
        
        pCloud.swap(pFiltCloud);
        
        //
        // Normals preprocess
        //
        
        // Create the normal estimation class, and pass the input dataset to it
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setInputCloud (pCloud);
        
        // Create an empty kdtree representation, and pass it to the normal estimation object.
        // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>());
        ne.setSearchMethod (tree);
        
        // Output datasets
        pcl::PointCloud<pcl::Normal>::Ptr pNormals (new pcl::PointCloud<pcl::Normal>);
        
        // Use all neighbors in a sphere of radius 3cm
        ne.setRadiusSearch (normalRadius);
        
        // Compute the features
        ne.compute (*pNormals);
        
        //
        // FPFH description extraction
        //
        
        pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh;
        fpfh.setInputCloud (pCloud);
        fpfh.setInputNormals (pNormals);
        // alternatively, if cloud is of tpe PointNormal, do pfhrgb.setInputNormals (cloud);
        
        // Create an empty kdtree representation, and pass it to the FPFH estimation object.
        // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
        tree = pcl::search::KdTree<pcl::PointXYZRGB>::Ptr(new pcl::search::KdTree<pcl::PointXYZRGB>);
        fpfh.setSearchMethod (tree);
        
        // Output datasets
        // * initialize outside
        
        // Use all neighbors in a sphere of radius 5cm
        // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
        fpfh.setRadiusSearch (pfhrgbRadius);
        
        // Compute the features
        fpfh.compute (descriptor);
    }
    
    void describePFHRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud,
                        float leafSize, float normalRadius, float pfhrgbRadius,
                        pcl::PointCloud<pcl::PFHRGBSignature250>& descriptor)
    {
        //
        // Downsampling
        //
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pFiltCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        
        pcl::VoxelGrid<pcl::PointXYZRGB> vox;
        vox.setInputCloud(pCloud);
        vox.setLeafSize(leafSize, leafSize, leafSize);
        vox.filter(*pFiltCloud);
        
        pCloud.swap(pFiltCloud);
        
        //
        // Normals preprocess
        //
        
        // Create the normal estimation class, and pass the input dataset to it
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setInputCloud (pCloud);
        
        // Create an empty kdtree representation, and pass it to the normal estimation object.
        // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>());
        ne.setSearchMethod (tree);
        
        // Output datasets
        pcl::PointCloud<pcl::Normal>::Ptr pNormals (new pcl::PointCloud<pcl::Normal>);
        
        // Use all neighbors in a sphere of radius 3cm
        ne.setRadiusSearch (normalRadius);
        
        // Compute the features
        ne.compute (*pNormals);
        
        //
        // FPFH description extraction
        //
        
        pcl::PFHRGBEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHRGBSignature250> pfhrgb;
        pfhrgb.setInputCloud (pCloud);
        pfhrgb.setInputNormals (pNormals);
        // alternatively, if cloud is of tpe PointNormal, do pfhrgb.setInputNormals (cloud);
        
        // Create an empty kdtree representation, and pass it to the FPFH estimation object.
        // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
        tree = pcl::search::KdTree<pcl::PointXYZRGB>::Ptr(new pcl::search::KdTree<pcl::PointXYZRGB>);
        pfhrgb.setSearchMethod (tree);
        
        // Output datasets
        // * initialize outside
        
        // Use all neighbors in a sphere of radius 5cm
        // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
        pfhrgb.setRadiusSearch (pfhrgbRadius);
        
        // Compute the features
        pfhrgb.compute (descriptor);
    }
};

class MockCloudject : public Cloudject
{
public:
    // Constructors
    
    MockCloudject() : Cloudject()
    {}
    
    MockCloudject(ForegroundRegion r) : Cloudject(r)
    {}
    
    MockCloudject(const MockCloudject& rhs) : Cloudject(rhs)
    {
        *this = rhs;
    }
    
    MockCloudject(const Cloudject& rhs) : Cloudject(rhs)
    {
        *this = rhs;
    }
    
    MockCloudject& operator=(const MockCloudject& rhs)
    {
        if (this != &rhs)
        {
            Cloudject::operator=(rhs);
            m_CloudPath = rhs.m_CloudPath;
            m_DescriptorPath = rhs.m_DescriptorPath;
        }
        return *this;
    }
    
    MockCloudject& operator=(const Cloudject& rhs)
    {
        if (this != &rhs)
            Cloudject::operator=(rhs);
        return *this;
    }
    
    void setCloudPath(std::string path)
    {
        m_CloudPath = path;
    }
    
    void setDescriptorPath(std::string path)
    {
        m_DescriptorPath = path;
    }
    
    void allocate()
    {
        pcl::PCDReader reader;
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        if (getCloud()->empty())
        {
            reader.read(m_CloudPath, *pCloud);
        }
        
        if (getDescriptor() == NULL)
        {
            std::string type = getDescriptorType();
            if (type.compare("fpfh33") == 0)
            {
                pcl::PointCloud<pcl::FPFHSignature33>* pDescriptor = new pcl::PointCloud<pcl::FPFHSignature33>();
                reader.read(m_DescriptorPath, *pDescriptor);
                setDescriptor((void*) pDescriptor);
            }
            else if (type.compare("pfhrgb250") == 0)
            {
                pcl::PointCloud<pcl::PFHRGBSignature250>* pDescriptor = new pcl::PointCloud<pcl::PFHRGBSignature250>();
                reader.read(m_DescriptorPath, *pDescriptor);
                setDescriptor((void*) pDescriptor);
            }
        }
    }
    
    void release()
    {
        Cloudject::release();
    }
    
    typedef boost::shared_ptr<MockCloudject> Ptr;
    
private:
    
    std::string                                     m_CloudPath;
    std::string                                     m_DescriptorPath;
};

#endif /* defined(__remedi3__Cloudject__) */
