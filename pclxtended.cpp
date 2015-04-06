//
//  pclxtended.cpp
//  remedi3
//
//  Created by Albert Clapés on 06/04/15.
//
//

#include "pclxtended.h"

pclx::Rectangle3D pclx::rectangleIntersection(pclx::Rectangle3D a, pclx::Rectangle3D b)
{
    float aib_min_x = std::max(a.min.x, b.min.x);
    float aib_min_y = std::max(a.min.y, b.min.y);
    float aib_min_z = std::max(a.min.z, b.min.z);
    
    float aib_max_x = std::min(a.max.x, b.max.x);
    float aib_max_y = std::min(a.max.y, b.max.y);
    float aib_max_z = std::min(a.max.z, b.max.z);
    
    pclx::Rectangle3D aib;
    aib.min = pcl::PointXYZ(aib_min_x, aib_min_y, aib_min_z);
    aib.max = pcl::PointXYZ(aib_max_x, aib_max_y, aib_max_z);
    
    return aib;
}

// Auxiliary function for varianceSubsampling(...)
float pclx::rectangleOverlap(pclx::Rectangle3D a, pclx::Rectangle3D b)
{
    pclx::Rectangle3D aib = pclx::rectangleIntersection(a,b);
    
    float overlap =  aib.area() / (a.area() + b.area() - aib.area());
    
    return overlap;
}

// Auxiliary function for getTrainingCloudjects(...)
// Check the amount of inclusion of "a" in "b"
float pclx::rectangleInclusion(pclx::Rectangle3D a, pclx::Rectangle3D b)
{
    pclx::Rectangle3D aib = pclx::rectangleIntersection(a,b);
    
    //    float aibArea = aib.area();
    //    float aArea = a.area();
    //    float bArea = b.area();
    float inclusion = aib.area() / std::min(a.area(),b.area());
    
    return inclusion;
}

void pclx::findCorrespondences(std::vector<std::vector<pcl::PointXYZ> > positions, float tol, std::vector<std::vector<std::pair<std::pair<int,int>,pcl::PointXYZ> > >& correspondences)
{
    correspondences.clear();
    
    // Keep already made assignations, to cut search paths
    std::vector<std::vector<bool> > assignations (positions.size());
    for (int v = 0; v < positions.size(); v++)
        assignations[v].resize(positions[v].size(), false);
    
    // Cumbersome internal (from this function) structure:
    // Vector of chains
    // A chain is a list of points
    // These points have somehow attached the 'view' and the 'index in the view'.
    for (int v = 0; v < positions.size(); v++)
    {
        for (int i = 0; i < positions[v].size(); i++)
        {
            if (!assignations[v][i])
            {
                std::vector<std::pair<std::pair<int,int>,pcl::PointXYZ> > chain; // points chain
                chain.push_back( std::pair<std::pair<int,int>,pcl::PointXYZ>(std::pair<int,int>(v,i), positions[v][i]) );
                assignations[v][i] = true;
                
                if (tol > 0)
                    findNextCorrespondence(positions, assignations, v+1, tol, chain); // recursion
                
                correspondences.push_back(chain);
            }
        }
    }
}


void pclx::findNextCorrespondence(std::vector<std::vector<pcl::PointXYZ> >& positions, std::vector<std::vector<bool> >& assignations, int v, float tol, std::vector<std::pair<std::pair<int,int>,pcl::PointXYZ> >& chain)
{
    if (v == positions.size())
    {
        return;
    }
    else
    {
        int candidateIdx = -1;
        bool bMoreThanOneCandidate = false;
        for (int j = 0; (j < positions[v].size()) && !bMoreThanOneCandidate; j++)
        {
            if (!assignations[v][j])
            {
                pcl::PointXYZ q = positions[v][j]; // candidate
                
                // To be a strong candidate is to fulfill the tolerance condition
                // for all the chain's former points
                bool bStrongCandidate = true;
                for (int i = 0; i < chain.size(); i++)
                {
                    pcl::PointXYZ p = chain[i].second;
                    float d = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2) + pow(p.z - q.z, 2));
                    
                    bStrongCandidate &= (d <= tol);
                }
                // If strong, nominate as temporary candidate
                if (bStrongCandidate)
                {
                    if (candidateIdx < 0) candidateIdx = j;
                    else bMoreThanOneCandidate = true;
                }
            }
        }
        // If one (and just one) candidate was found
        if (candidateIdx >= 0 && !bMoreThanOneCandidate)
        {
            chain.push_back( std::pair<std::pair<int,int>,pcl::PointXYZ>(std::pair<int,int>(v,candidateIdx), positions[v][candidateIdx]) );
            assignations[v][candidateIdx] = true;
        }
        
        findNextCorrespondence(positions, assignations, v+1, tol, chain);
    }
}

template<typename PointT>
void pclx::voxelize(typename pcl::PointCloud<PointT>::Ptr pCloud, pcl::PointCloud<PointT>& vloud, pcl::VoxelGrid<PointT>& vox, Eigen::Vector3f leafSize)
{
    vox.setInputCloud(pCloud);
    vox.setLeafSize(leafSize.x(), leafSize.y(), leafSize.z());
    
    // Reading errors are at (x,y,0), so filter z=0 values
    vox.setFilterFieldName("z");
    vox.setFilterLimits(-std::numeric_limits<float>::min(),std::numeric_limits<float>::min());
    vox.setFilterLimitsNegative(true);
    
    vox.setSaveLeafLayout(true); // the most important thing!
    vox.filter(vloud);
}

template<typename PointT>
float pclx::computeInclusion3d(typename pcl::PointCloud<PointT>::Ptr pCloud1, typename pcl::PointCloud<PointT>::Ptr pCloud2, Eigen::Vector3f leafSize)
{
    pcl::VoxelGrid<PointT> vox1, vox2;
    pcl::PointCloud<PointT> vloud1, vloud2;
    
    voxelize(pCloud1, vloud1, vox1, leafSize);
    voxelize(pCloud2, vloud2, vox2, leafSize);
    
    return pclx::computeInclusion3d(vox1, vox2);
}

template<typename PointT>
float pclx::computeInclusion3d(pcl::VoxelGrid<PointT>& vox1, pcl::VoxelGrid<PointT>& vox2)
{
    int voxels1, voxels2, voxelsI;
    pclx::countIntersections3d(vox1, vox2, voxels1, voxels2, voxelsI);
    
    float inc = ((float) voxelsI) / std::min(voxels1, voxels2);
    return inc;
}

template<typename PointT>
float pclx::computeOverlap3d(typename pcl::PointCloud<PointT>::Ptr pCloud1, typename pcl::PointCloud<PointT>::Ptr pCloud2, Eigen::Vector3f leafSize)
{
    pcl::VoxelGrid<PointT> vox1, vox2;
    pcl::PointCloud<PointT> vloud1, vloud2;
    
    voxelize(pCloud1, vloud1, vox1, leafSize);
    voxelize(pCloud2, vloud2, vox2, leafSize);
    
    return pclx::computeOverlap3d(vox1, vox2);
}

template<typename PointT>
float pclx::computeOverlap3d(pcl::VoxelGrid<PointT>& vox1, pcl::VoxelGrid<PointT>& vox2)
{
    int voxels1, voxels2, voxelsI;
    pclx::countIntersections3d(vox1, vox2, voxels1, voxels2, voxelsI);
    
    float ovl = ((float) voxelsI) / (voxels1 + voxels2 - voxelsI);
    return ovl;
}

template<typename PointT>
void pclx::countIntersections3d(pcl::VoxelGrid<PointT>& vox1, pcl::VoxelGrid<PointT>& vox2, int& voxels1, int& voxels2, int& voxelsI)
{
    voxels1 = voxels2 = voxelsI = 0;
    
    Eigen::Vector3i min1 = vox1.getMinBoxCoordinates();
    Eigen::Vector3i min2 = vox2.getMinBoxCoordinates();
    Eigen::Vector3i max1 = vox1.getMaxBoxCoordinates();
    Eigen::Vector3i max2 = vox2.getMaxBoxCoordinates();
    
    Eigen::Vector3i minU (min1.x() < min2.x() ? min1.x() : min2.x(),
                          min1.y() < min2.y() ? min1.y() : min2.y(),
                          min1.z() < min2.z() ? min1.z() : min2.z());
    Eigen::Vector3i maxU (max1.x() < max2.x() ? max1.x() : max2.x(),
                          max1.y() < max2.y() ? max1.y() : max2.y(),
                          max1.z() < max2.z() ? max1.z() : max2.z());
    
    for (int i = minU.x(); i < maxU.x(); i++)
    {
        for (int j = minU.y(); j < maxU.x(); j++)
        {
            for (int k = minU.z(); k < maxU.z(); k++)
            {
                int idx1 = vox1.getCentroidIndexAt(Eigen::Vector3i(i,j,k));
                int idx2 = vox2.getCentroidIndexAt(Eigen::Vector3i(i,j,k));
                
                // Not the most readable if-structure but for the sake of
                // efficiency
                if (idx1 != -1)
                {
                    if (idx2 != -1)
                    {
                        voxelsI++;
                        voxels2++;
                    }
                    
                    voxels1++;
                }
                else if (idx2 != -1)
                {
                    if (idx1 != -1)
                    {
                        voxelsI++;
                        voxels1++;
                    }
                    
                    voxels2++;
                }
            }
        }
    }
}


template void pclx::voxelize(pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud, pcl::PointCloud<pcl::PointXYZ>& vloud, pcl::VoxelGrid<pcl::PointXYZ>& vox, Eigen::Vector3f leafSize);
template void pclx::voxelize( pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud, pcl::PointCloud<pcl::PointXYZRGB>& vloud, pcl::VoxelGrid<pcl::PointXYZRGB>& vox, Eigen::Vector3f leafSize);

template float pclx::computeOverlap3d<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud1,  pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud2, Eigen::Vector3f leafSize);
template float pclx::computeOverlap3d<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud2, Eigen::Vector3f leafSize);
template float pclx::computeOverlap3d<pcl::PointXYZ>(pcl::VoxelGrid<pcl::PointXYZ>& vox1, pcl::VoxelGrid<pcl::PointXYZ>& vox2);
template float pclx::computeOverlap3d<pcl::PointXYZRGB>(pcl::VoxelGrid<pcl::PointXYZRGB>& vox1, pcl::VoxelGrid<pcl::PointXYZRGB>& vox2);

template float pclx::computeInclusion3d<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud1,  pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud2, Eigen::Vector3f leafSize);
template float pclx::computeInclusion3d<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud2, Eigen::Vector3f leafSize);
template float pclx::computeInclusion3d<pcl::PointXYZ>(pcl::VoxelGrid<pcl::PointXYZ>& vox1, pcl::VoxelGrid<pcl::PointXYZ>& vox2);
template float pclx::computeInclusion3d<pcl::PointXYZRGB>(pcl::VoxelGrid<pcl::PointXYZRGB>& vox1, pcl::VoxelGrid<pcl::PointXYZRGB>& vox2);

template void pclx::countIntersections3d(pcl::VoxelGrid<pcl::PointXYZ>& vox1, pcl::VoxelGrid<pcl::PointXYZ>& vox2, int& voxels1, int& voxels2, int& voxelsI);
template void pclx::countIntersections3d(pcl::VoxelGrid<pcl::PointXYZRGB>& vox1, pcl::VoxelGrid<pcl::PointXYZRGB>& vox2, int& voxels1, int& voxels2, int& voxelsI);
