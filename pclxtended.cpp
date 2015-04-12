//
//  pclxtended.cpp
//  remedi3
//
//  Created by Albert Clapés on 06/04/15.
//
//

#include "pclxtended.h"
#include <pcl/segmentation/extract_clusters.h>

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

void pclx::findCorrespondencesBasedOnInclusion(std::vector<std::vector<VoxelGridPtr> > grids, float tol, std::vector<std::vector<std::pair<std::pair<int,int>,VoxelGridPtr> > >& correspondences)
{
    correspondences.clear();
    
    // Keep already made assignations, to cut search paths
    std::vector<std::vector<bool> > assignations (grids.size());
    for (int v = 0; v < grids.size(); v++)
        assignations[v].resize(grids[v].size(), false);
    
    for (int v = 0; v < grids.size(); v++)
    {
        for (int i = 0; i < grids[v].size(); i++)
        {
            if (!assignations[v][i])
            {
                std::vector<std::pair<std::pair<int,int>,VoxelGridPtr> > chain; // points chain
                chain.push_back( std::pair<std::pair<int,int>,VoxelGridPtr>(std::pair<int,int>(v,i), grids[v][i]) );
                assignations[v][i] = true;
                
                if (tol >= 0)
                    findNextCorrespondenceBasedOnInclusion(grids, assignations, v+1, tol, chain); // recursion
                
                correspondences.push_back(chain);
            }
        }
    }
}

void pclx::findNextCorrespondenceBasedOnInclusion(std::vector<std::vector<VoxelGridPtr> >& grids, std::vector<std::vector<bool> >& assignations, int v, float tol, std::vector<std::pair<std::pair<int,int>,VoxelGridPtr> >& chain)
{
    if (v == grids.size())
    {
        return;
    }
    else
    {
        std::vector<std::pair<std::pair<int,int>,VoxelGridPtr> > chainTmp;
        for (int j = 0; j < grids[v].size(); j++)
        {
            if (!assignations[v][j])
            {
                VoxelGridPtr q = grids[v][j]; // candidate
                
                for (int i = 0; i < chain.size(); i++)
                {
                    VoxelGridPtr p = chain[i].second;
                    float inc = pclx::computeInclusion3d(*q,*p);
                    
                    if (inc >= tol)
                    {
                        chainTmp.push_back( std::pair<std::pair<int,int>,VoxelGridPtr>(std::pair<int,int>(v,j), grids[v][j]) );
                        assignations[v][j] = true;
                    }
                }
            }
        }
        
        chain.insert(chain.end(), chainTmp.begin(), chainTmp.end());
        
        findNextCorrespondenceBasedOnInclusion(grids, assignations, v+1, tol, chain);
    }
}

void pclx::findCorrespondencesBasedOnOverlap(std::vector<std::vector<VoxelGridPtr> > grids, float tol, std::vector<std::vector<std::pair<std::pair<int,int>,VoxelGridPtr> > >& correspondences)
{
    correspondences.clear();
    
    // Keep already made assignations, to cut search paths
    std::vector<std::vector<bool> > assignations (grids.size());
    for (int v = 0; v < grids.size(); v++)
        assignations[v].resize(grids[v].size(), false);
    
    for (int v = 0; v < grids.size(); v++)
    {
        for (int i = 0; i < grids[v].size(); i++)
        {
            if (!assignations[v][i])
            {
                std::vector<std::pair<std::pair<int,int>,VoxelGridPtr> > chain; // points chain
                chain.push_back( std::pair<std::pair<int,int>,VoxelGridPtr>(std::pair<int,int>(v,i), grids[v][i]) );
                assignations[v][i] = true;
                
                if (tol >= 0)
                    findNextCorrespondenceBasedOnOverlap(grids, assignations, v+1, tol, chain); // recursion
                
                correspondences.push_back(chain);
            }
        }
    }
}

void pclx::findNextCorrespondenceBasedOnOverlap(std::vector<std::vector<VoxelGridPtr> >& grids, std::vector<std::vector<bool> >& assignations, int v, float tol, std::vector<std::pair<std::pair<int,int>,VoxelGridPtr> >& chain)
{
    if (v == grids.size())
    {
        return;
    }
    else
    {
        std::vector<std::pair<std::pair<int,int>,VoxelGridPtr> > chainTmp;
        for (int j = 0; j < grids[v].size(); j++)
        {
            if (!assignations[v][j])
            {
                VoxelGridPtr q = grids[v][j]; // candidate
                
                for (int i = 0; i < chain.size(); i++)
                {
                    VoxelGridPtr p = chain[i].second;
                    float ovl = pclx::computeOverlap3d(*q,*p);
                    
                    if (ovl >= tol)
                    {
                        chainTmp.push_back( std::pair<std::pair<int,int>,VoxelGridPtr>(std::pair<int,int>(v,j), grids[v][j]) );
                        assignations[v][j] = true;
                    }
                }
            }
        }
        
        chain.insert(chain.end(), chainTmp.begin(), chainTmp.end());
        
        findNextCorrespondenceBasedOnOverlap(grids, assignations, v+1, tol, chain);
    }
}

template<typename PointT>
void pclx::downsample(typename pcl::PointCloud<PointT>::Ptr pCloudIn, Eigen::Vector3f leafSize, pcl::PointCloud<PointT>& cloudOut)
{
    pcl::VoxelGrid<PointT> vox;
    vox.setInputCloud(pCloudIn);
    vox.setLeafSize(leafSize.x(), leafSize.y(), leafSize.z());
    
    vox.filter(cloudOut);
}

template<typename PointT>
void pclx::voxelize(typename pcl::PointCloud<PointT>::Ptr pCloud, Eigen::Vector3f leafSize, pcl::PointCloud<PointT>& vloud, pcl::VoxelGrid<PointT>& vox)
{
    vox.setInputCloud(pCloud);
    vox.setLeafSize(leafSize.x(), leafSize.y(), leafSize.z());

    vox.setSaveLeafLayout(true); // the most important thing!
    vox.filter(vloud);
}

template<typename PointT>
void voxelize(std::vector<typename pcl::PointCloud<PointT>::Ptr> clouds, Eigen::Vector3f leafSize, std::vector<typename pcl::PointCloud<PointT>::Ptr>& vlouds, std::vector<typename boost::shared_ptr<pcl::VoxelGrid<PointT> > >& grids)
{
    grids.resize(clouds.size());
    for (int i = 0; i < clouds.size(); i++)
    {
        typename pcl::PointCloud<PointT>::Ptr pVloud (new pcl::PointCloud<PointT>);
        grids[i] = typename boost::shared_ptr<pcl::VoxelGrid<PointT> > (new pcl::VoxelGrid<PointT>);
        pclx::voxelize(clouds[i], leafSize, *pVloud, *(grids[i]));
    }
}

void voxelize(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds, Eigen::Vector3f leafSize, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& vlouds, std::vector<boost::shared_ptr<pcl::VoxelGrid<pcl::PointXYZ> > >& grids)
{
    grids.resize(clouds.size());
    for (int i = 0; i < clouds.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pVloud (new pcl::PointCloud<pcl::PointXYZ>);
        grids[i] = boost::shared_ptr<pcl::VoxelGrid<pcl::PointXYZ> > (new pcl::VoxelGrid<pcl::PointXYZ>);
        pclx::voxelize(clouds[i], leafSize, *pVloud, *(grids[i]));
    }
}

void voxelize(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds, Eigen::Vector3f leafSize, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& vlouds, std::vector<boost::shared_ptr<pcl::VoxelGrid<pcl::PointXYZRGB> > >& grids)
{
    grids.resize(clouds.size());
    for (int i = 0; i < clouds.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pVloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        grids[i] = boost::shared_ptr<pcl::VoxelGrid<pcl::PointXYZRGB> > (new pcl::VoxelGrid<pcl::PointXYZRGB>);
        pclx::voxelize(clouds[i], leafSize, *pVloud, *(grids[i]));
    }
}

template<typename PointT>
float pclx::computeInclusion3d(typename pcl::PointCloud<PointT>::Ptr pCloud1, typename pcl::PointCloud<PointT>::Ptr pCloud2, Eigen::Vector3f leafSize)
{
    pcl::VoxelGrid<PointT> vox1, vox2;
    pcl::PointCloud<PointT> vloud1, vloud2;
    
    voxelize(pCloud1, leafSize, vloud1, vox1);
    voxelize(pCloud2, leafSize, vloud2, vox2);
    
    return pclx::computeInclusion3d(vox1, vox2);
}

template<typename PointT>
float pclx::computeInclusion3d(pcl::VoxelGrid<PointT>& vox1, pcl::VoxelGrid<PointT>& vox2)
{
    int voxels1, voxels2, voxelsI;
    pclx::countIntersections3d(vox1, vox2, voxels1, voxels2, voxelsI);
    
    if (voxelsI < 1)
        return 0.f;
    else
        return ((float) voxelsI) / std::min(voxels1, voxels2);
}

template<typename PointT>
float pclx::computeOverlap3d(typename pcl::PointCloud<PointT>::Ptr pCloud1, typename pcl::PointCloud<PointT>::Ptr pCloud2, Eigen::Vector3f leafSize)
{
    pcl::VoxelGrid<PointT> vox1, vox2;
    pcl::PointCloud<PointT> vloud1, vloud2;
    
    voxelize(pCloud1, leafSize, vloud1, vox1);
    voxelize(pCloud2, leafSize, vloud2, vox2);
    
    return pclx::computeOverlap3d(vox1, vox2);
}

template<typename PointT>
float pclx::computeOverlap3d(pcl::VoxelGrid<PointT>& vox1, pcl::VoxelGrid<PointT>& vox2)
{
    int voxels1, voxels2, voxelsI;
    pclx::countIntersections3d(vox1, vox2, voxels1, voxels2, voxelsI);
    
    if (voxelsI < 1)
        return 0.f; // avoid zero-division
    else
        return ((float) voxelsI) / (voxels1 + voxels2 - voxelsI);
}

template<typename PointT>
void pclx::countIntersections3d(pcl::VoxelGrid<PointT>& vox1, pcl::VoxelGrid<PointT>& vox2, int& voxels1, int& voxels2, int& voxelsI)
{
    voxels1 = voxels2 = voxelsI = 0;
    
    Eigen::Vector3i min1 = vox1.getMinBoxCoordinates();
    Eigen::Vector3i min2 = vox2.getMinBoxCoordinates();
    
    Eigen::Vector3i max1 = vox1.getMaxBoxCoordinates();
    Eigen::Vector3i max2 = vox2.getMaxBoxCoordinates();
    
    Eigen::Vector3i minI (min1.x() > min2.x() ? min1.x() : min2.x(),
                          min1.y() > min2.y() ? min1.y() : min2.y(),
                          min1.z() > min2.z() ? min1.z() : min2.z());
    Eigen::Vector3i maxI (max1.x() < max2.x() ? max1.x() : max2.x(),
                          max1.y() < max2.y() ? max1.y() : max2.y(),
                          max1.z() < max2.z() ? max1.z() : max2.z());
    
    for (int i = minI.x(); i <= maxI.x(); i++)
    {
        for (int j = minI.y(); j <= maxI.y(); j++)
        {
            for (int k = minI.z(); k <= maxI.z(); k++)
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

template<typename PointT>
void pclx::clusterize(typename pcl::PointCloud<PointT>::Ptr pCloud, float tol, int minSize, std::vector<typename pcl::PointCloud<PointT>::Ptr>& clusters)
{
    clusters.clear();
    
    // Creating the KdTree object for the search method of the extraction
    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud (pCloud);
    
    std::vector<pcl::PointIndices> cluster_indices;
    typename pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (tol);
    ec.setMinClusterSize (minSize);
    ec.setMaxClusterSize (pCloud->points.size());
    ec.setSearchMethod (tree);
    ec.setInputCloud (pCloud);
    ec.extract (cluster_indices);
    
    std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
    for ( ; it != cluster_indices.end (); ++it)
    {
        typename pcl::PointCloud<PointT>::Ptr pCluster (new pcl::PointCloud<PointT>);
        
        std::vector<int>::const_iterator pit = it->indices.begin();
        for ( ; pit != it->indices.end (); pit++)
            pCluster->points.push_back (pCloud->points[*pit]); //*
        
        pCluster->width = pCluster->points.size ();
        pCluster->height = 1;
        pCluster->is_dense = true;
        
        clusters.push_back(pCluster);
    }
}

template<typename PointT>
void pclx::biggestEuclideanCluster(typename pcl::PointCloud<PointT>::Ptr pCloud, float tol, int minSize, pcl::PointCloud<PointT>& cluster)
{
    // Not even try
    // bc the entire cloud is smaller than the minSize
    if (pCloud->points.size() < minSize)
        return;
    
    typename pcl::PointCloud<PointT>::Ptr pNanFreeCloud (new pcl::PointCloud<PointT>);
    
    std::vector< int > indices;
    pcl::removeNaNFromPointCloud(*pCloud,*pNanFreeCloud,indices);
    
    // Creating the KdTree object for the search method of the extraction
    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud (pNanFreeCloud);
    
    std::vector<pcl::PointIndices> clusterIndices;
    
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setInputCloud (pNanFreeCloud);
    ec.setClusterTolerance (tol); // 2cm
    ec.setMinClusterSize (minSize);
    ec.setMaxClusterSize (std::numeric_limits<float>::max());
    ec.setSearchMethod (tree);
    
    ec.extract (clusterIndices);
    
    if (clusterIndices.size() > 0)
    {
        // biggest is the first in the list of extractions' indices
        std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin();
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
            cluster.points.push_back (pNanFreeCloud->points[*pit]);
        
        cluster.width = cluster.points.size ();
        cluster.height = 1;
        cluster.is_dense = false;
    }
}

template void pclx::downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr pCloudIn, Eigen::Vector3f leafSize, pcl::PointCloud<pcl::PointXYZ>& cloudOut);
template void pclx::downsample(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloudIn, Eigen::Vector3f leafSize, pcl::PointCloud<pcl::PointXYZRGB>& cloudOut);

template void pclx::voxelize<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud, Eigen::Vector3f leafSize, pcl::PointCloud<pcl::PointXYZ>& vloud, pcl::VoxelGrid<pcl::PointXYZ>& vox);
template void pclx::voxelize<pcl::PointXYZRGB>( pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud, Eigen::Vector3f leafSize, pcl::PointCloud<pcl::PointXYZRGB>& vloud, pcl::VoxelGrid<pcl::PointXYZRGB>& vox);

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

template void pclx::clusterize<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud, float tol, int minSize, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters);
template void pclx::clusterize<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud, float tol, int minSize, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters);

template void pclx::biggestEuclideanCluster<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ>::Ptr, float tol, int minSize, pcl::PointCloud<pcl::PointXYZ>&);
template void pclx::biggestEuclideanCluster<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB>::Ptr, float tol, int minSize, pcl::PointCloud<pcl::PointXYZRGB>&);
