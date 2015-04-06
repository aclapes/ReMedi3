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

