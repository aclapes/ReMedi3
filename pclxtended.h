//
//  pclxtended.h
//  remedi3
//
//  Created by Albert Clapés on 06/04/15.
//
//

#ifndef __remedi3__pclxtended__
#define __remedi3__pclxtended__

#include <stdio.h>
#include <pcl/point_types.h>

namespace pclx
{
    //
    // Inline classes
    //
    
    class Rectangle3D
    {
    public:
        pcl::PointXYZ min;
        pcl::PointXYZ max;
        
        Rectangle3D () {}
        Rectangle3D (const Rectangle3D& rhs) { *this = rhs; }
        Rectangle3D& operator=(const Rectangle3D& rhs)
        {
            if (this != &rhs)
            {
                min = rhs.min;
                max = rhs.max;
            }
            return *this;
        }
        
        float area()
        {
            return ((max.x > min.x) ? (max.x - min.x) : 0.f) *
            ((max.y > min.y) ? (max.y - min.y) : 0.f) *
            ((max.z > min.z) ? (max.z - min.z) : 0.f);
        }
        friend std::ostream& operator<<(std::ostream& os, const Rectangle3D& r)
        {
            return os << "[" << r.min.x << "," << r.min.y << "," << r.min.z << ";\n"
            << r.max.x << "," << r.max.y << "," << r.max.z << "]";
        }
    };
    
    //
    // Functions
    //
    
    float rectangleInclusion(Rectangle3D a, Rectangle3D b);
    float rectangleOverlap(Rectangle3D a, Rectangle3D b);
    Rectangle3D rectangleIntersection(Rectangle3D a, Rectangle3D b);
    
    void findCorrespondences(std::vector<std::vector<pcl::PointXYZ> > positions, float tol, std::vector<std::vector<std::pair<std::pair<int,int>,pcl::PointXYZ> > >&  correspondences);
    void findNextCorrespondence(std::vector<std::vector<pcl::PointXYZ> >& detections, std::vector<std::vector<bool> >& assignations, int v, float tol, std::vector<std::pair<std::pair<int,int>,pcl::PointXYZ> >& chain);
}

#endif /* defined(__remedi3__pclxtended__) */
