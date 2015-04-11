//
//  MultiviewVisualizer.h
//  remedi3
//
//  Created by Albert Clapés on 11/04/15.
//
//

#ifndef __remedi3__MultiviewVisualizer__
#define __remedi3__MultiviewVisualizer__

#include <stdio.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "InteractiveRegisterer.h"
#include "ColorDepthFrame.h"
#include "Cloudject.hpp"

#include <boost/shared_ptr.hpp>

class MultiviewVisualizer
{
public:
    MultiviewVisualizer();
    MultiviewVisualizer(const MultiviewVisualizer& rhs);
    MultiviewVisualizer& operator=(const MultiviewVisualizer& rhs);
    
    void create(int V, InteractiveRegisterer::Ptr pRegisterer);
    void visualize(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> interactors, std::vector<std::vector<Cloudject::Ptr> > interactions, std::vector<std::vector<Cloudject::Ptr> > actors); // monocular
    void visualize(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> interactors, std::vector<std::vector<Cloudject::Ptr> > interactions, std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences); // multiview
    
    typedef boost::shared_ptr<MultiviewVisualizer> Ptr;
    
private:
    pcl::visualization::PCLVisualizer::Ptr m_pViz;
    std::vector<int> m_Viewports;
};

#endif /* defined(__remedi3__MultiviewVisualizer__) */
