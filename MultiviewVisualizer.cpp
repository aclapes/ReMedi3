//
//  MultiviewVisualizer.cpp
//  remedi3
//
//  Created by Albert Clapés on 11/04/15.
//
//

#include "MultiviewVisualizer.h"


MultiviewVisualizer::MultiviewVisualizer()
{
    
}

MultiviewVisualizer::MultiviewVisualizer(const MultiviewVisualizer& rhs)
{
    *this = rhs;
}

MultiviewVisualizer& MultiviewVisualizer::operator=(const MultiviewVisualizer& rhs)
{
    if (this != &rhs)
    {
        m_Viewports = rhs.m_Viewports;
    }
    return *this;
}

void MultiviewVisualizer::create(int V, InteractiveRegisterer::Ptr pRegisterer)
{
    m_Viewports.resize(V + 1);
    
    m_pViz = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer);
    for (int v = 0; v < (V + 1); v++)
    {
        m_pViz->createViewPort(v*(1.f/(V+1)), 0, (v+1)*(1.f/(V+1)), 1, m_Viewports[v]);
        m_pViz->setBackgroundColor (1, 1, 1, m_Viewports[v]);
        m_pViz->addCoordinateSystem(0.1, 0, 0, 0, "cs" + boost::lexical_cast<string>(v), m_Viewports[v]);
        pRegisterer->setDefaultCamera(*m_pViz, m_Viewports[v]);
    }
}

void MultiviewVisualizer::visualize(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> interactors, std::vector<std::vector<Cloudject::Ptr> > interactions, std::vector<std::vector<Cloudject::Ptr> > actors)
{
    // Count the number of views (viewports - 1)
    int V = m_Viewports.size() - 1;
    
    for (int v = 0; v < V; v++)
    {
        m_pViz->removeAllPointClouds(m_Viewports[v]);
        m_pViz->removeAllShapes(m_Viewports[v]);
        
        for (int i = 0; i < interactors.size(); i++)
            m_pViz->addPointCloud(interactors[i], "interactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), m_Viewports[v] );
        
        for (int i = 0; i < interactions[v].size(); i++)
        {
            m_pViz->addPointCloud(interactions[v][i]->getRegisteredCloud(), "nactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), m_Viewports[v] );
            m_pViz->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 0, "nactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), m_Viewports[v]);
        }
        
        for (int i = 0; i < actors[v].size(); i++)
        {
            m_pViz->addPointCloud(actors[v][i]->getRegisteredCloud(), "actor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), m_Viewports[v] );
            m_pViz->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, g_Colors[i%14][0], g_Colors[i%14][1], g_Colors[i%14][2], "actor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), m_Viewports[v]);
        }
    }
    
    m_pViz->spinOnce(50);
}

void MultiviewVisualizer::visualize(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> interactors, std::vector<std::vector<Cloudject::Ptr> > interactions, std::vector<std::vector<std::pair<int, Cloudject::Ptr> > > correspondences)
{
    // Count the number of views (viewports - 1)
    int V = m_Viewports.size() - 1;
    
    for (int v = 0; v < V; v++)
    {
        m_pViz->removeAllPointClouds(m_Viewports[v]);
        m_pViz->removeAllShapes(m_Viewports[v]);
    }
    
    for (int v = 0; v < V; v++)
    {
        for (int i = 0; i < interactors.size(); i++)
            m_pViz->addPointCloud(interactors[i], "interactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), m_Viewports[v] );
        
        for (int i = 0; i < interactions[v].size(); i++)
        {
            m_pViz->addPointCloud(interactions[v][i]->getRegisteredCloud(), "nactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), m_Viewports[v] );
            m_pViz->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 0, "nactor" + boost::lexical_cast<string>(v) + "-" + boost::lexical_cast<string>(i), m_Viewports[v]);
        }
    }
    
    for (int i = 0; i < correspondences.size(); i++)
    {
        std::vector<VoxelGridPtr> grids (correspondences[i].size());
        for (int j = 0; j < correspondences[i].size(); j++)
        {
            int v = correspondences[i][j].first;
            
            // Visualize in color and boxed in the viewpoirt
            
            m_pViz->addPointCloud(correspondences[i][j].second->getRegisteredCloud(), "actor" + boost::lexical_cast<string>(i) + "-" + boost::lexical_cast<string>(j), m_Viewports[v] );
            
            grids[j] = correspondences[i][j].second->getRegisteredGrid();
            Eigen::Vector3f leafSize = grids[j]->getLeafSize();
            Eigen::Vector3i min = grids[j]->getMinBoxCoordinates();
            Eigen::Vector3i max = grids[j]->getMaxBoxCoordinates();
            
            m_pViz->addCube(min.x() * leafSize.x(), max.x() * leafSize.x(), min.y() * leafSize.y(), max.y() * leafSize.y(), min.z() * leafSize.z(), max.z() * leafSize.z(), g_Colors[i%14][0], g_Colors[i%14][1], g_Colors[i%14][2], "cube" + boost::lexical_cast<string>(i) + "-" + boost::lexical_cast<string>(j), m_Viewports[v]);
            
            // Visualize shaded and with the grid boxes in the fusion viewport (to notice the overlap)
            for (int x = min.x(); x < max.x(); x++) for (int y = min.y(); y < max.y(); y++) for (int z = min.z(); z < max.z(); z++)
            {
                if (grids[j]->getCentroidIndexAt(Eigen::Vector3i(x,y,z)) > 0)
                    m_pViz->addCube(x * leafSize.x(), (x+1) * leafSize.x(), y * leafSize.y(), (y+1) * leafSize.y(), z * leafSize.z(), (z+1) * leafSize.z(), g_Colors[v%14][0], g_Colors[v%14][1], g_Colors[v%14][2], "cube" + boost::lexical_cast<string>(i) + "-" + boost::lexical_cast<string>(j) + "-" + boost::lexical_cast<string>(x) + "-" + boost::lexical_cast<string>(y) + "-" + boost::lexical_cast<string>(z), m_Viewports[V]);
            }
        }
        
        Eigen::Vector3f leafSize = grids[0]->getLeafSize();
        
        for (int j1 = 0; j1 < correspondences[i].size(); j1++)
        {
            for (int j2 = j1 + 1; j2 < correspondences[i].size(); j2++)
            {
                Eigen::Vector3f leafSize = grids[j1]->getLeafSize();
                Eigen::Vector3f leafSizeAux = grids[j2]->getLeafSize();
                // Check they are equal
                
                Eigen::Vector3i min1 = grids[j1]->getMinBoxCoordinates();
                Eigen::Vector3i min2 = grids[j2]->getMinBoxCoordinates();
                Eigen::Vector3i max1 = grids[j1]->getMaxBoxCoordinates();
                Eigen::Vector3i max2 = grids[j2]->getMaxBoxCoordinates();
                
                Eigen::Vector3i minI (min1.x() > min2.x() ? min1.x() : min2.x(),
                                      min1.y() > min2.y() ? min1.y() : min2.y(),
                                      min1.z() > min2.z() ? min1.z() : min2.z());
                Eigen::Vector3i maxI (max1.x() < max2.x() ? max1.x() : max2.x(),
                                      max1.y() < max2.y() ? max1.y() : max2.y(),
                                      max1.z() < max2.z() ? max1.z() : max2.z());
                
                for (int x = minI.x(); x < maxI.x(); x++) for (int y = minI.y(); y < maxI.y(); y++) for (int z = minI.z(); z < maxI.z(); z++)
                {
                    int idx1 = grids[j1]->getCentroidIndexAt(Eigen::Vector3i(x,y,z));
                    int idx2 = grids[j2]->getCentroidIndexAt(Eigen::Vector3i(x,y,z));
                    if (idx1 != 1 && idx2 != -1)
                    {
                        std::string name = "ovl_sphere_" + boost::lexical_cast<string>(i) + "-" + boost::lexical_cast<string>(x) + "-" + boost::lexical_cast<string>(y) + "-" + boost::lexical_cast<string>(z);
                        m_pViz->addSphere(pcl::PointXYZ(x*leafSize.x()+(leafSize.x()/2.f),y*leafSize.y()+(leafSize.y()/2.f),z*leafSize.z()+(leafSize.z()/2.f)), leafSize.x()/2.f, 0, 0, 1, name, m_Viewports[V]);
                    }
                }
            }
        }
    }
    
    m_pViz->spinOnce(50);
}
