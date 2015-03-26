//
//  features.cpp
//  remedi3
//
//  Created by Albert Clapés on 11/03/15.
//
//

#include "features.h"
#include <boost/assign/std/vector.hpp> // for 'operator+=()'

using namespace boost::assign; // bring 'operator+=()' into scope

void remedi::features::describeCloudjectsWithPFHRGB(std::vector<Cloudject::Ptr>& cloudjects, float leafSize, float normalRadius, float pfhRadius)
{
    std::vector<float> parameters;
    parameters += leafSize, normalRadius, pfhRadius;
    
    std::vector<Cloudject::Ptr>::iterator it;
    for (it = cloudjects.begin(); it != cloudjects.end(); it++)
        (*it)->describe("pfhrgb250", parameters);
}

void remedi::features::describeCloudjectsWithFPFH(std::vector<Cloudject::Ptr>& cloudjects, float leafSize, float normalRadius, float pfhRadius)
{
    std::vector<float> parameters;
    parameters += leafSize, normalRadius, pfhRadius;
    
    std::vector<Cloudject::Ptr>::iterator it;
    for (it = cloudjects.begin(); it != cloudjects.end(); it++)
        (*it)->describe("fpfh33", parameters);
}

//void remedi::features::transformDescriptorsToSample(const std::list<Cloudject>& cloudjects, std::vector<const char*> objectLabels, std::list<std::vector<float> >& X, std::list<std::vector<int> >& Y)
//{
//    X.clear();
//    
//    std::list<Cloudject>::const_iterator it;
//    for (it = cloudjects.begin(); it != cloudjects.end(); ++it)
//    {
//        // Prepare the vector of multiple labels for this cloudject instance
//        std::vector<int> y (objectLabels.size());
//        for (int k = 0; k < objectLabels.size(); k++)
//        {
//            it->isLabel(objectLabels[k]) ? (y[k] = 1) : (y[k] = -1);
//        }
//        
//        // Unroll the descriptor in a list of points and a replica of ml-vector
//        // for each
//        std::list<std::vector<float> > cjSample = it->toSample();
//        std::list<std::vector<int> > cjSampleLabels (cjSample.size(), y);
//
//        X.splice(X.end(), cjSample);
//        Y.splice(Y.end(), cjSampleLabels);
//    }
//}

