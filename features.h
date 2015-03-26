//
//  features.h
//  remedi3
//
//  Created by Albert Clapés on 11/03/15.
//
//

#ifndef __remedi3__features__
#define __remedi3__features__

#include <stdio.h>

#include "Cloudject.hpp"

namespace remedi
{
    namespace features
    {
        void describeCloudjectsWithPFHRGB(std::vector<Cloudject::Ptr>& cloudjects, float leafSize, float normalRadius, float pfhRadius);
        void describeCloudjectsWithFPFH(std::vector<Cloudject::Ptr>& cloudjects, float leafSize, float normalRadius, float pfhRadius);
        
//        void transformDescriptorsToSample(const std::list<Cloudject>& cloudjects, std::vector<const char*> objectLabels, std::list<std::vector<float> >& X, std::list<std::vector<int> >& Y);
    }
}

#endif /* defined(__remedi3__features__) */
