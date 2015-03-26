//
//  KinectReader.h
//  remedi2
//
//  Created by Albert Clapés on 13/07/14.
//
//

#ifndef remedi2_KinectReader_hpp
#define remedi2_KinectReader_hpp

#include <iostream>

#include "Sequence.h"
#include "ColorDepthFrame.h"

using namespace std;

class KinectReader
{
public:
    KinectReader();
    KinectReader(const KinectReader& rhs);
    
    KinectReader& operator=(const KinectReader& rhs);
    
    void setPreallocation(bool prealloc = true);
    
    void read(vector<string> dirsPaths, Sequence<Frame>& seq);
//    void read(vector<string> dirsPaths, Sequence<ColorFrame>& seq);
//    void read(vector<string> dirsPaths, Sequence<DepthFrame>& seq);
    void read(vector<string> colorDirsPaths, vector<string> depthDirsPaths, Sequence<ColorDepthFrame>& seq);
    
private:
    void loadFilePaths(string dir, const char* filetype, vector<string>& filePaths);
    
    bool m_bPreallocation;
};

#endif
