//
//  KinectReader.cpp
//  remedi2
//
//  Created by Albert Clapés on 13/07/14.
//
//

#include "KinectReader.h"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>

KinectReader::KinectReader() : m_bPreallocation(false)
{}

KinectReader::KinectReader(const KinectReader& rhs) : m_bPreallocation(false)
{
    *this = rhs;
}

KinectReader& KinectReader::operator=(const KinectReader& rhs)
{
    if (this != &rhs)
    {
        m_bPreallocation = rhs.m_bPreallocation;
    }
    
    return *this;
}

void KinectReader::setPreallocation(bool prealloc)
{
    m_bPreallocation = prealloc;
}

void KinectReader::read(vector<string> dirsPaths, Sequence<Frame>& seq)
{
    for (int v = 0; v < dirsPaths.size(); v++)
    {
        vector<string> framesPaths;
        loadFilePaths(dirsPaths[v], "png", framesPaths);
        
        for (int f = 0; f < framesPaths.size(); f++)
        {
            seq.addFrameFilePath(framesPaths[f], v);
        }
    }
    
    if (m_bPreallocation) seq.allocate();
}

//void KinectReader::read(vector< pair<string,string> > dirsPaths, Sequence<ColorFrame>& seq)
//{
//    for (int v = 0; v < dirsPaths.size(); v++)
//    {
//        vector<string> framesPaths;
//        loadFilePaths(paths[v].first, "png", framesPaths);
//        
//        for (int f = 0; f < colorDirs.size(); f++)
//        {
//            seq.addFrameFilePath(colorDirs[f], v);
//        }
//    }
//    
//    if (m_bPreallocation) seq.allocate();
//}
//
//void KinectReader::read(vector< pair<string,string> > dirsPaths, Sequence<DepthFrame>& seq)
//{
//    for (int v = 0; v < dirsPaths.size(); v++)
//    {
//        vector<string> framesPaths;
//        loadFilePaths(paths[v].first, "png", framesPaths);
//        
//        for (int f = 0; f < depthDirs.size(); f++)
//        {
//            seq.addFrameFilePath(depthDirs[f], v);
//        }
//    }
//    
//    if (m_bPreallocation) seq.allocate();
//}

void KinectReader::read(vector<string> colorDirsPaths, vector<string> depthDirsPaths, Sequence<ColorDepthFrame>& seq)
{
    assert( colorDirsPaths.size() == depthDirsPaths.size() );
    
    for (int v = 0; v < colorDirsPaths.size(); v++)
    {
        vector<string> colorFramesPaths, depthFramesPaths;
        loadFilePaths(colorDirsPaths[v], "png", colorFramesPaths);
        loadFilePaths(depthDirsPaths[v], "png", depthFramesPaths);
        
        assert ( colorFramesPaths.size() == depthFramesPaths.size() );
        
        for (int f = 0; f < colorFramesPaths.size(); f++)
        {
            seq.addFramesFilePath(colorFramesPaths[f], depthFramesPaths[f], v);
        }
    }
    
    if (m_bPreallocation) seq.allocate();
}

void KinectReader::loadFilePaths(string dir, const char* filetype, vector<string>& filePaths)
{
 	filePaths.clear();
    
    string extension = "." + string(filetype);
    const char* path = dir.c_str();
	if( boost::filesystem::exists( path ) )
	{
		boost::filesystem::directory_iterator end;
		boost::filesystem::directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( !boost::filesystem::is_directory( *iter ) && (iter->path().extension().string().compare(extension) == 0) )
            {
                string aux = iter->path().string();
				filePaths.push_back(iter->path().string());
            }
		}
	}
}