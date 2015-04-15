//
//  Sequence.h
//  remedi
//
//  Created by Albert Clapés on 15/05/14.
//
//

#ifndef __remedi__Sequence__
#define __remedi__Sequence__

#include "ColorFrame.h"
#include "DepthFrame.h"
#include "ColorDepthFrame.h"

#include <iostream>
#include <vector>

using namespace std;

template<typename FrameT>
class SequenceBase
{
public:
    SequenceBase();
    SequenceBase(vector<const char*> viewsNames);
    SequenceBase(vector<vector<string> > paths);
    SequenceBase(vector<vector<typename FrameT::Ptr> > stream);
    
    SequenceBase(const SequenceBase<FrameT>& rhs);
    SequenceBase<FrameT>& operator=(const SequenceBase<FrameT>& rhs);
    
    string getPath() const;
    void setPath(string path);

    string getName() const;
    void setName(string name);
    
    int getNumOfViews() const;
    void setViewsNames(vector<const char*> viewsNames);
    
    const char* getViewName(int v) const;
    vector<const char*> getViewsNames() const;
    
    vector<vector<string> > getFramesPaths() const;
    void setFramesPaths(vector<vector<string> > paths);
    
    void addStream(vector<typename FrameT::Ptr> stream);
    void setStream(vector<typename FrameT::Ptr> stream, int view);
    
    vector<vector<typename FrameT::Ptr> > getStreams() const;
    void setStreams(vector<vector<typename FrameT::Ptr> > streams);
    
    void addFrameFilePath(string framePath, int view);
    void addFrameFilePath(vector<string> framePaths);
    
    void addFrame(typename FrameT::Ptr frame, int view);
    void addFrame(vector<typename FrameT::Ptr> frames);
    
    bool hasNextFrames(int step = 1);
    bool hasPreviousFrames(int step = 1);
    
    vector<typename FrameT::Ptr> nextFrames(int step = 1);
    vector<typename FrameT::Ptr> previousFrames(int step = 1);
    vector<typename FrameT::Ptr> getFrames(int i);
    
    void next(int step = 1);
    void previous(int step = 1);
    
    vector<typename FrameT::Ptr> getFrames(vector<string> filenames);
    vector<typename FrameT::Ptr> getFrames(vector<string> filenames, vector<int>& indices);
    vector<string> getFramesFilenames();
    
    int getMaxDelay();
    int getMinNumOfFrames();
    int getMaxNumOfFrames();
    
    void getNumOfFramesOfViews(vector<int>& numOfFrames);
    int getTotalNumOfFrames();
    vector<int> getCurrentFramesID();
    vector<float> getProgress();
    vector<int> at();
    
    vector<int> getFrameIndices() const;
    void setFrameIndices(vector<int> indices);
    
    vector<int> getDelays() const;
    void setDelays(vector<int> delays);
    
    vector<int> getFrameCounters() const;
    
    void restart();
    
    //virtual void allocate() = 0;
	void allocate();
    
    typedef boost::shared_ptr<SequenceBase<FrameT> > Ptr;
    
private:
    void readFrame(vector<string> paths, int i, FrameT& frame);
   
protected:
    //
    // Attributes
    //
    
    vector<const char*> m_ViewsNames;

    string m_Path;
    string m_Name;
    
    vector< vector<string> > m_Paths;
    vector< vector<typename FrameT::Ptr> > m_Streams;
    
    vector<int> m_FrameIndices;
    vector<int> m_Delays;
};


template<typename FrameT>
class Sequence : public SequenceBase<FrameT>
{
public:
    Sequence();
    Sequence(vector<const char*> viewsNames);
    Sequence(vector<vector<string> > paths);
    Sequence(vector<vector<typename FrameT::Ptr> > stream);

    Sequence(const Sequence<FrameT>& rhs);
    Sequence<FrameT>& operator=(const Sequence<FrameT>& rhs);
    
	//void allocate();

    typedef boost::shared_ptr<Sequence<Frame> > Ptr;
};

template<>
class Sequence<ColorFrame> : public SequenceBase<ColorFrame>
{
public:
    Sequence();
    Sequence(vector<const char*> viewsNames);
    Sequence(vector< vector<string> > paths);
    Sequence(vector<vector<ColorFrame::Ptr> > stream);
    
    Sequence(const Sequence<ColorFrame>& rhs);
    Sequence<ColorFrame>& operator=(const Sequence<ColorFrame>& rhs);
    
    Sequence(const Sequence<ColorDepthFrame>& rhs);
    Sequence<ColorFrame>& operator=(const Sequence<ColorDepthFrame>& rhs);
    
	//void allocate();

    typedef boost::shared_ptr<Sequence<ColorFrame> > Ptr;
};

template<>
class Sequence<DepthFrame> : public SequenceBase<DepthFrame>
{
public:
    Sequence();
    Sequence(vector<const char*> viewsNames);
    Sequence(vector< vector<string> > paths);
    Sequence(vector<vector<DepthFrame::Ptr> > stream);
    
    Sequence(const Sequence<DepthFrame>& rhs);
    Sequence<DepthFrame>& operator=(const Sequence<DepthFrame>& rhs);
    
    Sequence(const Sequence<ColorDepthFrame>& rhs);
    Sequence<DepthFrame>& operator=(const Sequence<ColorDepthFrame>& rhs);
    
	//void allocate();

    void setReferencePoints(vector<pcl::PointXYZ> points);
    vector<pcl::PointXYZ> getReferencePoints();

    void setRegistrationTransformations(vector<Eigen::Matrix4f> transformations);
    
    typedef boost::shared_ptr<Sequence<DepthFrame> > Ptr;

private:
    vector<pcl::PointXYZ> m_ReferencePoints;
    vector<Eigen::Matrix4f> m_Transformations;
};


template<>
class Sequence<ColorDepthFrame>
{
public:
    Sequence();
    Sequence(vector<const char*> viewsNames);
    Sequence(vector< vector< pair<string,string> > > paths);
    Sequence(vector<vector<ColorDepthFrame::Ptr> > stream);
    
    Sequence(const Sequence<ColorDepthFrame>& rhs);
    Sequence<ColorDepthFrame>& operator=(const Sequence<ColorDepthFrame>& rhs);
    
    string getPath() const;
    void setPath(string path);
    
    string getName() const;
    void setName(string name);
    
    int getNumOfViews() const;
    void setViewsNames(vector<const char*> viewsNames);
    
    const char* getViewName(int v) const;
    vector<const char*> getViewsNames() const;
    
    vector< vector< pair<string,string> > > getFramesPaths() const;
    void setFramesPaths(vector< vector< pair<string,string> > > paths);
    
    void addStream(vector<ColorDepthFrame::Ptr> stream);
    void setStream(vector<ColorDepthFrame::Ptr> stream, int view);
    void setStreams(vector<vector<ColorDepthFrame::Ptr> > streams);
    vector<vector<ColorDepthFrame::Ptr> > getStreams() const;
    
    void addFramesFilePath(string colorPath, string depthPath, int view);
    void addFrameFilePath(vector< pair<string,string> > framePaths);
    
    void addFrame(ColorDepthFrame::Ptr frame, int view);
    void addFrame(vector<ColorDepthFrame::Ptr> frames);
    
    bool hasNextFrames(int step = 1);
    bool hasPreviousFrames(int step = 1);
    
    vector<ColorDepthFrame::Ptr> nextFrames(int step = 1);
    vector<ColorDepthFrame::Ptr> previousFrames(int step = 1);
    vector<ColorDepthFrame::Ptr> getFrames(int i);
    
    void next(int step = 1);
    void previous(int step = 1);
        
    vector<ColorDepthFrame::Ptr> getFrames(vector<string> filenames);
    vector<ColorDepthFrame::Ptr> getFrames(vector<string> filenames, vector<int>& indices);
    vector<string> getFramesFilenames();
    
    int getMaxDelay();
    int getMinNumOfFrames();
    int getMaxNumOfFrames();
    
    void getNumOfFramesOfViews(vector<int>& numOfFrames);
    vector<int> getCurrentFramesID();
    vector<float> getProgress();
    vector<int> at();
    
    vector<int> getDelays() const;
    void setDelays(vector<int> delays);
    
    vector<int> getFrameIndices() const;
    void setFrameIndices(vector<int> indices);
    
    vector<int> getFrameCounters() const;
    
    void setReferencePoints(vector<pcl::PointXYZ> points);
    vector<pcl::PointXYZ> getReferencePoints() const;
    
    vector<Eigen::Matrix4f> getRegistrationTransformations() const;
    void setRegistrationTransformations(vector<Eigen::Matrix4f> transformations);
    
    void restart();
    
    void allocate();
    
    typedef boost::shared_ptr<Sequence<ColorDepthFrame> > Ptr;
    
private:
    void readFrame(vector< pair<string,string> > paths, int i, ColorDepthFrame& frame);
    
    //
    // Attributes
    //
    
    vector<const char*> m_ViewsNames;
    
    string m_Path;
    string m_Name;
    
    vector< vector< pair<string,string> > > m_Paths;
    vector< vector<ColorDepthFrame::Ptr> > m_Streams;
    
    vector<int> m_FrameIndices;
    vector<int> m_Delays;
    
    vector<pcl::PointXYZ> m_ReferencePoints;
    vector<Eigen::Matrix4f> m_Transformations;
};

#endif /* defined(__remedi__Sequence__) */
