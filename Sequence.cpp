//
//  Sequence.cpp
//  remedi
//
//  Created by Albert Clapés on 15/05/14.
//
//

#include "Sequence.h"
#include "conversion.h"

//
// Sequence
//

template<typename FrameT>
SequenceBase<FrameT>::SequenceBase()
{
}

template<typename FrameT>
SequenceBase<FrameT>::SequenceBase(vector<const char*> viewsNames) : m_ViewsNames(viewsNames)
{
    m_Paths.resize(viewsNames.size());
    m_Streams.resize(viewsNames.size());
    m_FrameIndices.resize(viewsNames.size(), -1);
    m_Delays.resize(viewsNames.size(), 0);
}

template<typename FrameT>
SequenceBase<FrameT>::SequenceBase(vector<vector<string> > paths)
{
    m_Paths = paths;
    
    m_Streams.resize(m_Paths.size());
    m_FrameIndices.resize(m_Paths.size(), -1);
    m_Delays.resize(m_Paths.size(), 0);
}

template<typename FrameT>
SequenceBase<FrameT>::SequenceBase(const SequenceBase<FrameT>& rhs)
{
    *this = rhs;
}

template<typename FrameT>
SequenceBase<FrameT>& SequenceBase<FrameT>::operator=(const SequenceBase<FrameT>& rhs)
{
    if (this != &rhs)
    {
        m_Path = rhs.m_Path;
        m_Name = rhs.m_Name;
        m_Paths = rhs.m_Paths;
        m_Streams = rhs.m_Streams;
        m_FrameIndices = rhs.m_FrameIndices;
        m_Delays = rhs.m_Delays;
    }
    
    return *this;
}

template<typename FrameT>
string SequenceBase<FrameT>::getPath() const
{
    return m_Path;
}

template<typename FrameT>
void SequenceBase<FrameT>::setPath(string path)
{
    m_Path = path;
}

template<typename FrameT>
string SequenceBase<FrameT>::getName() const
{
    return m_Name;
}

template<typename FrameT>
void SequenceBase<FrameT>::setName(string name)
{
    m_Name = name;
}

template<typename FrameT>
int SequenceBase<FrameT>::getNumOfViews() const
{
    return m_Paths.size();
}

template<typename FrameT>
void SequenceBase<FrameT>::setViewsNames(vector<const char*> viewsNames)
{
    m_Paths.clear();
    m_FrameIndices.clear();
    m_Delays.clear();
    
    m_Paths.resize(viewsNames.size());
    m_FrameIndices.resize(viewsNames.size(), -1);
    m_Delays.resize(viewsNames.size(),0);
}

template<typename FrameT>
const char* SequenceBase<FrameT>::getViewName(int v) const
{
    return m_ViewsNames[v];
}

template<typename FrameT>
vector<const char*> SequenceBase<FrameT>::getViewsNames() const
{
    return m_ViewsNames;
}

template<typename FrameT>
vector<vector<string> > SequenceBase<FrameT>::getFramesPaths() const
{
    return m_Paths;
}

template<typename FrameT>
void SequenceBase<FrameT>::setFramesPaths(vector<vector<string> > paths)
{
    m_Paths = paths;
    
    m_FrameIndices.resize(m_Paths.size(), -1);
    m_Delays.resize(m_Paths.size(), 0);
}

template<typename FrameT>
void SequenceBase<FrameT>::addStream(vector<typename FrameT::Ptr> stream)
{
    m_Streams.push_back(stream);
    m_FrameIndices.push_back(-1);
    
    m_Delays.resize(m_Streams.size());
}

template<typename FrameT>
void SequenceBase<FrameT>::setStream(vector<typename FrameT::Ptr> stream, int view)
{
    while (view >= m_Streams.size())
        m_Streams.push_back(vector<typename FrameT::Ptr>());
    
    m_Streams[view] = stream;
}

template<typename FrameT>
vector<vector<typename FrameT::Ptr> > SequenceBase<FrameT>::getStreams() const
{
    return m_Streams;
}

template<typename FrameT>
void SequenceBase<FrameT>::setStreams(vector<vector<typename FrameT::Ptr> > streams)
{
    m_Streams = streams;
    
    m_FrameIndices.resize(m_Streams.size(), -1);
    m_Delays.resize(m_Streams.size(), 0);
}

template<typename FrameT>
void SequenceBase<FrameT>::addFrameFilePath(string framePath, int view)
{
    while (view >= m_Paths.size())
    {
        m_Paths.push_back(vector<string>());
        m_Streams.push_back(vector<typename FrameT::Ptr>());
        m_FrameIndices.push_back(-1);
        
        m_Delays.push_back(0);
    }
    
    m_Paths[view].push_back(framePath);
}

template<typename FrameT>
void SequenceBase<FrameT>::addFrameFilePath(vector<string> framePaths)
{
    for (int i = 0; i < m_Paths.size(); i++)
        m_Paths[i].push_back(framePaths[i]);
}

template<typename FrameT>
void SequenceBase<FrameT>::addFrame(typename FrameT::Ptr frame, int view)
{
    m_Streams[view].push_back(frame);
}

template<typename FrameT>
void SequenceBase<FrameT>::addFrame(vector<typename FrameT::Ptr> frame)
{
    for (int i = 0; i < m_Streams.size(); i++)
        m_Streams[i].push_back(frame[i]);
}

template<typename FrameT>
bool SequenceBase<FrameT>::hasNextFrames(int step)
{
    for (int i = 0; i < m_Paths.size(); i++)
        if (m_FrameIndices[i] + m_Delays[i] + step > m_Paths[i].size() - 1)
            return false;
    
    return true;
}

template<typename FrameT>
bool SequenceBase<FrameT>::hasPreviousFrames(int step)
{
    for (int i = 0; i < m_Paths.size(); i++)
        if (m_FrameIndices[i] + m_Delays[i] - step < 0)
            return false;
    
    return true;
}

template<typename FrameT>
vector<typename FrameT::Ptr> SequenceBase<FrameT>::nextFrames(int step)
{
    for (int v = 0; v < m_Paths.size(); v++)
        m_FrameIndices[v] += step;
    
    vector<typename FrameT::Ptr> frames;
    for (int v = 0; v < m_Paths.size(); v++)
    {
        typename FrameT::Ptr frame (new FrameT);
        int delay = (m_Delays.size()) > 0 ? m_Delays[v] : 0;
        if (m_Streams[v].size() > 0)
            frame = m_Streams[v][m_FrameIndices[v] + delay]; // preallocation
        else
            readFrame(m_Paths[v], m_FrameIndices[v] + delay, *frame);
        
        frame->setPath(m_Paths[v][m_FrameIndices[v] + m_Delays[v]]);
        frame->setFilename( getFilenameFromPath(frame->getPath()) );
        frames.push_back(frame);
    }

    return frames;
}

template<typename FrameT>
vector<typename FrameT::Ptr> SequenceBase<FrameT>::previousFrames(int step)
{
    for (int v = 0; v < m_Paths.size(); v++)
        m_FrameIndices[v] -= step;
    
    vector<typename FrameT::Ptr> frames;
    for (int v = 0; v < m_Paths.size(); v++)
    {
        typename FrameT::Ptr frame (new FrameT);
        if (m_Streams[v].size() > 0)
            frame = m_Streams[v][m_FrameIndices[v] + m_Delays[v]]; // preallocation
        else
            readFrame(m_Paths[v], m_FrameIndices[v] + m_Delays[v], *frame);
                                  
        frame->setPath(m_Paths[v][m_FrameIndices[v] + m_Delays[v]]);
        frame->setFilename( getFilenameFromPath(frame->getPath()) );
        frames.push_back(frame);
    }
    
    return frames;
}

template<typename FrameT>
vector<typename FrameT::Ptr> SequenceBase<FrameT>::getFrames(int i)
{
    vector<typename FrameT::Ptr> frames;
    for (int v = 0; v < m_Paths.size(); v++)
    {
        typename FrameT::Ptr frame (new FrameT);
        if (m_Streams[v].size() > 0)
            frame = m_Streams[v][i];
        else
            readFrame(m_Paths[v], i, *frame);
        
        frames.push_back(frame);
    }
    
    return frames;
}

template<typename FrameT>
void SequenceBase<FrameT>::next(int step)
{
    for (int v = 0; v < m_Paths.size(); v++)
        m_FrameIndices[v] += step;
}

template<typename FrameT>
void SequenceBase<FrameT>::previous(int step)
{
    for (int v = 0; v < m_Paths.size(); v++)
        m_FrameIndices[v] -= step;
}

template<typename FrameT>
vector<typename FrameT::Ptr> SequenceBase<FrameT>::getFrames(vector<string> filenames)
{
    vector<int> indices;
    return getFrames(filenames, indices); // indices discarded
}

template<typename FrameT>
vector<typename FrameT::Ptr> SequenceBase<FrameT>::getFrames(vector<string> filenames, vector<int>& indices)
{
    assert (filenames.size() == m_Paths.size());
    
    vector<typename FrameT::Ptr> frames (m_Paths.size());
    indices.resize(m_Paths.size());
    for (int v = 0; v < m_Paths.size(); v++)
    {
        int idx = searchByName(m_Paths[v], filenames[v]);
        
        typename FrameT::Ptr frame (new FrameT);
        if (m_Streams[v].size() > 0)
            frame = m_Streams[v][idx];
        else
            readFrame(m_Paths[v], idx, *frame);
        
        frames[v] = frame;
        indices[v] = idx;
    }
    
    return frames;
}

// Get current frames filenames
template<typename FrameT>
vector<string> SequenceBase<FrameT>::getFramesFilenames()
{
    vector<string> filenames (m_Paths.size());
    
    for (int v = 0; v < filenames.size(); v++)
    {
        filenames[v] = getFilenameFromPath(m_Paths[v][m_FrameIndices[v] + m_Delays[v]]);
    }
    
    return filenames;
}

template<typename FrameT>
int SequenceBase<FrameT>::getMinNumOfFrames()
{
    int min = std::numeric_limits<int>::max();
    for (int v = 0; v < getNumOfViews(); v++)
    {
        int numOfFramesOfView = m_Paths[v].size() - m_Delays[v];
        if (numOfFramesOfView < min)
            min = numOfFramesOfView;
    }
    
    return min;
}

template<typename FrameT>
void SequenceBase<FrameT>::getNumOfFramesOfViews(vector<int>& numOfFrames)
{
    for (int v = 0; v < getNumOfViews(); v++)
        numOfFrames.push_back(m_Paths[v].size());
}

template<typename FrameT>
vector<int> SequenceBase<FrameT>::getCurrentFramesID()
{
    vector<int> counters;
    for (int i = 0; i < m_Paths.size(); i++)
        counters.push_back(m_FrameIndices[i] + m_Delays[i]);
    
    return counters;
}

template<typename FrameT>
vector<float> SequenceBase<FrameT>::getProgress()
{
    vector<float> progresses;
    for (int i = 0; i < m_Paths.size(); i++)
        progresses.push_back(((float) (m_FrameIndices[i] + m_Delays[i])) / m_Paths[i].size());
    
    return progresses;
}

template<typename FrameT>
vector<int> SequenceBase<FrameT>::at()
{
    vector<int> frameDelayedCounter(m_Streams.size());
    
    for (int i = 0; i < m_Streams.size(); i++)
        frameDelayedCounter[i] = m_FrameIndices[i] + m_Delays[i];
    
    return frameDelayedCounter;
}

template<typename FrameT>
vector<int> SequenceBase<FrameT>::getFrameIndices() const
{
    return m_FrameIndices;
}

template<typename FrameT>
vector<int> SequenceBase<FrameT>::getFrameCounters() const
{
    vector<int> counters (m_FrameIndices.size());

    for (int v = 0; v < m_FrameIndices.size(); v++)
    counters[v] = m_FrameIndices[v] + m_Delays[v];

    return counters;
}

template<typename FrameT>
void SequenceBase<FrameT>::setFrameIndices(vector<int> indices)
{
    m_FrameIndices = indices;
}

template<typename FrameT>
vector<int> SequenceBase<FrameT>::getDelays() const
{
    return m_Delays;
}

template<typename FrameT>
void SequenceBase<FrameT>::setDelays(vector<int> delays)
{
    m_Delays = delays;
}

template<typename FrameT>
void SequenceBase<FrameT>::restart()
{
    m_FrameIndices.clear();
    m_FrameIndices.resize(m_Streams.size(), -1);
}

template<typename FrameT>
void SequenceBase<FrameT>::readFrame(vector<string> paths, int f, FrameT& frame)
{
    assert (f < paths.size());
    
    frame = FrameT( cv::imread(paths[f].c_str(), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR) );
}

template<typename FrameT>
void SequenceBase<FrameT>::allocate()
{
    m_Streams.resize(m_Paths.size());
    
    for (int v = 0; v < m_Paths.size(); v++)
    {
        for (int f = 0; f < m_Paths[v].size(); f++)
        {
            typename FrameT::Ptr frame (new FrameT);
            
            int flags = CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR;
            frame->set( cv::imread(m_Paths[v][f], flags) );
            
            m_Streams[v].push_back(frame);
        }
    }
}


//
// Sequence
//

template<typename FrameT>
Sequence<FrameT>::Sequence()
{
}

template<typename FrameT>
Sequence<FrameT>::Sequence(vector<const char*> viewsNames)
: SequenceBase<FrameT>(viewsNames)
{
}

template<typename FrameT>
Sequence<FrameT>::Sequence(vector< vector<string> > paths)
: SequenceBase<FrameT>(paths)
{
}

template<typename FrameT>
Sequence<FrameT>::Sequence(const Sequence<FrameT>& rhs)
: SequenceBase<FrameT>(rhs)
{
    *this = rhs;
}

template<typename FrameT>
Sequence<FrameT>& Sequence<FrameT>::operator=(const Sequence<FrameT>& rhs)
{
    if (this != &rhs)
    {
        SequenceBase<FrameT>::operator=(rhs);
    }
    
    return *this;
}

//
// Sequence<ColorFrame>
//

Sequence<ColorFrame>::Sequence()
{
}

Sequence<ColorFrame>::Sequence(vector<const char*> viewsNames)
: SequenceBase<ColorFrame>(viewsNames)
{
}

Sequence<ColorFrame>::Sequence(vector< vector<string> > paths)
: SequenceBase<ColorFrame>(paths)
{
}

Sequence<ColorFrame>::Sequence(const Sequence<ColorFrame>& rhs)
: SequenceBase<ColorFrame>(rhs)
{
    *this = rhs;
}

Sequence<ColorFrame>& Sequence<ColorFrame>::operator=(const Sequence<ColorFrame>& rhs)
{
    if (this != &rhs)
    {
        SequenceBase<ColorFrame>::operator=(rhs);
    }
    
    return *this;
}

Sequence<ColorFrame>::Sequence(const Sequence<ColorDepthFrame>& rhs)
: SequenceBase<ColorFrame>()
{
    *this = rhs;
}

Sequence<ColorFrame>& Sequence<ColorFrame>::operator=(const Sequence<ColorDepthFrame>& rhs)
{
    m_Path = rhs.getPath();
    m_Name = rhs.getName();
    
    // Paths copy
    vector<vector< pair<string,string> > > paths = rhs.getFramesPaths();
    m_Paths.resize(paths.size());
    for (int v = 0; v < paths.size(); v++)
        for (int f = 0; f < paths[v].size(); f++)
            m_Paths[v].push_back(paths[v][f].first);
    
    // Streams copy
    vector< vector<ColorDepthFrame::Ptr> > streams = rhs.getStreams();
    m_Streams.resize(streams.size());
    for (int v = 0; v < streams.size(); v++)
        for (int f = 0; f < streams[v].size(); f++)
            m_Streams[v].push_back(streams[v][f]);
    
    m_FrameIndices = rhs.getFrameIndices();
    m_Delays = rhs.getDelays();
    
    return *this;
}


//
// Sequence<DepthFrame>
//

Sequence<DepthFrame>::Sequence()
{
}

Sequence<DepthFrame>::Sequence(vector<const char*> viewsNames)
: SequenceBase<DepthFrame>(viewsNames)
{
}

Sequence<DepthFrame>::Sequence(vector< vector<string> > paths)
: SequenceBase<DepthFrame>(paths)
{
}

Sequence<DepthFrame>::Sequence(const Sequence<DepthFrame>& rhs)
: SequenceBase<DepthFrame>(rhs)
{
    *this = rhs;
}

Sequence<DepthFrame>::Sequence(const Sequence<ColorDepthFrame>& rhs)
: SequenceBase<DepthFrame>()
{
    *this = rhs;
}

Sequence<DepthFrame>& Sequence<DepthFrame>::operator=(const Sequence<DepthFrame>& rhs)
{
    if (this != &rhs)
    {
        SequenceBase<DepthFrame>::operator=(rhs);
        m_ReferencePoints = rhs.m_ReferencePoints;
        m_Transformations = rhs.m_Transformations;
    }
    
    return *this;
}

Sequence<DepthFrame>& Sequence<DepthFrame>::operator=(const Sequence<ColorDepthFrame>& rhs)
{
    m_Path = rhs.getPath();
    m_Name = rhs.getName();
    
    // Paths copy
    vector<vector< pair<string,string> > > paths = rhs.getFramesPaths();
    m_Paths.resize(paths.size());
    for (int v = 0; v < paths.size(); v++)
        for (int f = 0; f < paths[v].size(); f++)
            m_Paths[v].push_back(paths[v][f].second);
    
    // Streams copy
    vector< vector<ColorDepthFrame::Ptr> > streams = rhs.getStreams();
    m_Streams.resize(streams.size());
    for (int v = 0; v < streams.size(); v++)
        for (int f = 0; f < streams[v].size(); f++)
            m_Streams[v].push_back(streams[v][f]);
    
    m_FrameIndices = rhs.getFrameIndices();
    m_Delays = rhs.getDelays();
    
    m_ReferencePoints = rhs.getReferencePoints();
    m_Transformations = rhs.getRegistrationTransformations();
    
    return *this;
}

void Sequence<DepthFrame>::setReferencePoints(vector<pcl::PointXYZ> points)
{
    m_ReferencePoints = points;
}

vector<pcl::PointXYZ> Sequence<DepthFrame>::getReferencePoints()
{
    return m_ReferencePoints;
}

void Sequence<DepthFrame>::setRegistrationTransformations(vector<Eigen::Matrix4f> transformations)
{
    m_Transformations = transformations;
}

//
// Sequence<ColorDepthFrame>
//

Sequence<ColorDepthFrame>::Sequence()
{
}

Sequence<ColorDepthFrame>::Sequence(vector<const char*> viewsNames) : m_ViewsNames(viewsNames)
{
    m_Paths.resize(viewsNames.size());
    m_Streams.resize(viewsNames.size());
    m_FrameIndices.resize(viewsNames.size(), -1);
    m_Delays.resize(viewsNames.size(), 0);
}

Sequence<ColorDepthFrame>::Sequence(vector<vector< pair<string,string> > > paths)
{
    m_Paths.clear();
    m_FrameIndices.clear();
    m_Delays.clear();
    
    m_Paths = paths;
    
    m_Streams.resize(m_Paths.size());
    m_FrameIndices.resize(m_Paths.size(), -1);
    m_Delays.resize(m_Paths.size(), 0);
}

Sequence<ColorDepthFrame>::Sequence(const Sequence& rhs)
{
    *this = rhs;
}

Sequence<ColorDepthFrame>& Sequence<ColorDepthFrame>::operator=(const Sequence<ColorDepthFrame>& rhs)
{
    if (this != &rhs)
    {
        m_Path = rhs.m_Path;
        m_Name = rhs.m_Name;
        m_Paths = rhs.m_Paths;
        m_Streams = rhs.m_Streams;
        m_FrameIndices = rhs.m_FrameIndices;
        m_Delays = rhs.m_Delays;
        
        m_ReferencePoints = rhs.m_ReferencePoints;
        m_Transformations = rhs.m_Transformations;
    }
    
    return *this;
}

string Sequence<ColorDepthFrame>::getPath() const
{
    return m_Path;
}

void Sequence<ColorDepthFrame>::setPath(string path)
{
    m_Path = path;
}

string Sequence<ColorDepthFrame>::getName() const
{
    return m_Name;
}

void Sequence<ColorDepthFrame>::setName(string name)
{
    m_Name = name;
}

int Sequence<ColorDepthFrame>::getNumOfViews() const
{
    return m_Paths.size();
}

void Sequence<ColorDepthFrame>::setViewsNames(vector<const char*> viewsNames)
{
    m_Paths.clear();
    m_FrameIndices.clear();
    m_Delays.clear();
    
    m_Paths.resize(viewsNames.size());
    m_FrameIndices.resize(viewsNames.size(), -1);
    m_Delays.resize(viewsNames.size(),0);
}

const char* Sequence<ColorDepthFrame>::getViewName(int v) const
{
    return m_ViewsNames[v];
}

vector<const char*> Sequence<ColorDepthFrame>::getViewsNames() const
{
    return m_ViewsNames;
}

void Sequence<ColorDepthFrame>::setFramesPaths(vector<vector< pair<string,string> > > paths)
{
    m_Paths = paths;
    
    m_FrameIndices.resize(m_Paths.size(), -1);
    m_Delays.resize(m_Paths.size(), 0);
}

vector<vector< pair<string,string> > > Sequence<ColorDepthFrame>::getFramesPaths() const
{
    return m_Paths;
}

void Sequence<ColorDepthFrame>::addStream(vector<ColorDepthFrame::Ptr> stream)
{
    m_Delays.clear();
    
    m_Streams.push_back(stream);
    m_FrameIndices.push_back(-1);
    
    m_Delays.resize(m_Streams.size());
}

void Sequence<ColorDepthFrame>::setStream(vector<ColorDepthFrame::Ptr> stream, int view)
{
    while (view >= m_Streams.size())
        m_Streams.push_back(vector<ColorDepthFrame::Ptr>());
    
    m_Streams[view] = stream;
}

void Sequence<ColorDepthFrame>::setStreams(vector<vector<ColorDepthFrame::Ptr> > streams)
{
    m_Delays.clear();
    m_FrameIndices.clear();
    
    m_Streams = streams;
    
    m_FrameIndices.resize(m_Streams.size(), -1);
    m_Delays.resize(m_Streams.size(), 0);
}

vector<vector<ColorDepthFrame::Ptr> > Sequence<ColorDepthFrame>::getStreams() const
{
    return m_Streams;
}

void Sequence<ColorDepthFrame>::addFramesFilePath(string colorPath, string depthPath, int view)
{
    while (view >= m_Paths.size())
    {
        m_Paths.push_back(vector< pair<string,string> >());
        m_Streams.push_back(vector<ColorDepthFrame::Ptr>());
        m_FrameIndices.push_back(-1);
        
        m_Delays.push_back(0);
    }
    
    m_Paths[view].push_back( pair<string,string>(colorPath,depthPath) );
}

void Sequence<ColorDepthFrame>::addFrameFilePath(vector< pair<string,string> > framePaths)
{
    for (int i = 0; i < m_Paths.size(); i++)
        m_Paths[i].push_back(framePaths[i]);
}

void Sequence<ColorDepthFrame>::addFrame(ColorDepthFrame::Ptr frame, int view)
{
    m_Streams[view].push_back(frame);
}

void Sequence<ColorDepthFrame>::addFrame(vector<ColorDepthFrame::Ptr> frame)
{
    for (int i = 0; i < m_Streams.size(); i++)
        m_Streams[i].push_back(frame[i]);
}

bool Sequence<ColorDepthFrame>::hasNextFrames(int step)
{
    for (int i = 0; i < m_Paths.size(); i++)
        if (m_FrameIndices[i] + m_Delays[i] + step > m_Paths[i].size() - 1)
            return false;
    
    return true;
}

bool Sequence<ColorDepthFrame>::hasPreviousFrames(int step)
{
    for (int i = 0; i < m_Paths.size(); i++)
        if (m_FrameIndices[i] + m_Delays[i] - step < -1)
            return false;
    
    return true;
}

vector<ColorDepthFrame::Ptr> Sequence<ColorDepthFrame>::nextFrames(int step)
{
    for (int v = 0; v < m_Paths.size(); v++)
        m_FrameIndices[v] += step;
    
    vector<ColorDepthFrame::Ptr> frames;
    for (int v = 0; v < m_Paths.size(); v++)
    {
        ColorDepthFrame::Ptr frame (new ColorDepthFrame);
        int delay = (m_Delays.size()) > 0 ? m_Delays[v] : 0;
        if (m_Streams[v].size() > 0)
            frame = m_Streams[v][m_FrameIndices[v] + delay]; // preallocation
        else
            readFrame(m_Paths[v], m_FrameIndices[v] + delay, *frame);
        
        frame->setPath(m_Paths[v][m_FrameIndices[v] + m_Delays[v]].first);
        frame->setFilename( getFilenameFromPath(frame->getPath()) );
        
        if (v < m_Transformations.size())
        {
            frame->setReferencePoint(m_ReferencePoints[v]);
            frame->setRegistrationTransformation(m_Transformations[v]);
        }
        
        frames.push_back(frame);
    }
    
    return frames;
}

vector<ColorDepthFrame::Ptr> Sequence<ColorDepthFrame>::previousFrames(int step)
{
    for (int v = 0; v < m_Paths.size(); v++)
        m_FrameIndices[v] -= step;
    
    vector<ColorDepthFrame::Ptr> frames;
    for (int v = 0; v < m_Paths.size(); v++)
    {
        ColorDepthFrame::Ptr frame (new ColorDepthFrame);
        if (m_Streams[v].size() > 0)
            frame = m_Streams[v][m_FrameIndices[v] + m_Delays[v]]; // preallocation
        else
            readFrame(m_Paths[v], m_FrameIndices[v] + m_Delays[v], *frame);
                                  
        frame->setPath(m_Paths[v][m_FrameIndices[v] + m_Delays[v]].first);
        frame->setFilename( getFilenameFromPath(frame->getPath()) );
        
        if (v < m_Transformations.size())
        {
            frame->setReferencePoint(m_ReferencePoints[v]);
            frame->setRegistrationTransformation(m_Transformations[v]);
        }
        
        frames.push_back(frame);
    }
    
    return frames;
}

vector<ColorDepthFrame::Ptr> Sequence<ColorDepthFrame>::getFrames(int i)
{
    vector<ColorDepthFrame::Ptr> frames;
    for (int v = 0; v < m_Paths.size(); v++)
    {
        ColorDepthFrame::Ptr frame (new ColorDepthFrame);
        if (m_Streams[v].size() > 0)
            frame = m_Streams[v][i];
        else
            readFrame(m_Paths[v], i, *frame);
        
        if (v < m_Transformations.size())
        {
            frame->setReferencePoint(m_ReferencePoints[v]);
            frame->setRegistrationTransformation(m_Transformations[v]);
        }
        
        frames.push_back(frame);
    }
    
    return frames;
}

void Sequence<ColorDepthFrame>::next(int step)
{
    for (int v = 0; v < m_Paths.size(); v++)
        m_FrameIndices[v] += step;
}

void Sequence<ColorDepthFrame>::previous(int step)
{
    for (int v = 0; v < m_Paths.size(); v++)
        m_FrameIndices[v] -= step;
}

vector<ColorDepthFrame::Ptr> Sequence<ColorDepthFrame>::getFrames(vector<string> filenames)
{
    vector<int> indices;
    return getFrames(filenames, indices);
}

vector<ColorDepthFrame::Ptr> Sequence<ColorDepthFrame>::getFrames(vector<string> filenames, vector<int>& indices)
{
    indices.clear();
    assert (filenames.size() == m_Paths.size());
    
    vector<ColorDepthFrame::Ptr> frames (m_Paths.size());
    indices.resize(m_Paths.size());
    for (int v = 0; v < m_Paths.size(); v++)
    {
        int idx = searchByName(m_Paths[v], filenames[v]);
        
        ColorDepthFrame::Ptr frame (new ColorDepthFrame);
        if (m_Streams[v].size() > 0)
            frame = m_Streams[v][idx];
        else
            readFrame(m_Paths[v], idx, *frame);
        
        if (v < m_Transformations.size())
        {
            frame->setReferencePoint(m_ReferencePoints[v]);
            frame->setRegistrationTransformation(m_Transformations[v]);
        }
        
        frames[v] = frame;
        indices[v] = idx;
    }
    
    return frames;
}

// Get current frames filenames
vector<string> Sequence<ColorDepthFrame>::getFramesFilenames()
{
    vector<string> filenames (m_Paths.size());
    
    for (int v = 0; v < filenames.size(); v++)
    {
        filenames[v] = getFilenameFromPath(m_Paths[v][m_FrameIndices[v] + m_Delays[v]].first);
    }
    
    return filenames;
}

int Sequence<ColorDepthFrame>::getMinNumOfFrames()
{
    int min = INT_MAX;
    for (int v = 0; v < getNumOfViews(); v++)
    {
        int numOfFramesOfView = m_Paths[v].size() - m_Delays[v];
        if (numOfFramesOfView < min)
            min = numOfFramesOfView;
    }
    
    return min;
}

void Sequence<ColorDepthFrame>::getNumOfFramesOfViews(vector<int>& numOfFrames)
{
    for (int v = 0; v < getNumOfViews(); v++)
        numOfFrames.push_back(m_Paths[v].size());
}

vector<int> Sequence<ColorDepthFrame>::getCurrentFramesID()
{
    vector<int> counters;
    for (int i = 0; i < m_Paths.size(); i++)
        counters.push_back(m_FrameIndices[i] + m_Delays[i]);
    
    return counters;
}

vector<float> Sequence<ColorDepthFrame>::getProgress()
{
    vector<float> progresses;
    for (int i = 0; i < m_Paths.size(); i++)
        progresses.push_back(((float) (m_FrameIndices[i] + m_Delays[i])) / m_Paths[i].size());
    
    return progresses;
}

vector<int> Sequence<ColorDepthFrame>::at()
{
    vector<int> frameDelayedCounter(m_Streams.size());
    
    for (int i = 0; i < m_Streams.size(); i++)
        frameDelayedCounter[i] = m_FrameIndices[i] + m_Delays[i];
    
    return frameDelayedCounter;
}

void Sequence<ColorDepthFrame>::setDelays(vector<int> delays)
{
    m_Delays = delays;
}

vector<int> Sequence<ColorDepthFrame>::getDelays() const
{
    return m_Delays;
}

void Sequence<ColorDepthFrame>::setFrameIndices(vector<int> indices)
{
    m_FrameIndices = indices;
}

vector<int> Sequence<ColorDepthFrame>::getFrameIndices() const
{
    return m_FrameIndices;
}

vector<int> Sequence<ColorDepthFrame>::getFrameCounters() const
{
    vector<int> counters (m_FrameIndices.size());
    
    for (int v = 0; v < m_FrameIndices.size(); v++)
        counters[v] = m_FrameIndices[v] + m_Delays[v];
    
    return counters;
}

void Sequence<ColorDepthFrame>::setReferencePoints(vector<pcl::PointXYZ> points)
{
    m_ReferencePoints = points;
}

vector<pcl::PointXYZ> Sequence<ColorDepthFrame>::getReferencePoints() const
{
    return m_ReferencePoints;
}

vector<Eigen::Matrix4f> Sequence<ColorDepthFrame>::getRegistrationTransformations() const
{
    return m_Transformations;
}

void Sequence<ColorDepthFrame>::setRegistrationTransformations(vector<Eigen::Matrix4f> transformations)
{
    m_Transformations = transformations;
}

void Sequence<ColorDepthFrame>::restart()
{
    m_FrameIndices.clear();
    m_FrameIndices.resize(m_Streams.size(), -1);
}

void Sequence<ColorDepthFrame>::allocate()
{
    m_Streams.clear();
    m_Streams.resize(m_Paths.size());
    
    for (int v = 0; v < m_Paths.size(); v++)
    {
        for (int f = 0; f < m_Paths[v].size(); f++)
        {
            ColorDepthFrame::Ptr frame (new ColorDepthFrame);
            
            int flags = CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR;
            frame->set( cv::imread(m_Paths[v][f].first, flags), cv::imread(m_Paths[v][f].second, flags) );
            
            m_Streams[v].push_back(frame);
        }
    }
}

void Sequence<ColorDepthFrame>::readFrame(vector< pair<string,string> > paths, int f, ColorDepthFrame& frame)
{
    assert (f < paths.size());
    
    int flags = CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR;
    cv::Mat color = cv::imread(paths[f].first, flags);
    cv::Mat depth = cv::imread(paths[f].second, flags);
    frame.set( color, depth );
}

//
// Template instanciation
// ------------------------------------------------------
template class SequenceBase<Frame>;
template class SequenceBase<ColorFrame>;
template class SequenceBase<DepthFrame>;

template class Sequence<Frame>;
// ------------------------------------------------------
