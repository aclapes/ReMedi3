#include "BackgroundSubtractor.h"

#include "DepthFrame.h"
#include "ColorFrame.h"

#include "cvxtended.h"

#include <boost/timer.hpp>


template<typename FrameT>
BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::BackgroundSubtractorBase()
: m_NumOfSamples(400), m_NumOfMixtureComponents(5), m_LearningRate(0.02), m_BackgroundRatio(0.999), m_VarThresholdGen(3), m_OpeningSize(0), m_bRecompute(true)
{ }

template<typename FrameT>
BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::BackgroundSubtractorBase(const BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>& rhs)
{
    *this = rhs;
}

template<typename FrameT>
BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>& BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::operator=(const BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>& rhs)
{
    if (this != &rhs)
    {
        m_XRes = rhs.m_XRes;
        m_YRes = rhs.m_YRes;
        
        m_NumOfSamples = rhs.m_NumOfSamples;
        
        m_NumOfMixtureComponents = rhs.m_NumOfMixtureComponents;
        m_LearningRate = rhs.m_LearningRate;
        m_BackgroundRatio = rhs.m_BackgroundRatio;
        m_VarThresholdGen = rhs.m_VarThresholdGen;
        m_OpeningSize = rhs.m_OpeningSize;
        
        m_bRecompute = rhs.m_bRecompute;
        m_Subtractors = rhs.m_Subtractors;
    }
    
    return *this;
}

template<typename FrameT>
void BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::setFramesResolution(int xres, int yres)
{
    m_XRes = xres;
    m_YRes = yres;

    m_bRecompute = true;
}

template<typename FrameT>
void BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::setNumOfMixtureComponents(int k)
{
    m_NumOfMixtureComponents = k;
    
    m_bRecompute = true;
}

template<typename FrameT>
void BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::setLearningRate(float rate)
{
    m_LearningRate = rate;
    
    m_bRecompute = true;
}

template<typename FrameT>
void BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::setBackgroundRatio(float bgratio)
{
    m_BackgroundRatio = bgratio;
    
    m_bRecompute = true;
}

template<typename FrameT>
void BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::setVarThresholdGen(float var)
{
    m_VarThresholdGen = var;
    
    m_bRecompute = true;
}

template<typename FrameT>
void BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::setOpeningSize(int level)
{
    m_OpeningSize = level;
}

template<typename FrameT>
int BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::getNumOfMixtureComponents()
{
    return m_NumOfMixtureComponents;
}

template<typename FrameT>
float BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::getLearningRate()
{
    return m_LearningRate;
}

template<typename FrameT>
float BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::getBackgroundRatio()
{
    return m_BackgroundRatio;
}

template<typename FrameT>
float BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::getVarThresholdGen()
{
    return m_VarThresholdGen;
}

template<typename FrameT>
int BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, FrameT>::getOpeningSize()
{
    return m_OpeningSize;
}


//
// BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>
//

BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::BackgroundSubtractor()
: BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, ColorFrame>(), m_bShadowsModeling(false)
{
    
}

BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::BackgroundSubtractor(const BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>& rhs)
: BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, ColorFrame>(rhs)
{
    *this = rhs;
}

BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>& BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::operator=(const BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>& rhs)
{
    if (this != &rhs)
    {
        BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, ColorFrame>::operator=(rhs);
        
        m_pSeq = rhs.m_pSeq;
        m_InputFrames = rhs.m_InputFrames;
    }
    
    return *this;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setBackgroundSequence(Sequence<ColorFrame>::Ptr pSeq, int n)
{
    m_bRecompute = true;
    m_pSeq = pSeq;
    m_NumOfSamples = n;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setInputFrames(vector<ColorFrame::Ptr> frames)
{
    m_InputFrames = frames;
}

int BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::getNumOfViews() const
{
    return m_pSeq->getNumOfViews();
}

vector<ColorFrame::Ptr> BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::getInputFrames() const
{
    return m_InputFrames;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setShadowsModeling(bool flag)
{
    m_bShadowsModeling = flag;
    
    m_bRecompute = true;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::model()
{
    if (m_bRecompute)
    {
        m_Subtractors.clear();
        m_Subtractors.resize(m_pSeq->getNumOfViews());
        for (int v = 0; v < m_pSeq->getNumOfViews(); v++)
        {
            m_Subtractors[v] = cv::BackgroundSubtractorMOG2(m_NumOfSamples, m_NumOfMixtureComponents, m_bShadowsModeling);
            m_Subtractors[v].set("backgroundRatio", m_BackgroundRatio);
            m_Subtractors[v].set("varThresholdGen", m_VarThresholdGen);
        }
        
        vector<ColorFrame::Ptr> colorFrames;
        
        int t = 1;
        m_pSeq->restart();
        while (t <= m_NumOfSamples && m_pSeq->hasNextFrames())
        {
            colorFrames = m_pSeq->nextFrames();
            
            vector<cv::Mat> images (m_pSeq->getNumOfViews());
            vector<cv::Mat> masks (images.size());
            
            for (int v = 0; v < m_pSeq->getNumOfViews(); v++)
            {
                cv::Mat image;
                if (m_bShadowsModeling) image = colorFrames[v]->get();
                else image = bgr2hs(colorFrames[v]->get());
                
                m_Subtractors[v](image, masks[v], (m_LearningRate == 0 || t == 1) ? (1.f/t) : m_LearningRate);
            }
            
            t++;
        }
        
        m_bRecompute = false;
    }
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::subtract(vector<cv::Mat>& fgMasks)
{
    fgMasks.clear();
    fgMasks.resize(m_InputFrames.size());
	for (int v = 0; v < m_InputFrames.size(); v++)
        subtract(m_Subtractors[v], m_InputFrames[v], fgMasks[v]);
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::subtract(cv::BackgroundSubtractorMOG2& subtractor, ColorFrame::Ptr colorFrame, cv::Mat& fgMask)
{
    cv::Mat image;
    if (m_bShadowsModeling) image = colorFrame->get();
    else image = bgr2hs(colorFrame->get());
    
    cv::Mat mask;
    subtractor(image, mask, 0);
    
    fgMask = (mask == 255); // no background (0), no shadows (127)

    if (m_OpeningSize < 0)
        cvx::close(fgMask, abs(m_OpeningSize), fgMask);
    else
        cvx::open(fgMask, m_OpeningSize, fgMask);
}

//
// BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>
//

BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::BackgroundSubtractor()
: BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, DepthFrame>()
{
    
}

BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::BackgroundSubtractor(const BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>& rhs)
: BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, DepthFrame>(rhs)
{
    *this = rhs;
}

BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>& BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::operator=(const BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>& rhs)
{
    if (this != &rhs)
    {
        BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, DepthFrame>::operator=(rhs);
        
        m_pSeq = rhs.m_pSeq;
        m_InputFrames = rhs.m_InputFrames;
    }
    
    return *this;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::setBackgroundSequence(Sequence<DepthFrame>::Ptr pSeq, int n)
{
    m_pSeq = pSeq;
    m_NumOfSamples = n;
    
    m_bRecompute = true;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::setInputFrames(vector<DepthFrame::Ptr> frames)
{
    m_InputFrames = frames;
}

int BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::getNumOfViews() const
{
    return m_pSeq->getNumOfViews();
}

vector<DepthFrame::Ptr> BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::getInputFrames() const
{
    return m_InputFrames;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::model()
{
    if (m_bRecompute)
    {
        m_Subtractors.clear();
        m_Subtractors.resize(m_pSeq->getNumOfViews());
        for (int v = 0; v < m_pSeq->getNumOfViews(); v++)
        {
            m_Subtractors[v] = cv::BackgroundSubtractorMOG2(m_NumOfSamples, m_NumOfMixtureComponents, false);
            m_Subtractors[v].set("backgroundRatio", m_BackgroundRatio);
            m_Subtractors[v].set("varThresholdGen", m_VarThresholdGen);
        }
        
        vector<DepthFrame::Ptr> depthFrames;
        
        int t = 1;
        m_pSeq->restart();
        while (t <= m_NumOfSamples && m_pSeq->hasNextFrames())
        {
            depthFrames = m_pSeq->nextFrames();
            
            vector<cv::Mat> depthMaps (m_pSeq->getNumOfViews());
            vector<cv::Mat> depthMasks (depthMaps.size());
            for (int v = 0; v < m_pSeq->getNumOfViews(); v++)
            {
                cv::Mat depthMap;
                depthFrames[v]->get().convertTo(depthMap, cv::DataType<float>::type);
                m_Subtractors[v](depthMap, depthMasks[v], (m_LearningRate == 0 || t == 1) ? (1.f/t) : m_LearningRate);
            }
            
            t++;
        }
        
        m_bRecompute = false;
    }
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::subtract(vector<cv::Mat>& fgMasks)
{
    fgMasks.clear();
    fgMasks.resize(m_InputFrames.size());
	for (int v = 0; v < m_InputFrames.size(); v++)
        subtract(m_Subtractors[v], m_InputFrames[v], fgMasks[v]);
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::subtract(cv::BackgroundSubtractorMOG2& subtractor, DepthFrame::Ptr depthFrame, cv::Mat& fgMask)
{
    cv::Mat depthMap;
    depthFrame->get().convertTo(depthMap, cv::DataType<float>::type);
    
    cv::Mat mask;
    subtractor(depthMap, mask, 0);
    
    fgMask = (mask == 255); // no background (0), no shadows (127)
    
    if (m_OpeningSize < 0)
        cvx::close(fgMask, abs(m_OpeningSize), fgMask);
    else
        cvx::open(fgMask, m_OpeningSize, fgMask);
}

//
// BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>
//

BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::BackgroundSubtractor()
: BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>(), BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>()
{
    
}

BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::BackgroundSubtractor(const BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>& rhs)
: BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>(rhs), BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>(rhs)
{
    *this = rhs;
}

BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>& BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::operator=(const BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>& rhs)
{
    if (this != &rhs)
    {
        // Parents' initialization
        BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::operator=(rhs);
        BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::operator=(rhs);
        
        // Its own
        m_XRes = rhs.m_XRes;
        m_YRes = rhs.m_YRes;
        
        m_NumOfSamples = rhs.m_NumOfSamples;
        
        m_NumOfMixtureComponents = rhs.m_NumOfMixtureComponents;
        m_LearningRate = rhs.m_LearningRate;
        m_BackgroundRatio = rhs.m_BackgroundRatio;
        m_VarThresholdGen = rhs.m_VarThresholdGen;
        m_OpeningSize = rhs.m_OpeningSize;
        
        m_bRecompute =rhs.m_bRecompute;
        m_Subtractors = rhs.m_Subtractors;
        
        m_Modality = rhs.m_Modality;
        
        m_pSeq = rhs.m_pSeq;
        m_InputFrames = rhs.m_InputFrames;
    }
    
    return *this;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::setBackgroundSequence(Sequence<ColorDepthFrame>::Ptr pSeq, int n)
{
    // Parents' initialization
    Sequence<ColorFrame>::Ptr pColorSeq (new Sequence<ColorFrame>);
    Sequence<DepthFrame>::Ptr pDepthSeq (new Sequence<DepthFrame>);
    
    *pColorSeq = *pSeq;
    *pDepthSeq = *pSeq;
    
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setBackgroundSequence(pColorSeq, n);
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::setBackgroundSequence(pDepthSeq, n);
    
    // Its own
    m_bRecompute = true;
    m_pSeq = pSeq;
    m_NumOfSamples = n;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::setInputFrames(vector<ColorDepthFrame::Ptr> frames)
{
    // Parents' initialization
    vector<ColorFrame::Ptr> colorFrames (frames.size());
    vector<DepthFrame::Ptr> depthFrames (frames.size());
    
    for (int v = 0; v < frames.size(); v++)
    {
        colorFrames[v] = frames[v];
        depthFrames[v] = frames[v];
    }
    
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setInputFrames(colorFrames);
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::setInputFrames(depthFrames);
    
    // Its own
    m_InputFrames = frames;
}

int BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::getNumOfViews() const
{
    return m_pSeq->getNumOfViews();
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::setFramesResolution(int xres, int yres)
{
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setFramesResolution(xres, yres);
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::setFramesResolution(xres, yres);
    
    m_XRes = xres;
    m_YRes = yres;
    
    m_bRecompute = true;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::setNumOfMixtureComponents(int k)
{
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setNumOfMixtureComponents(k);
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::setNumOfMixtureComponents(k);
    
    m_NumOfMixtureComponents = k;
    
    m_bRecompute = true;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::setLearningRate(float rate)
{
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setLearningRate(rate);
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::setLearningRate(rate);
    
    m_LearningRate = rate;
    
    m_bRecompute = true;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::setBackgroundRatio(float bgratio)
{
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setBackgroundRatio(bgratio);
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::setBackgroundRatio(bgratio);
    
    m_BackgroundRatio = bgratio;
    
    m_bRecompute = true;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::setVarThresholdGen(float var)
{
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setVarThresholdGen(var);
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::setVarThresholdGen(var);
    
    m_VarThresholdGen = var;
    
    m_bRecompute = true;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::setOpeningSize(int level)
{
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setOpeningSize(level);
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::setOpeningSize(level);
    
    m_OpeningSize = level;
    
    m_bRecompute = true;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::setModality(int m)
{
    m_Modality = m;
    
    BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setShadowsModeling( m == 2 ); // ReMedi::COLOR_WITH_SHADOWS
}

int BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::getModality() const
{
    return m_Modality;
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::model()
{
    if ( m_Modality == 0 )
    {
        BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setShadowsModeling(false);
        BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::model();
    }
    else if ( m_Modality == 1 )
    {
        BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::model();
    }
    else if ( m_Modality == 2 )
    {
        BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::setShadowsModeling(true);
        BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::model();
    }
    else if ( m_Modality == 3 )
    {
        _model();
    }
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::_model()
{
    if (m_bRecompute)
    {
        m_Subtractors.clear();
        m_Subtractors.resize(m_pSeq->getNumOfViews());
        for (int v = 0; v < m_pSeq->getNumOfViews(); v++)
        {
            m_Subtractors[v] = cv::BackgroundSubtractorMOG2(m_NumOfSamples, m_NumOfMixtureComponents, false);
            m_Subtractors[v].set("backgroundRatio", m_BackgroundRatio);
            m_Subtractors[v].set("varThresholdGen", m_VarThresholdGen);
        }
        
        vector<ColorDepthFrame::Ptr> frames;
        
        int t = 1;
        m_pSeq->restart();
        while (t <= m_NumOfSamples && m_pSeq->hasNextFrames())
        {
            frames = m_pSeq->nextFrames();
            
            vector<cv::Mat> masks (frames.size());
            for (int v = 0; v < m_pSeq->getNumOfViews(); v++)
            {
                cv::Mat colorImage, depthMap;
                frames[v]->getColor().convertTo(colorImage, cv::DataType<float>::type);
                frames[v]->getDepth().convertTo(depthMap, cv::DataType<float>::type);
                
                cv::Mat m = bgrd2hsd(colorImage, depthMap);
                m_Subtractors[v](m, masks[v], (m_LearningRate == 0 || t == 1) ? (1.f/t) : m_LearningRate);
            }
            
            t++;
        }
        
        m_bRecompute = false;
    }
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::subtract(vector<cv::Mat>& fgMasks)
{
    if ( m_Modality == 0 || m_Modality == 2 )
    {
        BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>::subtract( fgMasks );
    }
    else if ( m_Modality == 1 )
    {
        BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>::subtract( fgMasks );
    }
    else if ( m_Modality == 3 )
    {
        fgMasks.clear();
        fgMasks.resize(m_InputFrames.size());
        for (int v = 0; v < m_InputFrames.size(); v++)
            subtract( m_Subtractors[v], m_InputFrames[v], fgMasks[v] );
    }
}

void BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorDepthFrame>::subtract(cv::BackgroundSubtractorMOG2& subtractor, ColorDepthFrame::Ptr frame, cv::Mat& fgMask)
{
    cv::Mat colorImage, depthMap;
    frame->getColor().convertTo(colorImage, cv::DataType<float>::type);
    frame->getDepth().convertTo(depthMap, cv::DataType<float>::type);
    
    cv::Mat image = bgrd2hsd(colorImage, depthMap);
    cv::Mat mask;
    subtractor(image, mask, 0);

    fgMask = (mask == 255); // no background (0), no shadows (127)
    
    if (m_OpeningSize < 0)
        cvx::close(fgMask, abs(m_OpeningSize), fgMask);
    else
        cvx::open(fgMask, m_OpeningSize, fgMask);
}

// Template instantiaions

//template class BackgroundSubtractorBase<cv::BackgroundSubtractorGMG, Frame>;
//template class BackgroundSubtractorBase<cv::BackgroundSubtractorGMG, ColorFrame>;
//template class BackgroundSubtractorBase<cv::BackgroundSubtractorGMG, DepthFrame>;
//
//template class BackgroundSubtractor<cv::BackgroundSubtractorGMG, Frame>;
//template class BackgroundSubtractor<cv::BackgroundSubtractorGMG, ColorFrame>;
//template class BackgroundSubtractor<cv::BackgroundSubtractorGMG, DepthFrame>;

template class BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, Frame>;
template class BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, ColorFrame>;
template class BackgroundSubtractorBase<cv::BackgroundSubtractorMOG2, DepthFrame>;

template class BackgroundSubtractor<cv::BackgroundSubtractorMOG2, Frame>;
//template class BackgroundSubtractor<cv::BackgroundSubtractorMOG2, ColorFrame>;
//template class BackgroundSubtractor<cv::BackgroundSubtractorMOG2, DepthFrame>;
