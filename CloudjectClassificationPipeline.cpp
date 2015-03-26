//
//  ClassificationPipeline.cpp
//  remedi3
//
//  Created by Albert Clapés on 12/03/15.
//
//

#include "CloudjectClassificationPipeline.h"
#include "constants.h"

#include "statistics.h"
#include "cvxtended.h"

#include <boost/assign/std/vector.hpp> // for 'operator+=()'

using namespace boost::assign; // bring 'operator+=()' into scope

// ----------------------------------------------------------------------------
// CloudjectClassificationPipelineBase
// ----------------------------------------------------------------------------

template<typename T>
CloudjectClassificationPipelineBase<T>::CloudjectClassificationPipelineBase()
: m_NormType(-1), m_ClassifierType(""), m_DimRedVariance(0), m_DimRedNumFeatures(0), m_BestValPerformance(0)
{
    
}

template<typename T>
CloudjectClassificationPipelineBase<T>::CloudjectClassificationPipelineBase(const CloudjectClassificationPipelineBase<T>& rhs)
{
    *this = rhs;
}

template<typename T>
CloudjectClassificationPipelineBase<T>::~CloudjectClassificationPipelineBase()
{
//    if (m_SvmModelsPtrs.size() > 0)
//    {
//        for (int i = 0; i < m_SvmModelsPtrs.size(); i++)
//            if (m_SvmModelsPtrs[i] != NULL)
//                delete m_SvmModelsPtrs[i];
//    }
}

template<typename T>
CloudjectClassificationPipelineBase<T>& CloudjectClassificationPipelineBase<T>::operator=(const CloudjectClassificationPipelineBase<T>& rhs)
{
    if (this != &rhs)
    {
        m_InputCloudjects = rhs.m_InputCloudjects;
        m_Categories = rhs.m_Categories;
        
        m_NormType = rhs.m_NormType;
        m_NormParams = rhs.m_NormParams;
        
        m_NumFolds = rhs.m_NumFolds;
        m_Partitions = rhs.m_Partitions;
        
        m_DimRedVariance = rhs.m_DimRedVariance;
        m_DimRedNumFeatures = rhs.m_DimRedNumFeatures;
        m_PCA = rhs.m_PCA;
        
        m_ClassifierType = rhs.m_ClassifierType;
        m_ClassifierValParams = rhs.m_ClassifierValParams;
        m_ClassifierParams = rhs.m_ClassifierParams;
        m_ClassifierModels = rhs.m_ClassifierModels;
        
        m_BestValPerformance = rhs.m_BestValPerformance;
    }
    
    return *this;
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::setInputCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::vector<const char*> categories)
{
    m_InputCloudjects = cloudjects;
    m_Categories = categories;
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::setValidation(int numFolds)
{
    m_NumFolds = numFolds;
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::setNormalization(int normType)
{
    m_NormType = normType;
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::setDimRedVariance(double var)
{
    m_DimRedVariance = var;
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::setDimRedNumFeatures(int n)
{
    m_DimRedNumFeatures = n;
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::normalize(cv::Mat X, cv::Mat& Xn, std::vector<cv::Mat>& parameters)
{
    Xn.create(X.rows, X.cols, cv::DataType<float>::type);
    
    // By columns
    if (m_NormType == CCPIPELINE_NORM_STD)
    {
        parameters.resize(2, cv::Mat(1, X.cols, cv::DataType<float>::type));
        
        for (int j = 0; j < X.cols; j++)
        {
            cv::Scalar mean, stddev;
            cv::meanStdDev(X.col(j), mean, stddev);
            
            cv::Mat col = (X.col(j) - mean.val[0]) / stddev.val[0];
            col.copyTo(Xn.col(j));
            
            parameters[0].at<float>(0,j) = mean.val[0];
            parameters[1].at<float>(0,j) = stddev.val[0];
        }
    }
    else if (m_NormType == CCPIPELINE_NORM_MINMAX)
    {
        parameters.resize(2, cv::Mat(1, X.cols, cv::DataType<float>::type));
        
        for (int j = 0; j < X.cols; j++)
        {
            double minVal, maxVal;
            cv::minMaxLoc(X.col(j), &minVal, &maxVal);
            
            cv::Mat col = (X.col(j) - minVal) / (maxVal - minVal);
            col.copyTo(Xn.col(j));
            
            parameters[0].at<float>(0,j) = minVal;
            parameters[1].at<float>(0,j) = maxVal;
        }
    }
    else if (m_NormType == CCPIPELINE_NORM_L1)
    {
        for (int i = 0; i < X.rows; i++)
            Xn.row(i) = (X.row(i) / cv::norm(X.row(i), cv::NORM_L1));
    }
    else if (m_NormType == CCPIPELINE_NORM_L2)
    {
        for (int i = 0; i < X.rows; i++)
            Xn.row(i) = (X.row(i) / cv::norm(X.row(i), cv::NORM_L2));
    }
    else
    {
        X.copyTo(Xn);
    }
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::normalize(cv::Mat X, std::vector<cv::Mat> parameters, cv::Mat& Xn)
{
    Xn.create(X.rows, X.cols, cv::DataType<float>::type);
    
    // By columns
    if (m_NormType == CCPIPELINE_NORM_STD)
    {
        for (int j = 0; j < X.cols; j++)
        {
            cv::Mat col = (X.col(j) - parameters[0].at<float>(0,j)) / parameters[1].at<float>(0,j);
            col.copyTo(Xn.col(j));
        }
    }
    else if (m_NormType == CCPIPELINE_NORM_MINMAX)
    {
        parameters.resize(2, cv::Mat(1, X.cols, cv::DataType<float>::type));
        
        for (int j = 0; j < X.cols; j++)
        {
            cv::Mat col = (X.col(j) - parameters[0].at<float>(0,j)) / (parameters[0].at<float>(0,j) - parameters[0].at<float>(0,j));
            col.copyTo(Xn.col(j));
        }
    }
    else if (m_NormType == CCPIPELINE_NORM_L1)
    {
        for (int i = 0; i < X.rows; i++)
            Xn.row(i) = (X.row(i) / cv::norm(X.row(i), cv::NORM_L1));
    }
    else if (m_NormType == CCPIPELINE_NORM_L2)
    {
        for (int i = 0; i < X.rows; i++)
            Xn.row(i) = (X.row(i) / cv::norm(X.row(i), cv::NORM_L2));
    }
    else
    {
        X.copyTo(Xn);
    }
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::setClassifierType(std::string classifierType)
{
    m_ClassifierType = classifierType;
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::setClassifierValidationParameters(std::vector<std::vector<float> > classifierParams)
{
    m_ClassifierValParams = classifierParams;
}

template<typename T>
float CloudjectClassificationPipelineBase<T>::validate(cv::Mat X, cv::Mat Y)
{
    std::vector<std::vector<float> > classifierValCombs;
    expandParameters(m_ClassifierValParams, classifierValCombs);
    
    // If not indicated explicitly the number of dims to keep after reduction
    if (m_DimRedNumFeatures == 0)
        m_DimRedNumFeatures = (int) (sqrtf(X.rows)); // just try to avoid the curse of dim
    
    std::vector<cv::Mat> P (m_NumFolds);
    // Model selection
    for (int t = 0; t < m_NumFolds; t++)
    {
        cv::Mat XTr, YTr, XTe, YTe;
        getTrainingFold(X, Y, t, XTr, YTr);
        getTestFold(X, Y, t, XTe, YTe);
        
        cv::Mat XnTr, XnTe;
        std::vector<cv::Mat> normParameters;
        normalize(XTr, XnTr, normParameters);
        normalize(XTe, normParameters, XnTe);
        
        cv::Mat XpTr, XpTe;
        if (m_DimRedVariance > 0 || m_DimRedNumFeatures > 0)
        {
            if (m_DimRedVariance > 0)
            {
//                std::cout << cv::Mat(XnTr, cv::Rect(0,0,XnTr.cols,5)) << std::endl;
                m_PCA.computeVar(XnTr, cv::Mat(), CV_PCA_DATA_AS_ROW, m_DimRedVariance);
//                std::cout<< m_PCA.eigenvectors << std::endl;
//                std::cout<< m_PCA.eigenvalues << std::endl;
            }
            else
                m_PCA(XnTr, cv::noArray(), CV_PCA_DATA_AS_ROW,
                      (m_DimRedNumFeatures < XnTr.cols) ? m_DimRedNumFeatures : XnTr.cols);

            XpTr = m_PCA.project(XnTr);
            XpTe = m_PCA.project(XnTe);
        }
        else
        {
            XpTr = XnTr; // no projection, just normalization
            XpTe = XnTe;
        }
        
        P[t].create(classifierValCombs.size(), YTr.cols, cv::DataType<float>::type);
        for (int i = 0; i < classifierValCombs.size(); i++)
        {
            for (int j = 0; j < YTr.cols; j++)
            {
                CvSVM::CvSVM svm;
                svm.train(XpTr, YTr.col(j), cv::Mat(), cv::Mat(),
                          CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 0, classifierValCombs[i][0], 0, classifierValCombs[i][1], 0, 0, 0,
                                      cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 500, 1e-4)));
                
                cv::Mat fTe; // predictions
                svm.predict(XpTe, fTe);

                float acc = accuracy(YTe.col(j), fTe);
                P[t].at<float>(i,j) = acc;
            }
        }
    }
    
    cv::Mat S (classifierValCombs.size(), Y.cols, cv::DataType<float>::type, cv::Scalar(0));
    for (int t = 0; t < m_NumFolds; t++)
        S += (P[t] / m_NumFolds);

    
    float avgValAcc = 0;
    
    double minVal, maxVal;
    cv::Point minIdx, maxIdx;
    
    std::vector<std::vector<float> > classifierParamsTmp (Y.cols);
    for (int j = 0; j < Y.cols; j++)
    {
        cv::minMaxLoc(S.col(j), &minVal, &maxVal, &minIdx, &maxIdx);
       
        avgValAcc += (maxVal / Y.cols);
        classifierParamsTmp[j] = classifierValCombs[maxIdx.y];
    }
    
    if (avgValAcc > m_BestValPerformance)
        m_ClassifierParams = classifierParamsTmp;
    
    return avgValAcc;
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::train(cv::Mat X, cv::Mat Y)
{
    // Final model training
    cv::Mat Xn;
    normalize(X, Xn, m_NormParams);
    
    cv::Mat Xp;
    if (m_DimRedVariance > 0 || m_DimRedNumFeatures > 0)
    {
        if (m_DimRedVariance > 0)
            m_PCA.computeVar(Xn, cv::noArray(), CV_PCA_DATA_AS_ROW, m_DimRedVariance);
        else
            m_PCA(Xn, cv::noArray(), CV_PCA_DATA_AS_ROW, m_DimRedNumFeatures);
        
        Xp = m_PCA.project(Xn);
    }
    else
    {
        Xp = Xn;
    }
    
    m_ClassifierModels.resize(Y.cols, NULL);
    for (int i = 0; i < Y.cols; i++)
    {
        if (m_ClassifierType == "svmrbf")
        {
            CvSVM* svm = new CvSVM;
            svm->train(Xp, Y.col(i), cv::Mat(), cv::Mat(),
                       CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 0, m_ClassifierParams[i][0], 0, m_ClassifierParams[i][1], 0, 0, 0, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 500, 1e-4)));
            m_ClassifierModels[i] = (void*) svm;
        }
    }
    
    m_InputCloudjects = boost::shared_ptr<std::list<Cloudject::Ptr> >(); // no longer needed
}

// Protected methods

template<typename T>
void CloudjectClassificationPipelineBase<T>::predict(cv::Mat X, cv::Mat& P, cv::Mat& F)
{
    cv::Mat Xn;
    normalize(X, m_NormParams, Xn);
    
    cv::Mat Xp;
    if (m_DimRedVariance > 0 || m_DimRedNumFeatures > 0)
        Xp = m_PCA.project(Xn);
    else
        Xp = Xn;
    
    P.create(Xp.rows, m_ClassifierModels.size(), cv::DataType<int>::type);
    F.create(Xp.rows, m_ClassifierModels.size(), cv::DataType<float>::type);
    for (int j = 0; j < m_ClassifierModels.size(); j++)
    {
        if (m_ClassifierType == "svmrbf")
        {
            for (int i = 0; i < Xp.rows; i++)
            {
                P.at<int>(i,j) = ((CvSVM*) m_ClassifierModels[j])->predict(Xp.row(i));
                F.at<float>(i,j) = ((CvSVM*) m_ClassifierModels[j])->predict(Xp.row(i), true);
            }
        }
    }
}

template<typename T>
std::vector<float> CloudjectClassificationPipelineBase<T>::evaluate(cv::Mat Y, cv::Mat F)
{
    std::vector<float> accs (Y.cols);
    
    for (int j = 0; j < Y.cols; j++)
        accs[j] = accuracy(Y.col(j), F.col(j));
    
    return accs;
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::getLabels(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::vector<const char*> categories, cv::Mat& Y)
{
    // Format the labels of the sample X
    Y.release();
    Y.create(cloudjects->size(), categories.size(), cv::DataType<int>::type);
    
    int i = 0;
    std::list<Cloudject::Ptr>::iterator it;
    for (it = cloudjects->begin(); it != cloudjects->end(); ++it, ++i)
        for (int j = 0; j < categories.size(); j++)
            Y.at<int>(i,j) = ((*it)->isRegionLabel(categories[j]) ? 1 : -1);
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::createValidationPartitions(cv::Mat Y, int numFolds, cv::Mat& partitions)
{
    // Create the partitions taking into account the labels (strafified CV)
    
    cv::Mat multipliers = cvx::powspace(0, Y.cols, 10);
    cv::Mat groups (Y.rows, 1, cv::DataType<int>::type, cv::Scalar(0));
    
    for (int i = 0; i < Y.rows; i++) for (int j = 0; j < Y.cols; j++)
        groups.at<int>(i,0) += Y.at<int>(i,j) * multipliers.at<int>(j,0);
    
    cvpartition(groups, numFolds, 42, partitions); // stratification
}

// Private methods

template<typename T>
void CloudjectClassificationPipelineBase<T>::getTrainingFold(cv::Mat X, cv::Mat Y, int t, cv::Mat& XTr, cv::Mat& YTr)
{
    cvx::indexMat(X, XTr, m_Partitions != t);
    cvx::indexMat(Y, YTr, m_Partitions != t);
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::getTestFold(cv::Mat X, cv::Mat Y, int t, cv::Mat& XTe, cv::Mat& YTe)
{
    cvx::indexMat(X, XTe, m_Partitions == t);
    cvx::indexMat(Y, YTe, m_Partitions == t);
}

// ----------------------------------------------------------------------------
// CloudjectClassificationPipeline<pcl::PFHRGBSignature250>
// ----------------------------------------------------------------------------

CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::CloudjectClassificationPipeline()
: CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>()
{
}

CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::CloudjectClassificationPipeline(const CloudjectClassificationPipeline<pcl::PFHRGBSignature250>& rhs)
: CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>(rhs)
{
    *this = rhs;
}

CloudjectClassificationPipeline<pcl::PFHRGBSignature250>& CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::operator=(const CloudjectClassificationPipeline<pcl::PFHRGBSignature250>& rhs)
{
    if (this != &rhs)
    {
        m_q = rhs.m_q;
        m_C = rhs.m_C;
    }
    
    return *this;
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::bowSampleFromCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, int q, cv::Mat& W, cv::Mat& Q)
{
    cv::Mat X, c;
    std::map<std::string,cv::Mat> XMap, cMap;
    
    int numInstances = 0;
    int numPointInstances = 0;
    std::map<std::string,int> numInstancesMap;
    std::map<std::string,int> numPointInstancesMap; // total no. descriptor points (no. rows of X and c actually)

    std::list<Cloudject::Ptr>::iterator it;
    for (it = cloudjects->begin(); it != cloudjects->end(); ++it)
    {
        (*it)->allocate();
    
        // General couting
        numInstances ++;
        numPointInstances += (*it)->getNumOfDescriptorPoints();
        
        // Category counting
        std::set<std::string> labels = (*it)->getRegionLabels();
        std::set<std::string>::iterator jt;
        for (jt = labels.begin(); jt != labels.end(); ++jt)
        {
            numInstancesMap[*jt]++;
            numPointInstancesMap[*jt] += (*it)->getNumOfDescriptorPoints();
        }
    }
    
    for (int k = 0; k < m_Categories.size(); k++)
    {
        XMap[m_Categories[k]].create(numPointInstancesMap[m_Categories[k]], 250, cv::DataType<float>::type);
        cMap[m_Categories[k]].create(numPointInstancesMap[m_Categories[k]], 1, cv::DataType<int>::type);
    }
    
    std::map<std::string,int> jPtrMap; // counter for descriptors
    std::map<std::string,int> iPtrMap; // counter (id) for cloudjects
    for (it = cloudjects->begin(); (it != cloudjects->end()); ++it)
    {
        pcl::PointCloud<pcl::PFHRGBSignature250>* pDescriptor = ((pcl::PointCloud<pcl::PFHRGBSignature250>*) (*it)->getDescriptor());
        std::set<std::string> labels = (*it)->getRegionLabels();
        
        std::set<std::string>::iterator jt;
        for (jt = labels.begin(); jt != labels.end(); ++jt)
        {
            for (int p = 0; (p < pDescriptor->points.size()); p++)
            {
                cv::Mat row (1, 250, cv::DataType<float>::type, pDescriptor->points[p].histogram);
                row.copyTo(XMap[*jt].row(jPtrMap[*jt]));
                
                cMap[*jt].at<int>(jPtrMap[*jt],0) = iPtrMap[*jt];
                
                jPtrMap[*jt]++;
            }
        
            iPtrMap[*jt]++;
        }
    }
    
    // Find quantization vectors
    std::vector<cv::Mat> QVec (m_Categories.size());
    for (int k = 0; k < m_Categories.size(); k++)
    {
        cv::Mat u;
        cv::kmeans(XMap[m_Categories[k]], q, u, cv::TermCriteria(), 3, cv::KMEANS_PP_CENTERS, QVec[k]);
    }
    cv::vconcat(QVec,Q);

    XMap.clear();
    cMap.clear();
    
    X.create(numPointInstances, 250, cv::DataType<float>::type);
    c.create(numPointInstances,   1, cv::DataType<int>::type);
    
    int jPtr = 0; // counter for descriptors
    int iPtr = 0; // counter (id) for cloudjects
    for (it = cloudjects->begin(); (it != cloudjects->end()); ++it)
    {
        pcl::PointCloud<pcl::PFHRGBSignature250>* pDescriptor = ((pcl::PointCloud<pcl::PFHRGBSignature250>*) (*it)->getDescriptor());
        
        for (int p = 0; (p < pDescriptor->points.size()); p++)
        {
            cv::Mat row (1, 250, cv::DataType<float>::type, pDescriptor->points[p].histogram);
            row.copyTo(X.row(jPtr));
            
            c.at<int>(jPtr,0) = iPtr;
            
            jPtr++;
        }
        iPtr++;
//        (*it)->release();
    }
    
    W.create(numInstances, Q.rows, cv::DataType<float>::type);
    for (int i = 0; i < X.rows; i++)
    {
        int closestQIdx;
        float closestDist = std::numeric_limits<float>::max();
        for (int k = 0; k < Q.rows; k++)
        {
            cv::Scalar d = cv::norm(X.row(i), Q.row(k), cv::NORM_L2);
            if (d.val[0] < closestDist)
            {
                closestQIdx = k;
                closestDist = d.val[0];
            }
        }
        
        W.at<float>(c.at<int>(i,0), closestQIdx)++;
    }
    
    
//    Waux[k].create(numInstances[k], q, cv::DataType<float>::type);
//    for (int i = 0; i < Waux[k].rows; i++)
//    {
//        int cIdx = c[k].at<int>(i,0);
//        Waux[k].at<float>(cIdx, u[k].at<int>(i,0))++;
//    }
//    
//    
//    cv::Mat X, c;
//    X.create(numPointInstances, 250, cv::DataType<float>::type);
//    c.create(numPointInstances,   1, cv::DataType<int>::type);
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::bowSampleFromCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, cv::Mat Q, cv::Mat& W)
{
    int numInstances = 0;
    
    std::list<Cloudject::Ptr>::iterator it;
    for (it = cloudjects->begin(); it != cloudjects->end(); ++it)
        numInstances += ((pcl::PointCloud<pcl::PFHRGBSignature250>*) (*it)->getDescriptor())->points.size();
    
    
    // Create a sample of cloudjects' PFHRGB descriptors (all concatenated)
    cv::Mat X (numInstances, 250, cv::DataType<float>::type);
    cv::Mat c (numInstances, 1, cv::DataType<int>::type);
    
    int j = 0; // counter for descriptors
    int i = 0; // counter (id) for cloudjects
    for (it = cloudjects->begin(); it != cloudjects->end(); ++it, ++i)
    {
        pcl::PointCloud<pcl::PFHRGBSignature250>* pDescriptor = ((pcl::PointCloud<pcl::PFHRGBSignature250>*) (*it)->getDescriptor());
        for (int k = 0; k < pDescriptor->points.size(); k++)
        {
            cv::Mat row (1, 250, cv::DataType<float>::type, pDescriptor->points[k].histogram);
            row.copyTo(X.row(j));
            // Track to which cloudject and partition the PFHRGB descriptor belongs
            c.at<int>(j,0) = i;
            
            j++;
        }
    }
    
    // Use previously computed quantization vectors
    W.create(cloudjects->size(), Q.rows, cv::DataType<float>::type);
    for (int i = 0; i < X.rows; i++)
    {
        int closestQIdx;
        float closestDist = std::numeric_limits<float>::max();
        for (int k = 0; k < Q.rows; k++)
        {
            cv::Scalar d = cv::norm(X.row(i), Q.row(k), cv::NORM_L2);
            if (d.val[0] < closestDist)
            {
                closestQIdx = k;
                closestDist = d.val[0];
            }
        }
        
        W.at<float>(c.at<int>(i,0), closestQIdx)++;
    }
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::setQuantizationValidationParameters(std::vector<int> qs)
{
    m_qValParam = qs;
}

float CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::validate()
{
    // Input data already set
    assert (m_InputCloudjects->size() > 0);
    
    cv::Mat Y;
    getLabels(m_InputCloudjects, m_Categories, Y);
    
    createValidationPartitions(Y, m_NumFolds, m_Partitions);
    
    m_ClassifierParams.resize(Y.cols);
    
    m_qValPerfs.resize(m_qValParam.size());
    for (int i = 0; i < m_qValParam.size(); i++)
    {
        // Bag-of-words aggregation since PFHRGBSignature is a local descriptor
        cv::Mat W, C;
        bowSampleFromCloudjects(m_InputCloudjects, m_qValParam[i], W, C);
        
        // Eventually train
        m_qValPerfs[i] = CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::validate(W,Y);
        if (m_qValPerfs[i] > m_BestValPerformance)
        {
            m_q = m_qValParam[i];
            m_BestValPerformance = m_qValPerfs[i];
        }
    }
    
    return m_BestValPerformance;
}

std::vector<float> CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::getQuantizationValidationPerformances()
{
    return m_qValPerfs;
}

int CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::getQuantizationParameter()
{
    return m_q;
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::train()
{
    cv::Mat Y;
    getLabels(m_InputCloudjects, m_Categories, Y);
    
    // Bag-of-words aggregation since PFHRGBSignature is a local descriptor
    cv::Mat W, Q;
    bowSampleFromCloudjects(m_InputCloudjects, m_q, W, m_C);
    
    // Eventually train
    CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::train(W,Y);
}

std::vector<float> CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::predict(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::list<std::vector<int> >& predictions, std::list<std::vector<float> >& distsToMargin)
{
    cv::Mat Y;
    getLabels(cloudjects, m_Categories, Y);
    
    // Bag-of-words aggregation since PFHRGBSignature is a local descriptor
    cv::Mat W, Q;
    bowSampleFromCloudjects(cloudjects, m_C, W);
    
    assert (cloudjects->size() == W.rows); // a word per cloudject
    
    // Eventually predict
    cv::Mat P,F;
    CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::predict(W,P,F);
    
    cvx::convert(P, predictions);
    cvx::convert(F, distsToMargin);
    
    // Prepare the output
    std::vector<float> accs = CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::evaluate(Y,P);
    
    return accs;
}

std::vector<int> CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::predict(Cloudject::Ptr cloudject, std::vector<float>& distsToMargin)
{
    boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects (new std::list<Cloudject::Ptr>);
    cloudjects->push_back(cloudject);
    
    cv::Mat W, Q;
    bowSampleFromCloudjects(cloudjects, m_C, W);
    
    assert (W.rows == 1);
    
    // Eventually predict
    cv::Mat P,F;
    CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::predict(W,P,F);
    
    cv::Mat f = F.row(0);
    distsToMargin = std::vector<float>(f.begin<float>(), f.end<float>());
    
    cv::Mat p = P.row(0);
    std::vector<int> predictions ( p.begin<int>(), p.end<int>());
    return predictions;
}

template class CloudjectClassificationPipeline<pcl::PFHRGBSignature250>;

template void CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::setInputCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::vector<const char*> categories);
template void CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::setNormalization(int normType);
template void CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::setDimRedVariance(double var);
template void CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::setDimRedNumFeatures(int n);
template void CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::setClassifierType(std::string classifierType);
template void CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::setValidation(int numFolds);
template void CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::setClassifierValidationParameters(std::vector<std::vector<float> > classifierParams);
template void CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::train(cv::Mat X, cv::Mat Y);
template float CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::validate(cv::Mat X, cv::Mat Y);
template void CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::predict(cv::Mat X, cv::Mat& P, cv::Mat& F);
