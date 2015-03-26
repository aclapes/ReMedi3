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
boost::shared_ptr<std::list<Cloudject::Ptr> > CloudjectClassificationPipelineBase<T>::getInputCloudjects()
{
    return m_InputCloudjects;
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
cv::Mat CloudjectClassificationPipelineBase<T>::validate(cv::Mat XTr, cv::Mat XTe, cv::Mat YTr, cv::Mat YTe)
{
    std::vector<std::vector<float> > classifierValCombs;
    expandParameters(m_ClassifierValParams, classifierValCombs);
    
    cv::Mat P (classifierValCombs.size(), YTr.cols, cv::DataType<float>::type);
    
    for (int i = 0; i < classifierValCombs.size(); i++)
    {
        for (int j = 0; j < YTr.cols; j++)
        {
            CvSVM::CvSVM svm;
            svm.train(XTr, YTr.col(j), cv::Mat(), cv::Mat(),
                      CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 0, classifierValCombs[i][0], 0, classifierValCombs[i][1], 0, 0, 0, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 500, 1e-4)));
            
            cv::Mat fTe; // predictions
            svm.predict(XTe, fTe);
            
            float acc = accuracy(YTe.col(j), fTe);
            P.at<float>(i,j) = acc;
        }
    }
    
    return P;
    
//    // Update the performance average
//    if (m_P.empty())
//    {
//        m_P.create(classifierValCombs.size(), YTr.cols, cv::DataType<float>::type);
//        m_P.setTo(0);
//    }
//    m_P += (P / m_NumFolds);
//    
//    // Calculate
//    float valAcc = 0; // average among categories
//    
//    std::vector<std::vector<float> > classifierParamsTmp (YTr.cols);
//    for (int j = 0; j < YTr.cols; j++)
//    {
//        double minVal, maxVal;
//        cv::Point minIdx, maxIdx;
//        cv::minMaxLoc(m_P.col(j), &minVal, &maxVal, &minIdx, &maxIdx);
//        
//        valAcc += (maxVal / YTr.cols);
//        classifierParamsTmp[j] = classifierValCombs[maxIdx.y];
//    }
//    
//    if (valAcc > m_BestValPerformance)
//        m_ClassifierParams = classifierParamsTmp;
//    
//    return valAcc;
}

//template<typename T>
//float CloudjectClassificationPipelineBase<T>::validate(cv::Mat X, cv::Mat Y)
//{
//    std::vector<std::vector<float> > classifierValCombs;
//    expandParameters(m_ClassifierValParams, classifierValCombs);
//    
//    // If not indicated explicitly the number of dims to keep after reduction
//    if (m_DimRedNumFeatures == 0)
//        m_DimRedNumFeatures = (int) (sqrtf(X.rows)); // just try to avoid the curse of dim
//    
//    std::vector<cv::Mat> P (m_NumFolds);
//    // Model selection
//    for (int t = 0; t < m_NumFolds; t++)
//    {
//        cv::Mat XTr, YTr, XTe, YTe;
//        getTrainingFold(X, Y, t, XTr, YTr);
//        getTestFold(X, Y, t, XTe, YTe);
//        
//        cv::Mat XnTr, XnTe;
//        std::vector<cv::Mat> normParameters;
//        normalize(XTr, XnTr, normParameters);
//        normalize(XTe, normParameters, XnTe);
//        
//        cv::Mat XpTr, XpTe;
//
//        
//        P[t].create(classifierValCombs.size(), YTr.cols, cv::DataType<float>::type);
//        for (int i = 0; i < classifierValCombs.size(); i++)
//        {
//            for (int j = 0; j < YTr.cols; j++)
//            {
//                CvSVM::CvSVM svm;
//                svm.train(XpTr, YTr.col(j), cv::Mat(), cv::Mat(),
//                          CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 0, classifierValCombs[i][0], 0, classifierValCombs[i][1], 0, 0, 0,
//                                      cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 500, 1e-4)));
//                
//                cv::Mat fTe; // predictions
//                svm.predict(XpTe, fTe);
//
//                float acc = accuracy(YTe.col(j), fTe);
//                P[t].at<float>(i,j) = acc;
//            }
//        }
//    }
//    
//    cv::Mat S (classifierValCombs.size(), Y.cols, cv::DataType<float>::type, cv::Scalar(0));
//    for (int t = 0; t < m_NumFolds; t++)
//        S += (P[t] / m_NumFolds);
//
//    
//    float avgValAcc = 0;
//    
//    double minVal, maxVal;
//    cv::Point minIdx, maxIdx;
//    
//    std::vector<std::vector<float> > classifierParamsTmp (Y.cols);
//    for (int j = 0; j < Y.cols; j++)
//    {
//        cv::minMaxLoc(S.col(j), &minVal, &maxVal, &minIdx, &maxIdx);
//       
//        avgValAcc += (maxVal / Y.cols);
//        classifierParamsTmp[j] = classifierValCombs[maxIdx.y];
//    }
//    
//    if (avgValAcc > m_BestValPerformance)
//        m_ClassifierParams = classifierParamsTmp;
//    
//    return avgValAcc;
//}

template<typename T>
void CloudjectClassificationPipelineBase<T>::train(cv::Mat X, cv::Mat Y)
{
    // Final model training
    m_ClassifierModels.resize(Y.cols, NULL);
    for (int i = 0; i < Y.cols; i++)
    {
        if (m_ClassifierType == "svmrbf")
        {
            CvSVM* svm = new CvSVM;
            svm->train(X, Y.col(i), cv::Mat(), cv::Mat(),
                       CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 0, m_ClassifierParams[i][0], 0, m_ClassifierParams[i][1], 0, 0, 0, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 500, 1e-4)));
            m_ClassifierModels[i] = (void*) svm;
        }
    }
//    m_InputCloudjects = boost::shared_ptr<std::list<Cloudject::Ptr> >(); // no longer needed
}

// Protected methods

template<typename T>
void CloudjectClassificationPipelineBase<T>::predict(cv::Mat X, cv::Mat& P, cv::Mat& F)
{
    P.create(X.rows, m_ClassifierModels.size(), cv::DataType<int>::type);
    F.create(X.rows, m_ClassifierModels.size(), cv::DataType<float>::type);
    for (int j = 0; j < m_ClassifierModels.size(); j++)
    {
        if (m_ClassifierType == "svmrbf")
        {
            for (int i = 0; i < X.rows; i++)
            {
                P.at<int>(i,j)   = ((CvSVM*) m_ClassifierModels[j])->predict(X.row(i));
                F.at<float>(i,j) = ((CvSVM*) m_ClassifierModels[j])->predict(X.row(i), true);
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

template<typename T>
void CloudjectClassificationPipelineBase<T>::reduce(cv::Mat X, cv::Mat& Xr, cv::PCA& pca)
{
    if (m_DimRedNumFeatures == 0 && m_DimRedVariance == 0)
    {
        Xr = X;
    }
    else
    {
        if (m_DimRedNumFeatures > 0)
            pca(X, cv::noArray(), CV_PCA_DATA_AS_ROW, m_DimRedNumFeatures);
        else if (m_DimRedVariance > 0)
            pca.computeVar(X, cv::Mat(), CV_PCA_DATA_AS_ROW, m_DimRedVariance);
        Xr = pca.project(X);
    }
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::reduce(cv::Mat X, cv::PCA pca, cv::Mat& Xr)
{
    if (m_DimRedNumFeatures == 0 && m_DimRedVariance == 0)
        Xr = X;
    else
        Xr = pca.project(X);
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::getTrainingCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, int t, std::list<Cloudject::Ptr>& cloudjectsTr)
{
    cloudjectsTr.clear();

    int i = 0;
    std::list<Cloudject::Ptr>::iterator it;
    for (it = cloudjects->begin(); it != cloudjects->end(); ++it, ++i)
        if (m_Partitions.at<int>(i,0) != t)
            cloudjectsTr.push_back(*it);
}

template<typename T>
void CloudjectClassificationPipelineBase<T>::getTestCloudjects(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, int t, std::list<Cloudject::Ptr>& cloudjectsTe)
{
    cloudjectsTe.clear();
    
    int i = 0;
    std::list<Cloudject::Ptr>::iterator it;
    for (it = cloudjects->begin(); it != cloudjects->end(); ++it, ++i)
        if (m_Partitions.at<int>(i,0) == t)
            cloudjectsTe.push_back(*it);
}

// ----------------------------------------------------------------------------
// CloudjectClassificationPipeline<pcl::PFHRGBSignature250>
// ----------------------------------------------------------------------------

CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::CloudjectClassificationPipeline()
: CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>(), m_bCentersStratification(false), m_bPerStrateReduction(false), m_bGlobalQuantization(false)
{
}

CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::CloudjectClassificationPipeline(const CloudjectClassificationPipeline<pcl::PFHRGBSignature250>& rhs) : CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>(rhs), m_bCentersStratification(false), m_bPerStrateReduction(false), m_bGlobalQuantization(false)
{
    *this = rhs;
}

CloudjectClassificationPipeline<pcl::PFHRGBSignature250>& CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::operator=(const CloudjectClassificationPipeline<pcl::PFHRGBSignature250>& rhs)
{
    if (this != &rhs)
    {
        m_q = rhs.m_q;
        m_C = rhs.m_C;
        
        m_bGlobalQuantization = rhs.m_bGlobalQuantization;
        
        m_bCentersStratification = rhs. m_bCentersStratification;
        m_bPerStrateReduction = rhs.m_bPerStrateReduction;
    }
    
    return *this;
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::setPerFoldQuantization(bool bGlobalQuantization)
{
    m_bGlobalQuantization = bGlobalQuantization;
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::setCentersStratification(bool bCentersStratification)
{
    m_bCentersStratification = bCentersStratification;
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::setPerCategoryReduction(bool bPerStrateReduction)
{
    m_bPerStrateReduction = bPerStrateReduction;
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::cloudjectsToPointsSample(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, cv::Mat& X, cv::Mat& c)
{
    c.create(cloudjects->size(), 1, cv::DataType<int>::type);

    int numPointsInstances = 0;
    
    int i = 0;
    std::list<Cloudject::Ptr>::iterator cjIt;
    for (cjIt = cloudjects->begin(); cjIt != cloudjects->end(); ++cjIt, ++i)
    {
        (*cjIt)->allocate();
        
        c.at<int>(i,0) = ((pcl::PointCloud<pcl::PFHRGBSignature250>*) (*cjIt)->getDescriptor())->points.size();
        numPointsInstances += c.at<int>(i,0);
    }
    
    // Create a sample of cloudjects' PFHRGB descriptors (all concatenated)
    X.create(numPointsInstances, 250, cv::DataType<float>::type);
    
    int j = 0; // counter for descriptors
    for (cjIt = cloudjects->begin(); cjIt != cloudjects->end(); ++cjIt)
    {
        pcl::PointCloud<pcl::PFHRGBSignature250>* pDescriptor = ((pcl::PointCloud<pcl::PFHRGBSignature250>*) (*cjIt)->getDescriptor());
        for (int k = 0; k < pDescriptor->points.size(); k++)
        {
            cv::Mat row (1, 250, cv::DataType<float>::type, pDescriptor->points[k].histogram);
            row.copyTo(X.row(j++));
        }
        (*cjIt)->release();
    }
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::cloudjectsToPointsSample(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::map<std::string,cv::Mat>& X)
{
    X.clear();
    
    std::map<std::string,int> numPointInstancesMap; // total no. descriptor points (no. rows of X and c actually)
    
    std::list<Cloudject::Ptr>::iterator cjIt;
    for (cjIt = cloudjects->begin(); cjIt != cloudjects->end(); ++cjIt)
    {
        (*cjIt)->allocate(); // release before finishing the method calling
        
        // Category counting
        std::set<std::string> labels = (*cjIt)->getRegionLabels();
        std::set<std::string>::iterator lblIt;
        for (lblIt = labels.begin(); lblIt != labels.end(); ++lblIt)
            numPointInstancesMap[*lblIt] += (*cjIt)->getNumOfDescriptorPoints();
    }
    
    for (int k = 0; k < m_Categories.size(); k++)
        X[m_Categories[k]].create(numPointInstancesMap[m_Categories[k]], 250, cv::DataType<float>::type);
    
    std::map<std::string,int> ptrsMap; // counter for descriptors
    for (cjIt = cloudjects->begin(); cjIt != cloudjects->end(); ++cjIt)
    {
        pcl::PointCloud<pcl::PFHRGBSignature250>* pDescriptor = ((pcl::PointCloud<pcl::PFHRGBSignature250>*) (*cjIt)->getDescriptor());
        std::set<std::string> labels = (*cjIt)->getRegionLabels();
        
        std::set<std::string>::iterator lblIt;
        for (lblIt = labels.begin(); lblIt != labels.end(); ++lblIt)
        {
            for (int p = 0; (p < pDescriptor->points.size()); p++)
            {
                cv::Mat row (1, 250, cv::DataType<float>::type, pDescriptor->points[p].histogram);
                row.copyTo( X[*lblIt].row( ptrsMap[*lblIt]++ ) );
            }
        }
        
        (*cjIt)->release(); // here the release
    }
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::bow(cv::Mat X, int q, cv::Mat& Q)
{
    // Find quantization vectors
    cv::Mat u;
    cv::kmeans(X, q, u, cv::TermCriteria(), 3, cv::KMEANS_PP_CENTERS, Q);
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::bow(std::map<std::string,cv::Mat> X, int q, cv::Mat& Q)
{
    // Find quantization vectors
    std::vector<cv::Mat> QVec (m_Categories.size());
    
    for (int k = 0; k < m_Categories.size(); k++)
    {
        cv::Mat u;
        cv::kmeans(X[m_Categories[k]], q, u, cv::TermCriteria(), 3, cv::KMEANS_PP_CENTERS, QVec[k]);
    }
    
    cv::vconcat(QVec,Q);
}

void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::bow(cv::Mat X, cv::Mat c, cv::Mat Q, cv::Mat& W)
{
    // Indexing vector, associating X rows to global instance
    std::vector<cv::Mat> countings (c.rows);
    for (int i = 0; i < c.rows; i++)
        countings[i] = cv::Mat(c.at<int>(i,0), 1, c.type(), cv::Scalar(i));
    
    cv::Mat I;
    cv::vconcat(countings, I);
    
    // Word construction
    W.create(c.rows, Q.rows, cv::DataType<float>::type);
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
        
        W.at<float>(I.at<int>(i,0), closestQIdx)++;
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

    std::vector<cv::Mat> gQ (m_qValParam.size());
    if (m_bGlobalQuantization)
    {
        // Centers determined using K-Means++ in {validation U training} data.
        
        // This contamination is used only for the selection of number of quantization
        // centers. Normalization, dim reduction, and classifiers' parameters selections
        // are not affected. Reasons are: hypothesis that with this contamination
        // the selection of #centers will not be affected (bc all parameters are equally
        // benfited), and 10x speed increase.
        
        for (int i = 0; i < m_qValParam.size(); i++)
        {
            if (!m_bCentersStratification)
            {
                // Centers are selected among all the points without taking into
                // account categories
                cv::Mat X, c;
                cloudjectsToPointsSample(m_InputCloudjects, X, c);
                bow(X, m_qValParam[i], gQ[i]);
            }
            else
            {
                // Centers are selected in such a way we ensure a minimum number
                // of centers for each class (the parameters divided by the number
                // of categories)
                std::map<std::string,cv::Mat> XMap;
                cloudjectsToPointsSample(m_InputCloudjects, XMap);
                bow(XMap, floor(m_qValParam[i] / m_Categories.size()) + 1, gQ[i]);
            }
        }
    }
    
    std::vector<cv::Mat> S (m_qValParam.size());
    for (int t = 0; t < m_NumFolds; t++)
    {
        // Partitionate the data
        
        boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjectsTr (new std::list<Cloudject::Ptr>);
        boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjectsTe (new std::list<Cloudject::Ptr>);
        getTrainingCloudjects(m_InputCloudjects, t, *cloudjectsTr);
        getTestCloudjects(m_InputCloudjects, t, *cloudjectsTe);

        std::map<std::string,cv::Mat> XTrMap;
        if (!m_bGlobalQuantization)
            cloudjectsToPointsSample(cloudjectsTr, XTrMap);
        
        // Create the training sample and quantize
        
        cv::Mat XTr, XTe, cTr, cTe;
        cloudjectsToPointsSample(cloudjectsTr, XTr, cTr);
        cloudjectsToPointsSample(cloudjectsTe, XTe, cTe);
        
        // Get the partitioned data labels
        
        cv::Mat YTr, YTe;
        getLabels(cloudjectsTr, m_Categories, YTr);
        getLabels(cloudjectsTe, m_Categories, YTe);
        
        for (int i = 0; i < m_qValParam.size(); i++)
        {
            cv::Mat Q;
            if (!m_bGlobalQuantization)
            {
                if (!m_bCentersStratification)
                    bow(XTr, m_qValParam[i], Q);
                else
                    bow(XTrMap, m_qValParam[i], Q);
            }
            
            cv::Mat WTr, WTe;
            bow(XTr, cTr, (m_bGlobalQuantization ? gQ[i] : Q), WTr);
            bow(XTe, cTe, (m_bGlobalQuantization ? gQ[i] : Q), WTe);
            
            cv::Mat WnTr, WnTe;
            std::vector<cv::Mat> parameters;
            normalize(WTr, WnTr, parameters);
            normalize(WTe, parameters, WnTe);
            
            cv::Mat WpTr, WpTe;
            if (m_bCentersStratification && m_bPerStrateReduction)
            {
                std::vector<cv::PCA> PCAs;
                reduce(WnTr, m_qValParam[i] / m_Categories.size(), WpTr, PCAs);
                reduce(WnTe, PCAs, WpTe);
            }
            else
            {
                cv::PCA PCA;
                CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(WnTr, WpTr, PCA);
                CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(WnTe, PCA, WpTe);
            }
            
            cv::Mat P = CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::validate(WpTr, WpTe, YTr, YTe);
            
//            std::cout << P << std::endl;
            if (t == 0) S[i] = (P/m_NumFolds);
            else S[i] += (P/m_NumFolds);
        }
    }
    
    
    m_qValPerfs.resize(m_qValParam.size());
    for (int i = 0; i < m_qValParam.size(); i++)
    {
        std::cout << S[i] << std::endl;

        m_qValPerfs[i] = .0f;
        
        std::vector<std::vector<float> > classifierParamsTmp (Y.cols);
        for (int j = 0; j < Y.cols; j++)
        {
            double minVal, maxVal;
            cv::Point minIdx, maxIdx;
            cv::minMaxLoc(S[i].col(j), &minVal, &maxVal, &minIdx, &maxIdx);

            m_qValPerfs[i] += (maxVal / Y.cols);
        }
        
        if (m_qValPerfs[i] > m_BestValPerformance)
        {
            m_BestValPerformance = m_qValPerfs[i];
            m_q = m_qValParam[i]; // can be used in the training without explicit specification
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
//    cv::Mat Y;
//    getLabels(m_InputCloudjects, m_Categories, Y);
//    
//    // Bag-of-words aggregation since PFHRGBSignature is a local descriptor
//    cv::Mat W, Q;
//    bowSampleFromCloudjects(m_InputCloudjects, m_q, W, m_C);
//    
//    // Eventually train
//    CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::train(W,Y);
}

std::vector<float> CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::predict(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, std::list<std::vector<int> >& predictions, std::list<std::vector<float> >& distsToMargin)
{
//    cv::Mat Y;
//    getLabels(cloudjects, m_Categories, Y);
//    
//    // Bag-of-words aggregation since PFHRGBSignature is a local descriptor
//    cloudjectsToPointsSample(<#boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects#>, <#std::map<std::string, cv::Mat> &X#>)
//    cv::Mat W, Q;
//    bowSampleFromCloudjects(cloudjects, m_C, W);
//    
//    assert (cloudjects->size() == W.rows); // a word per cloudject
//    
//    // Eventually predict
//    cv::Mat P,F;
//    CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::predict(W,P,F);
//    
//    cvx::convert(P, predictions);
//    cvx::convert(F, distsToMargin);
//    
//    // Prepare the output
//    std::vector<float> accs = CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::evaluate(Y,P);
//    
//    return accs;
}

std::vector<int> CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::predict(Cloudject::Ptr cloudject, std::vector<float>& distsToMargin)
{
//    boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects (new std::list<Cloudject::Ptr>);
//    cloudjects->push_back(cloudject);
//    
//    cv::Mat W, Q;
//    bowSampleFromCloudjects(cloudjects, m_C, W);
//    
//    assert (W.rows == 1);
//    
//    // Eventually predict
//    cv::Mat P,F;
//    CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::predict(W,P,F);
//    
//    cv::Mat f = F.row(0);
//    distsToMargin = std::vector<float>(f.begin<float>(), f.end<float>());
//    
//    cv::Mat p = P.row(0);
//    std::vector<int> predictions ( p.begin<int>(), p.end<int>());
//    return predictions;
}
            
void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::reduce(cv::Mat X, int q, cv::Mat& Xr, std::vector<cv::PCA>& PCAs)
{
    int m = X.cols / q;
    PCAs.resize(q);
    
    std::vector<cv::Mat> XrVec (q);
    for (int i = 0; i < q; i++)
    {
        cv::Mat roi (X, cv::Rect(i*m, 0, m, X.rows));
        CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(roi, XrVec[i], PCAs[i]);
    }
    cv::hconcat(XrVec, Xr);
}
            
void CloudjectClassificationPipeline<pcl::PFHRGBSignature250>::reduce(cv::Mat X, std::vector<cv::PCA> PCAs, cv::Mat& Xr)
{
    int m = X.cols / PCAs.size();

    std::vector<cv::Mat> XrVec (PCAs.size());
    for (int i = 0; i < PCAs.size(); i++)
    {
        cv::Mat roi (X, cv::Rect(i*m, 0, m, X.rows));
        CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(roi, PCAs[i], XrVec[i]);
    }
    cv::hconcat(XrVec, Xr);
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
template void CloudjectClassificationPipelineBase<pcl::PFHRGBSignature250>::predict(cv::Mat X, cv::Mat& P, cv::Mat& F);
