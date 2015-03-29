//
//  ClassificationPipeline.cpp
//  remedi3
//
//  Created by Albert Clapés on 12/03/15.
//
//

#include "CloudjectSVMClassificationPipeline.h"
#include "constants.h"

#include "statistics.h"
#include "cvxtended.h"

#include <boost/assign/std/vector.hpp> // for 'operator+=()'
#include <boost/timer.hpp>

using namespace boost::assign; // bring 'operator+=()' into scope

// ----------------------------------------------------------------------------
// CloudjectSVMClassificationPipelineBase
// ----------------------------------------------------------------------------

template<typename T>
CloudjectSVMClassificationPipelineBase<T>::CloudjectSVMClassificationPipelineBase()
: m_NormType(-1), m_ClassifierType(""), m_DimRedVariance(0), m_DimRedNumFeatures(0), m_BestValPerformance(0)
{
    
}

template<typename T>
CloudjectSVMClassificationPipelineBase<T>::CloudjectSVMClassificationPipelineBase(const CloudjectSVMClassificationPipelineBase<T>& rhs)
{
    *this = rhs;
}

template<typename T>
CloudjectSVMClassificationPipelineBase<T>::~CloudjectSVMClassificationPipelineBase()
{
//    if (m_SvmModelsPtrs.size() > 0)
//    {
//        for (int i = 0; i < m_SvmModelsPtrs.size(); i++)
//            if (m_SvmModelsPtrs[i] != NULL)
//                delete m_SvmModelsPtrs[i];
//    }
}

template<typename T>
CloudjectSVMClassificationPipelineBase<T>& CloudjectSVMClassificationPipelineBase<T>::operator=(const CloudjectSVMClassificationPipelineBase<T>& rhs)
{
    if (this != &rhs)
    {
        m_InputMockCloudjects = rhs.m_InputMockCloudjects;
        m_Categories = rhs.m_Categories;
        
        m_NormType = rhs.m_NormType;
        m_NormParams = rhs.m_NormParams;
        
        m_NumFolds = rhs.m_NumFolds;
        m_Partitions = rhs.m_Partitions;
        
        m_DimRedVariance = rhs.m_DimRedVariance;
        m_DimRedNumFeatures = rhs.m_DimRedNumFeatures;
        
        m_ClassifiersValParamCombs = rhs.m_ClassifiersValParamCombs;
        m_ClassifiersValPerfs = rhs.m_ClassifiersValPerfs;
        
        m_ClassifierType = rhs.m_ClassifierType;
        m_ClassifierValParams = rhs.m_ClassifierValParams;
        m_ClassifierParams = rhs.m_ClassifierParams;
        m_ClassifierModels = rhs.m_ClassifierModels;
        
        m_BestValPerformance = rhs.m_BestValPerformance;
    }
    
    return *this;
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::setInputCloudjects(boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjects)
{
    m_InputMockCloudjects = cloudjects;
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::setCategories(std::vector<const char*> categories)
{
    m_Categories = categories;
}

//template<typename T>
//boost::shared_ptr<std::list<Cloudject::Ptr> > CloudjectSVMClassificationPipelineBase<T>::getInputCloudjects()
//{
//    return m_InputMockCloudjects;
//}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::setValidation(int numFolds)
{
    m_NumFolds = numFolds;
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::setNormalization(int normType)
{
    m_NormType = normType;
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::setDimRedVariance(double var)
{
    m_DimRedVariance = var;
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::setDimRedNumFeatures(int n)
{
    m_DimRedNumFeatures = n;
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::normalize(cv::Mat X, cv::Mat& Xn, std::vector<cv::Mat>& parameters)
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
void CloudjectSVMClassificationPipelineBase<T>::normalize(cv::Mat X, std::vector<cv::Mat> parameters, cv::Mat& Xn)
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
void CloudjectSVMClassificationPipelineBase<T>::setClassifierType(std::string classifierType)
{
    m_ClassifierType = classifierType;
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::setClassifierValidationParameters(std::vector<std::vector<float> > classifierParams)
{
    m_ClassifierValParams = classifierParams;
    expandParameters(m_ClassifierValParams, m_ClassifiersValParamCombs);
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::validate(cv::Mat XTr, cv::Mat XTe, cv::Mat YTr, cv::Mat YTe)
{
    m_ClassifiersValPerfs.create(m_ClassifiersValParamCombs.size(), YTr.cols, cv::DataType<float>::type);
    
    for (int i = 0; i < m_ClassifiersValParamCombs.size(); i++)
    {
        for (int j = 0; j < YTr.cols; j++)
        {
            CvSVM::CvSVM svm;
            svm.train(XTr, YTr.col(j), cv::Mat(), cv::Mat(),
                      CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 0, m_ClassifiersValParamCombs[i][0], 0, m_ClassifiersValParamCombs[i][1], 0, 0, 0, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 500, 1e-4)));
            
            cv::Mat fTe; // predictions
            svm.predict(XTe, fTe);
            
            float acc = accuracy(YTe.col(j), fTe);
            m_ClassifiersValPerfs.at<float>(i,j) = acc;
        }
    }
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::setClassifierParameters(std::vector<std::vector<float> > classifierParams)
{
    // Rows must be the number of classifiers, and columns the number of parameters
    m_ClassifierParams = classifierParams;
}

template<typename T>
std::vector<std::vector<float> > CloudjectSVMClassificationPipelineBase<T>::getValidationParametersCombinations()
{
    return m_ClassifiersValParamCombs;
}

template<typename T>
cv::Mat CloudjectSVMClassificationPipelineBase<T>::getValidationPerformances()
{
    return m_ClassifiersValPerfs;
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::train(cv::Mat X, cv::Mat Y)
{
    // Final model training
    m_ClassifierModels.resize(Y.cols, NULL);
    for (int i = 0; i < Y.cols; i++)
    {
        if (m_ClassifierType == "svmrbf")
        {
            CvSVM* svm = new CvSVM;
            svm->train(X, Y.col(i), cv::Mat(), cv::Mat(),
                       CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 0, m_ClassifierParams[i][0], 0, m_ClassifierParams[i][1], 0, 0, 0, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-3)));
            m_ClassifierModels[i] = (void*) svm;
        }
    }
//    m_InputMockCloudjects = boost::shared_ptr<std::list<Cloudject::Ptr> >(); // no longer needed
}

// Protected methods

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::predict(cv::Mat X, cv::Mat& P, cv::Mat& F)
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
std::vector<float> CloudjectSVMClassificationPipelineBase<T>::evaluate(cv::Mat Y, cv::Mat F)
{
    std::vector<float> accs (Y.cols);
    
    for (int j = 0; j < Y.cols; j++)
        accs[j] = accuracy(Y.col(j), F.col(j));
    
    return accs;
}

template<typename T> template<typename CloudjectTypePtr>
void CloudjectSVMClassificationPipelineBase<T>::getLabels(boost::shared_ptr<std::list<CloudjectTypePtr> > cloudjects, std::vector<const char*> categories, cv::Mat& Y)
{
    // Format the labels of the sample X
    Y.release();
    Y.create(cloudjects->size(), categories.size(), cv::DataType<int>::type);
    
    int i = 0;
    typename std::list<CloudjectTypePtr>::iterator it;
    for (it = cloudjects->begin(); it != cloudjects->end(); ++it, ++i)
        for (int j = 0; j < categories.size(); j++)
            Y.at<int>(i,j) = ((*it)->isRegionLabel(categories[j]) ? 1 : -1);
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::createValidationPartitions(cv::Mat Y, int numFolds, cv::Mat& partitions)
{
    // Create the partitions taking into account the labels (strafified CV)
    
    cv::Mat multipliers = cvx::powspace(0, Y.cols, 10);
    cv::Mat groups (Y.rows, 1, cv::DataType<int>::type, cv::Scalar(0));
    
    for (int i = 0; i < Y.rows; i++) for (int j = 0; j < Y.cols; j++)
        groups.at<int>(i,0) += Y.at<int>(i,j) * multipliers.at<int>(j,0);
    
    cvpartition(groups, numFolds, 42, partitions); // stratification
}

template<typename T>
void CloudjectSVMClassificationPipelineBase<T>::reduce(cv::Mat X, cv::Mat& Xr, cv::PCA& pca)
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
void CloudjectSVMClassificationPipelineBase<T>::reduce(cv::Mat X, cv::PCA pca, cv::Mat& Xr)
{
    if (m_DimRedNumFeatures == 0 && m_DimRedVariance == 0)
        Xr = X;
    else
        Xr = pca.project(X);
}

template<typename T> template<typename CloudjectTypePtr>
void CloudjectSVMClassificationPipelineBase<T>::getTrainingCloudjects(boost::shared_ptr<typename std::list<CloudjectTypePtr> > cloudjects, int t, typename std::list<CloudjectTypePtr>& cloudjectsTr)
{
    cloudjectsTr.clear();

    int i = 0;
    typename std::list<CloudjectTypePtr>::iterator it;
    for (it = cloudjects->begin(); it != cloudjects->end(); ++it, ++i)
        if (m_Partitions.at<int>(i,0) != t)
            cloudjectsTr.push_back(*it);
}

template<typename T>  template<typename CloudjectTypePtr>
void CloudjectSVMClassificationPipelineBase<T>::getTestCloudjects(boost::shared_ptr<typename std::list<CloudjectTypePtr> > cloudjects, int t, typename std::list<CloudjectTypePtr>& cloudjectsTe)
{
    cloudjectsTe.clear();
    
    int i = 0;
    typename std::list<CloudjectTypePtr>::iterator it;
    for (it = cloudjects->begin(); it != cloudjects->end(); ++it, ++i)
        if (m_Partitions.at<int>(i,0) == t)
            cloudjectsTe.push_back(*it);
}

// ----------------------------------------------------------------------------
// CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>
// ----------------------------------------------------------------------------

CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::CloudjectSVMClassificationPipeline()
: CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>(), m_bCentersStratification(false), m_bPerStrateReduction(false), m_bGlobalQuantization(false)
{
}

CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::CloudjectSVMClassificationPipeline(const CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>& rhs) : CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>(rhs), m_bCentersStratification(false), m_bPerStrateReduction(false), m_bGlobalQuantization(false)
{
    *this = rhs;
}

CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>& CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::operator=(const CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>& rhs)
{
    if (this != &rhs)
    {
        m_q = rhs.m_q;
        m_Centers = rhs.m_Centers;
        
        m_PCAs = rhs.m_PCAs;
        
        m_bGlobalQuantization = rhs.m_bGlobalQuantization;
        
        m_bCentersStratification = rhs. m_bCentersStratification;
        m_bPerStrateReduction = rhs.m_bPerStrateReduction;
    }
    
    return *this;
}

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::setGlobalQuantization(bool bGlobalQuantization)
{
    m_bGlobalQuantization = bGlobalQuantization;
}

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::setCentersStratification(bool bCentersStratification)
{
    m_bCentersStratification = bCentersStratification;
}

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::setPerCategoryReduction(bool bPerStrateReduction)
{
    m_bPerStrateReduction = bPerStrateReduction;
}

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::cloudjectsToPointsSample(boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjects, cv::Mat& X, cv::Mat& c)
{
    c.create(cloudjects->size(), 1, cv::DataType<int>::type);
    
    int numPointsInstances = 0;
    
    int i = 0;
    std::list<MockCloudject::Ptr>::iterator cjIt;
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

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::cloudjectsToPointsSample(boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects, cv::Mat& X, cv::Mat& c)
{
    c.create(cloudjects->size(), 1, cv::DataType<int>::type);
    
    int numPointsInstances = 0;
    
    int i = 0;
    std::list<Cloudject::Ptr>::iterator cjIt;
    for (cjIt = cloudjects->begin(); cjIt != cloudjects->end(); ++cjIt, ++i)
    {       
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
    }
}

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::cloudjectsToPointsSample(boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjects, std::map<std::string,cv::Mat>& X)
{
    X.clear();
    
    std::map<std::string,int> numPointInstancesMap; // total no. descriptor points (no. rows of X and c actually)
    
    std::list<MockCloudject::Ptr>::iterator cjIt;
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

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::bow(cv::Mat X, int q, cv::Mat& Q)
{
    // Find quantization vectors
    cv::Mat u;
    cv::kmeans(X, q, u, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, Q);
}

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::bow(std::map<std::string,cv::Mat> X, int q, cv::Mat& Q)
{
    // Find quantization vectors
    std::vector<cv::Mat> QVec (m_Categories.size());
    
    boost::timer t;
    for (int k = 0; k < m_Categories.size(); k++)
    {
        cv::Mat u;
        cv::kmeans(X[m_Categories[k]], q, u, cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, QVec[k]);
    }
    std::cout << "BOW centers' finding took " << t.elapsed() << " secs." << std::endl;
    
    cv::vconcat(QVec,Q);
}

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::bow(cv::Mat X, cv::Mat c, cv::Mat Q, cv::Mat& W)
{
    // Indexing vector, associating X rows to global instance
    std::vector<cv::Mat> countings (c.rows);
    for (int i = 0; i < c.rows; i++)
        countings[i] = cv::Mat(c.at<int>(i,0), 1, c.type(), cv::Scalar(i));
    
    cv::Mat I;
    cv::vconcat(countings, I);
    
    // Word construction
    W.create(c.rows, Q.rows, cv::DataType<float>::type);
    W.setTo(0);
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

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::setQuantizationValidationParameters(std::vector<int> qs)
{
    m_qValParam = qs;
}

float CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::validate()
{
    // Input data already set
    assert (m_InputMockCloudjects->size() > 0);
    m_qValPerfs.resize(m_qValParam.size());

    cv::Mat Y;
    getLabels(m_InputMockCloudjects, m_Categories, Y);
    createValidationPartitions(Y, m_NumFolds, m_Partitions);

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
                cloudjectsToPointsSample(m_InputMockCloudjects, X, c);
                bow(X, m_qValParam[i], gQ[i]);
            }
            else
            {
                // Centers are selected in such a way we ensure a minimum number
                // of centers for each class (the parameters divided by the number
                // of categories)
                std::map<std::string,cv::Mat> XMap;
                cloudjectsToPointsSample(m_InputMockCloudjects, XMap);
                int sq = floor(m_qValParam[i] / m_Categories.size());
                bow(XMap, (sq == 0) ? 1 : sq, gQ[i]);
            }
        }
    }
    
    std::vector<cv::Mat> S (m_qValParam.size());
    for (int t = 0; t < m_NumFolds; t++)
    {
        // Partitionate the data
        
        boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjectsTr (new std::list<MockCloudject::Ptr>);
        boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjectsTe (new std::list<MockCloudject::Ptr>);
        getTrainingCloudjects(m_InputMockCloudjects, t, *cloudjectsTr);
        getTestCloudjects(m_InputMockCloudjects, t, *cloudjectsTe);

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
                reduce(WnTr, m_qValParam[i], WpTr, PCAs);
                reduce(WnTe, PCAs, WpTe);
            }
            else
            {
                cv::PCA PCA;
                CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(WnTr, WpTr, PCA);
                CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(WnTe, PCA, WpTe);
            }
            
            CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::validate(WpTr, WpTe, YTr, YTe);
            cv::Mat P = CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::getValidationPerformances();
            
//            std::cout << P << std::endl;
            if (t == 0) S[i] = (P/m_NumFolds);
            else S[i] += (P/m_NumFolds);
        }
    }
    
    std::vector<std::vector<float> > classifiersValParamCombs = CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::getValidationParametersCombinations();
    
    m_qValPerfs.resize(m_qValParam.size());
    for (int i = 0; i < m_qValParam.size(); i++)
    {
//        std::cout << S[i] << std::endl;

        std::vector<std::vector<float> > bestParamsTmp (Y.cols);
        m_qValPerfs[i] = .0f;
        for (int j = 0; j < Y.cols; j++)
        {
            double minVal, maxVal;
            cv::Point minIdx, maxIdx;
            cv::minMaxLoc(S[i].col(j), &minVal, &maxVal, &minIdx, &maxIdx);

            bestParamsTmp[j] = classifiersValParamCombs[maxIdx.y];
            m_qValPerfs[i] += (maxVal / Y.cols);
        }
        
        if (m_qValPerfs[i] > m_BestValPerformance)
        {
            setClassifierParameters(bestParamsTmp);
            
            m_q = m_qValParam[i]; // can be used in the training without explicit specification
            m_BestValPerformance = m_qValPerfs[i];
        }
    }
    
    return m_BestValPerformance;
}

std::vector<float> CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::getQuantizationValidationPerformances()
{
    return m_qValPerfs;
}

int CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::getQuantizationParameter()
{
    return m_q;
}

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::setQuantizationParameter(int q)
{
    m_q = q;
}

void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::train()
{
    assert (m_InputMockCloudjects->size() > 0);
    
    cv::Mat Y;
    getLabels(m_InputMockCloudjects, m_Categories, Y);
    
    cv::Mat X, c;
    cloudjectsToPointsSample(m_InputMockCloudjects, X, c);
    
    if (!m_bCentersStratification)
    {
        bow(X, m_q, m_Centers);
    }
    else
    {
        std::map<std::string,cv::Mat> XMap;
        cloudjectsToPointsSample(m_InputMockCloudjects, XMap);
        int sq = m_q / m_Categories.size();
        bow(XMap, (sq == 0) ? 1 : sq, m_Centers);
    }
    
    cv::Mat W;
    bow(X, c, m_Centers, W);
    
    cv::Mat Wn;
    normalize(W, Wn, m_NormParams);
    
    cv::Mat Wp;
    if (m_bCentersStratification && m_bPerStrateReduction)
    {
        reduce(Wn, m_q, Wp, m_PCAs);
    }
    else
    {
        m_PCAs.resize(1);
        CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(Wn, Wp, m_PCAs[0]);
    }
    
    CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::train(Wp,Y);
}

std::vector<float> CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::predict(boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjects, std::list<std::vector<int> >& predictions, std::list<std::vector<float> >& distsToMargin)
{
    cv::Mat Y;
    getLabels(cloudjects, m_Categories, Y);

    // Bag-of-words aggregation since PFHRGBSignature is a local descriptor
    cv::Mat X, c;
    cloudjectsToPointsSample(cloudjects, X, c);

    cv::Mat W;
    bow(X, c, m_Centers, W);
    
    cv::Mat Wn;
    normalize(W, m_NormParams, Wn);
    
    cv::Mat Wp;
    if (m_bCentersStratification && m_bPerStrateReduction)
        reduce(Wn, m_PCAs, Wp);
    else
        CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(Wn, m_PCAs[0], Wp);
    
    cv::Mat P,F;
    CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::predict(Wp,P,F);
    
    cvx::convert(P, predictions);
    cvx::convert(F, distsToMargin);
    
    // Prepare the output
    std::vector<float> accs = CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::evaluate(Y,P);
    
    return accs;
}

std::vector<int> CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::predict(Cloudject::Ptr cloudject, std::vector<float>& distsToMargin)
{
    boost::shared_ptr<std::list<Cloudject::Ptr> > cloudjects (new std::list<Cloudject::Ptr>);
    cloudjects->push_back(cloudject);
    
    cv::Mat x, c;
    cloudjectsToPointsSample(cloudjects, x, c);
    
    cv::Mat w;
    bow(x, c, m_Centers, w);
    
    cv::Mat wn;
    normalize(w, m_NormParams, wn);
    
    cv::Mat wp;
    if (m_bCentersStratification && m_bPerStrateReduction)
        reduce(wn, m_PCAs, wp);
    else
        CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(wn, m_PCAs[0], wp);
    
    cv::Mat p,f;
    CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::predict(wp,p,f);
    
    cv::Mat fAux = f.row(0);
    distsToMargin = std::vector<float>(fAux.begin<float>(), fAux.end<float>());
    
    cv::Mat pAux = p.row(0);
    std::vector<int> predictions ( pAux.begin<int>(), pAux.end<int>());
    return predictions;
}
            
void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::reduce(cv::Mat X, int q, cv::Mat& Xr, std::vector<cv::PCA>& PCAs)
{
    int m = q / m_Categories.size();
    PCAs.resize(m_Categories.size());
    
    std::vector<cv::Mat> XrVec (m_Categories.size());
    for (int i = 0; i < m_Categories.size(); i++)
    {
        cv::Mat roi (X, cv::Rect(i*m, 0, m, X.rows));
        CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(roi, XrVec[i], PCAs[i]);
    }
    cv::hconcat(XrVec, Xr);
}
            
void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::reduce(cv::Mat X, const std::vector<cv::PCA>& PCAs, cv::Mat& Xr)
{
    int m = X.cols / PCAs.size();

    std::vector<cv::Mat> XrVec (PCAs.size());
    for (int i = 0; i < PCAs.size(); i++)
    {
        cv::Mat roi (X, cv::Rect(i*m, 0, m, X.rows));
        CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::reduce(roi, PCAs[i], XrVec[i]);
    }
    cv::hconcat(XrVec, Xr);
}

bool CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::load(std::string filename, std::string extension)
{
    std::string filenameWithSuffix = filename + extension; // extension is expected to already include the ext dot (.xxx)
    cv::FileStorage fs;
    if (extension.compare(".xml") == 0)
        fs.open(filenameWithSuffix, cv::FileStorage::READ | cv::FileStorage::FORMAT_XML);
    else // yml
        fs.open(filenameWithSuffix, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    
    if (!fs.isOpened())
        return false;
    
    // Save categories (problem with std::vector<const char*>, so convert to strings)
    std::vector<std::string> categories;
    fs["categories"] >> categories;
    m_Categories.resize(categories.size());
    for (int i = 0; i < m_Categories.size(); i++)
        m_Categories[i] = categories[i].c_str();
    
    // Save the normalization-related variables
    fs["normType"] >> m_NormType;
    if (m_NormType == CCPIPELINE_NORM_MINMAX || m_NormType == CCPIPELINE_NORM_STD)
    {
        m_NormParams.resize(2);
        fs["normParams-0"] >> m_NormParams[0];
        fs["normParams-1"] >> m_NormParams[1];
    }
    
    // Save the BOW-related variables
    fs["q"] >> m_q;
    fs["centers"] >> m_Centers;
    
    // Save the dim reduction-related variables
    fs["dimRedVariance"] >> m_DimRedVariance;
    fs["dimRedNumFeatures"] >> m_DimRedNumFeatures;
    fs["globalQuantization"] >> m_bGlobalQuantization;
    fs["centersStratification"] >> m_bCentersStratification;
    fs["perStrateReduction"] >> m_bPerStrateReduction;
    
    int n;
    fs["pcas-numberOf"] >> n;
    m_PCAs.resize(n);
    
    for (int i = 0; i < m_PCAs.size(); i++)
    {
        fs["pcas_mean-" + std::to_string(i)] >> m_PCAs[i].mean;
        fs["pcas_eigenvectors-" + std::to_string(i)] >> m_PCAs[i].eigenvectors;
        fs["pcas_eigenvalues-" + std::to_string(i)] >> m_PCAs[i].eigenvalues;
    }
    
    // Save the models' metadata
    fs["classifierModels-numberOf"] >> n;
    m_ClassifierModels.resize(n);
    fs["classifierType"] >> m_ClassifierType;
    
    // Save the models
    for (int i = 0; i < m_ClassifierModels.size(); i++)
    {
        std::string modelname = "classifierModels-" + std::to_string(i);
        if (m_ClassifierType.compare("svmrbf") == 0)
        {
            CvSVM* pModel = new CvSVM;
            pModel->load(filenameWithSuffix.c_str(), modelname.c_str());
            m_ClassifierModels[i] = (void*) pModel;
        }
    }
    
    fs.release();
    
    return true; // all OK
}


void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::save(std::string filename, std::string extension)
{
    std::string filenameWithSuffix = filename + extension; // extension is expected to already include the ext dot (.xxx)
    cv::FileStorage fs;
    
    if (extension.compare(".xml") == 0)
        fs.open(filenameWithSuffix, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_XML);
    else // yml
        fs.open(filenameWithSuffix, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    
    // Save categories (problem with std::vector<const char*>, so convert to strings)
    std::vector<std::string> categories (m_Categories.size());
    for (int i = 0; i < m_Categories.size(); i++)
        categories[i] = m_Categories[i];
    fs << "categories" << categories;
    
    // Save the normalization-related variables
    fs << "normType" << m_NormType;
    for (int i = 0; i < m_NormParams.size(); i++)
        fs << "normParams-" + std::to_string(i) << m_NormParams[i];
    
    // Save the BOW-related variables
    fs << "q" << m_q;
    fs << "centers" << m_Centers;

    // Save the dim reduction-related variables
    fs << "dimRedVariance" << m_DimRedVariance;
    fs << "dimRedNumFeatures" << m_DimRedNumFeatures;
    
    fs << "globalQuantization" << m_bGlobalQuantization;
    fs << "centersStratification" << m_bCentersStratification;
    fs << "perStrateReduction" << m_bPerStrateReduction;
    
    fs << "pcas-numberOf" << ((int) m_PCAs.size());
    for (int i = 0; i < m_PCAs.size(); i++)
    {
        fs << "pcas_mean-" + std::to_string(i) << m_PCAs[i].mean;
        fs << "pcas_eigenvectors-" + std::to_string(i) << m_PCAs[i].eigenvectors;
        fs << "pcas_eigenvalues-" + std::to_string(i) << m_PCAs[i].eigenvalues;
    }

    // Save the models' metadata
    fs << "classifierModels-numberOf" << ((int) m_ClassifierModels.size());
    fs << "classifierType" << m_ClassifierType;

    // Save the models
    for (int i = 0; i < m_ClassifierModels.size(); i++)
    {
        std::string modelname = "classifierModels-" + std::to_string(i);
        if (m_ClassifierType.compare("svmrbf") == 0)
            ((CvSVM*) m_ClassifierModels[i])->write(fs.operator*(), modelname.c_str());
    }
    
    fs.release();
}

//void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::saveModels(std::string filename, std::string extension)
//{
//    // Save the models
//    for (int i = 0; i < m_ClassifierModels.size(); i++)
//    {
//        if (m_ClassifierType.compare("svmrbf") == 0)
//        {
//            ((CvSVM*) m_ClassifierModels[i])->save(filename + "-M" + std::to_string(i) + extension);
//        }
//    }
//}

//void CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>::loadModels(std::string filename, std::string extension)
//{
//    // Save the models
//    fs << "classifiers_numberof" << m_ClassifierModels.size();
//    fs << "classifiers_type" << m_ClassifierType;
//    
//    for (int i = 0; i < m_ClassifierModels.size(); i++)
//    {
//        if (m_ClassifierType.compare("svmrbf") == 0)
//            ((CvSVM*) m_ClassifierModels[i])->save(filename + "-M" + std::to_string(i) + extension);
//    }
//}

template class CloudjectSVMClassificationPipeline<pcl::PFHRGBSignature250>;

template void CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::setInputCloudjects(boost::shared_ptr<std::list<MockCloudject::Ptr> > cloudjects);
template void CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::setCategories(std::vector<const char*> categories);
template void CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::setNormalization(int normType);
template void CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::setDimRedVariance(double var);
template void CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::setDimRedNumFeatures(int n);
template void CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::setClassifierType(std::string classifierType);
template void CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::setValidation(int numFolds);
template void CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::setClassifierValidationParameters(std::vector<std::vector<float> > classifierParams);
template void CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::train(cv::Mat X, cv::Mat Y);
template void CloudjectSVMClassificationPipelineBase<pcl::PFHRGBSignature250>::predict(cv::Mat X, cv::Mat& P, cv::Mat& F);
