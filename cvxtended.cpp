//
//  cvxtended.cpp
//  remedi2
//
//  Created by Albert Clapés on 02/04/14.
//
//

#include "cvxtended.h"
#include <iterator>

//#include <matio.h>

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());
    
    std::vector<T> result;
    result.reserve(a.size());
    
    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}

template std::vector<int> operator+(const std::vector<int>& a, const std::vector<int>& b);
template std::vector<float> operator+(const std::vector<float>& a, const std::vector<float>& b);
template std::vector<unsigned short> operator+(const std::vector<unsigned short>& a, const std::vector<unsigned short>& b);
template std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b);

// --------------------------
//      OpenCVeXtended
// --------------------------

void cvx::setMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
{
    if (logical)
        setMatLogically(src, dst, indices);
    else
        setMatPositionally(src, dst, indices);
}

void cvx::setMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
{
    bool rowwise = logicals.rows > 1;
    
    if (dst.empty())
    {
        if (rowwise)
            dst.create(logicals.rows, src.cols, src.type());
        else
            dst.create(src.rows, logicals.rows, src.type());
        
        dst.setTo(0);
    }
    
    int c = 0;
    int n = rowwise ? dst.rows : dst.cols;
    for (int i = 0; i < n; i++)
    {
        unsigned char indexed = rowwise ?
        logicals.at<unsigned char>(i,0) : logicals.at<unsigned char>(0,i);
        if (indexed)
            rowwise ? src.row(c++).copyTo(dst.row(i)) : src.col(c++).copyTo(dst.col(i));
    }
}

void cvx::setMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indexes)
{
    bool rowwise = indexes.rows > 1;
    
    int n = rowwise ? indexes.rows : indexes.cols;
    for (int i = 0; i < n; i++)
    {
        int idx = rowwise ? indexes.at<int>(i,0) : indexes.at<int>(i,0);
        rowwise ? src.row(i).copyTo(dst.row(idx)) : src.col(i).copyTo(dst.col(idx));
    }
}

void cvx::indexMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
{
    if (logical)
        indexMatLogically(src, dst, indices);
    else
        indexMatPositionally(src, dst, indices);
}

void cvx::indexMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
{
    bool rowwise = logicals.rows > 1;
    
    if (dst.empty())
    {
        if (rowwise)
            dst.create(0, src.cols, src.type());
        else
            dst.create(src.rows, 0, src.type());
    }
    
    int c = 0;
    for (int i = 0; i < (rowwise ? logicals.rows : logicals.cols); i++)
    {
        unsigned char indexed = rowwise ?
        logicals.at<unsigned char>(i,0) : logicals.at<unsigned char>(0,i);
        if (indexed)
            rowwise ? dst.push_back(src.row(i)) : dst.push_back(src.col(i));
    }
}

void cvx::indexMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indexes)
{
    if (dst.empty())
        dst.create(src.rows, src.cols, src.type());
    
    bool rowwise = indexes.rows > 1;
    
    for (int i = 0; i < (rowwise ? indexes.rows : indexes.cols); i++)
    {
        int idx = rowwise ? indexes.at<int>(i,0) : indexes.at<int>(i,0);
        rowwise ? src.row(idx).copyTo(dst.row(i)) : src.col(idx).copyTo(dst.col(i));
    }
}

cv::Mat cvx::indexMat(cv::Mat src, cv::Mat indices, bool logical)
{
    cv::Mat mat;
    
    cvx::indexMat(src, mat, indices, logical);

    return mat;
}

//void cvx::indexMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
//{
//    if (logical)
//        indexMatLogically(src, dst, indices);
//    else
//        indexMatPositionally(src, dst, indices);
//}
//
//void cvx::indexMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
//{
//    if (logicals.rows > 1) // row-wise
//        dst.create(0, src.cols, src.type());
//    else // col-wise
//        dst.create(src.rows, 0, src.type());
//
//    for (int i = 0; i < (logicals.rows > 1 ? logicals.rows : logicals.cols); i++)
//    {
//        if (logicals.rows > 1)
//        {
//            if (logicals.at<unsigned char>(i,0))
//            {
//                dst.push_back(src.row(i));
//            }
//        }
//        else
//        {
//            if (logicals.at<unsigned char>(0,i))
//                dst.push_back(src.col(i));
//        }
//    }
//}
//
//void cvx::indexMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices)
//{
//    dst.release();
//
//    if (indices.rows > 1) // row-wise
//        dst.create(indices.rows, src.cols, src.type());
//    else // col-wise
//        dst.create(src.rows, indices.cols, src.type());
//
//    for (int i = 0; i < (indices.rows > 1 ? indices.rows : indices.cols); i++)
//    {
//        int idx = indices.rows > 1 ? indices.at<int>(i,0) : indices.at<int>(0,i);
//
//        if (indices.rows > 1)
//            src.row(idx).copyTo(dst.row(i));
//        else
//            src.col(idx).copyTo(dst.col(i));
//    }
//}

void cvx::hmean(cv::Mat src, cv::Mat& mean)
{
    mean.release();
    
    cv::reduce(src, mean, 1, CV_REDUCE_AVG);
}

void cvx::vmean(cv::Mat src, cv::Mat& mean)
{
    mean.release();
    
    cv::reduce(src, mean, 0, CV_REDUCE_AVG);
}

void cvx::hist(cv::Mat src, int nbins, float min, float max, cv::Mat& hist)
{
    int histSize[] = { nbins };
    int channels[] = { 0 }; // 1 channel, number 0
    float range[] = { min, max } ;
    const float* ranges[] = { range };
    
    src.convertTo(src, cv::DataType<float>::type);
    
    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    
    hist.convertTo(hist, cv::DataType<int>::type);
}

void cvx::hist(cv::Mat src, cv::Mat msk, int nbins, float min, float max, cv::Mat& hist)
{
    int histSize[] = { nbins };
    int channels[] = { 0 }; // 1 channel, number 0
    float range[] = { min, max } ;
    const float* ranges[] = { range };
    
    src.convertTo(src, cv::DataType<float>::type);
    
    cv::calcHist(&src, 1, channels, msk, hist, 1, histSize, ranges, true, false);
    
    hist.convertTo(hist, cv::DataType<int>::type);
}

void cvx::cumsum(cv::Mat src, cv::Mat& dst)
{
    dst = src.clone();
    
    //cv::add(dst.row(0), src.row(0), dst.row(0));
    
    for (int i = 0; i < src.rows - 1; i++)
        for (int j = i+1; j < src.rows; j++)
            cv::add(dst.row(j), src.row(i), dst.row(j));
    
    for (int i = 0; i < src.cols - 1; i++)
        for (int j = i+1; j < src.cols; j++)
            cv::add(dst.col(j), src.col(j), dst.col(j));
}

cv::Mat cvx::linspace(int start, int end)
{
    cv::Mat l;
    
    cvx::linspace(start, end, l);
    
    return l;
}

cv::Mat cvx::powspace(int start, int end, int base)
{
    cv::Mat l;
    cvx::linspace(start, end, l);

    cv::Mat p (l.rows, 1, cv::DataType<int>::type);
    for (int i = 0; i < l.rows; i++)
    {
        p.at<int>(i,0) = powf(base, l.at<int>(i,0));
    }
    
    return p;
}

cv::Mat cvx::linspace(double start, double end, int n)
{
    cv::Mat l;
    
    cvx::linspace(start, end, n, l);
    
    return l;
}

void cvx::linspace(int start, int end, cv::Mat& mat)
{
    std::vector<int> v;
    cvx::linspace(start, end, v);
    
    mat.create(v.size(), 1, cv::DataType<int>::type);
    memcpy(mat.data, v.data(), sizeof(int) * v.size());
}

void cvx::linspace(double start, double end, int n, cv::Mat& mat)
{
    std::vector<double> v;
    cvx::linspace(start, end, n, v);

    mat.create(v.size(), 1, cv::DataType<double>::type);
    memcpy(mat.data, v.data(), sizeof(double) * v.size());
}

void cvx::linspace(int start, int end, std::vector<int>& v)
{
    for (int i = start; i < end; i++)
    {
        v.push_back(i);
    }
}

void cvx::linspace(double start, double end, int n, std::vector<double>& v)
{
    for (int i = 0; i < n; i++)
    {
        v.push_back(start + i * (end - start) / (n - 1));
    }
}

void cvx::load(std::string file, cv::Mat& mat, int format)
{
    cv::FileStorage fs(file, cv::FileStorage::READ | format);
    
    fs["mat"] >> mat;
    
    fs.release();
    
}

void cvx::save(std::string file, cv::Mat mat, int format)
{
    cv::FileStorage fs(file, cv::FileStorage::WRITE | format);
    
    fs << "mat" << mat;
    
    fs.release();
    
}

//template<typename T>
//cv::Mat cvx::matlabread(std::string file)
//{
//    cv::Mat mat;
//    
//    cvx::matlabread<T>(file, mat);
//    
//    return mat;
//}
//
//template<typename T>
//void cvx::matlabread(std::string file, cv::Mat& mat)
//{
//    mat_t *matfp;
//    matvar_t *matvar;
//
//    matfp = Mat_Open(file.c_str(), MAT_ACC_RDONLY);
//    
//    if ( NULL == matfp ) {
//        fprintf(stderr,"Error opening MAT file \"%s\"!\n", file.c_str());
//        return;
//    }
//
//    matvar = Mat_VarRead(matfp,"perMap");
//    if ( NULL == matvar ) {
//        fprintf(stderr, "Variable ’perMap’ not found, or error "
//                "reading MAT file\n");
//    }
//    else
//    {
//        int nrows, ncols;
//        nrows = matvar->dims[0];
//        ncols = matvar->dims[1];
//        
//        cv::Mat rmap (nrows, ncols, cv::DataType<T>::type, matvar->data);
//        mat = rmap.clone();
//        
//        Mat_VarFree(matvar);
//    }
//    
//    Mat_Close(matfp);
//}

void cvx::computePCA(cv::Mat src, cv::PCA& pca, cv::Mat& dst, int flags, double variance)
{
    pca.computeVar(src, cv::noArray(), flags, variance);
    
    dst.release();
    
    for (int i = 0; i < (flags == CV_PCA_DATA_AS_ROW ? src.rows : src.cols); i++)
    {
        cv::Mat p;
        if (flags == CV_PCA_DATA_AS_ROW)
            p = pca.project(src.row(i));
        else
            p = pca.project(src.col(i));
        
        dst.push_back(p);
    }
}

cv::Mat cvx::standardize(cv::Mat m, int dim)
{
    cv::Mat s (m.rows, m.cols, cv::DataType<float>::type);
    
    // iterate over variable dimension: rows-wise (0) or column-wise (1)
    for (int i = 0; i < (dim == 0 ? m.rows : m.cols); i++)
    {
        cv::Mat rowcol (dim == 0 ? m.row(i) : m.col(i));
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(rowcol, mean, stddev);
        cv::Mat stdrowcol = (rowcol - mean.val[0]) / stddev.val[0];
        rowcol.copyTo(dim == 0 ? s.row(i) : s.col(i));
    }
    
    return s;
}

void cvx::fillErrorsWithMedian(cv::InputArray src, int size, cv::OutputArray dst)
{
    cv::Mat _src = src.getMat();
    cv::Mat _dst (_src.rows, _src.cols, _src.type(), cv::Scalar(0));
    
    for (int i = size; i < _src.rows - size; i++) for (int j = size; j < _src.cols - size; j++)
    {
        if (_src.at<unsigned short>(i,j) > 0)
            _dst.at<unsigned short>(i,j) = _src.at<unsigned short>(i,j);
        else
        {
            int neighboringErrors = 0;
            std::vector<unsigned short> values;
            
            for (int y = -size; y <= size; y++) for (int x = -size; x <= size; x++)
            {
                int value = _src.at<unsigned short>(i+y, j+x);
                
                if (value == 0)
                    neighboringErrors++;
                else
                    values.push_back(value);
            }
            
            if (neighboringErrors < values.size() / 2)
            {
                std::sort(values.begin(), values.end());
                _dst.at<unsigned short>(i,j) = values[values.size() / 2];
            }
        }
    }
    
    dst.getMatRef() = _dst;
}

void cvx::dilate(cv::InputArray src, int size, cv::OutputArray dst)
{
    cv::Mat _src = src.getMat();
    cv::Mat& _dst = dst.getMatRef();
    
    if (size == 0)
    {
        _src.copyTo(_dst);
    }
    else if (size > 0)
    {
        cv::Mat element = cv::getStructuringElement( cv::MORPH_DILATE, cv::Size( 2 * size + 1, 2 * size + 1 ), cv::Point( size, size ) );
        
        cv::Mat aux;
        cv::dilate(_src, _dst, element);
    }
}

void cvx::erode(cv::InputArray src, int size, cv::OutputArray dst)
{
    cv::Mat _src = src.getMat();
    cv::Mat& _dst = dst.getMatRef();
    
    if (size == 0)
    {
        _src.copyTo(_dst);
    }
    else if (size > 0)
    {
        cv::Mat element = cv::getStructuringElement( cv::MORPH_ERODE, cv::Size( 2 * size + 1, 2 * size + 1 ), cv::Point( size, size ) );
        
        cv::Mat aux;
        cv::erode(_src, _dst, element);
    }
}

void cvx::open(cv::InputArray src, int size, cv::OutputArray dst)
{
//    cv::Mat aux;
//    erode(src.getMat(), size, aux);
//    dilate(aux, size, dst.getMatRef());
    
    cv::Mat _src = src.getMat();
    cv::Mat& _dst = dst.getMatRef();
    
    if (size == 0)
    {
        _src.copyTo(_dst);
    }
    else if (size > 0)
    {
        int erosion_size, dilation_size;
        erosion_size = dilation_size = size;
        
        cv::Mat erode_element = cv::getStructuringElement( cv::MORPH_ERODE, cv::Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ), cv::Point( erosion_size, erosion_size ) );
        cv::Mat dilate_element = cv::getStructuringElement( cv::MORPH_DILATE, cv::Size( 2 * dilation_size + 1, 2 * dilation_size + 1 ), cv::Point( dilation_size, dilation_size ) );
        
        cv::Mat aux;
        cv::erode(_src, aux, erode_element);
        cv::dilate(aux, _dst, dilate_element);
    }
}

void cvx::close(cv::InputArray src, int size, cv::OutputArray dst)
{
//    cv::Mat aux;
//    dilate(src.getMat(), size, aux);
//    erode(aux, size, dst.getMatRef());

    cv::Mat _src = src.getMat();
    cv::Mat& _dst = dst.getMatRef();
    
    if (size == 0)
    {
        _src.copyTo(_dst);
    }
    else if (size > 0)
    {
        int erosion_size, dilation_size;
        erosion_size = dilation_size = size;
        
        cv::Mat erode_element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ), cv::Point( erosion_size, erosion_size ) );
        cv::Mat dilate_element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2 * dilation_size + 1, 2 * dilation_size + 1 ), cv::Point( dilation_size, dilation_size ) );
        
        cv::Mat aux;
        cv::dilate(_src, aux, erode_element);
        cv::erode(aux, _dst, dilate_element);
    }
}

float cvx::overlap(cv::InputArray src1, cv::InputArray src2)
{
    cv::Mat bin1 = (src1.getMat() > 0);
    cv::Mat bin2 = (src2.getMat() > 0);
    
    cv::Mat bin1and2, bin1or2;
    cv::bitwise_and(bin1, bin2, bin1and2);
    cv::bitwise_or(bin1, bin2, bin1or2);
    
    // the factor 255 of the masks are inter-cancelled
    return cv::sum(bin1and2).val[0] / cv::sum(bin1or2).val[0];
}

// Measures the amount of src1 that is in src2
float cvx::contention(cv::InputArray src1, cv::InputArray src2)
{
    cv::Mat bin1 = (src1.getMat() > 0);
    cv::Mat bin2 = (src2.getMat() > 0);
    
    cv::Mat bin1and2, bin1or2;
    cv::bitwise_and(bin1, bin2, bin1and2);
    
    // the factor 255 of the masks are inter-cancelled
    return cv::sum(bin1and2).val[0] / cv::sum(bin1).val[0];
}

cv::Mat cvx::replicate(cv::Mat src, int times)
{
    std::vector<int> from_to (2*times);
    for (int i = 0; i < times; i++)
    {
        from_to[2*i] = 0;
        from_to[2*i+1] = i;
    }
    
    cv::Mat replicatedSrc ( src.rows, src.cols, CV_MAKETYPE(src.type(), times));
    cv::mixChannels(&src, 1, &replicatedSrc, 1, from_to.data(), times);
    
    return replicatedSrc;
}

void cvx::mean(cv::InputArray acc, cv::InputArray count, cv::OutputArray mean)
{
    cv::Mat _acc = acc.getMat();
    cv::Mat _count = count.getMat();
    cv::Mat& _mean = mean.getMatRef();
    
    cv::Mat accAux;
    _acc.convertTo(accAux, cv::DataType<float>::type);
    
    cv::Mat countAux;
    _count.convertTo(countAux, cv::DataType<float>::type);
    
    cv::divide(accAux, replicate(countAux, accAux.channels()), _mean);
}

void cvx::stddev(cv::InputArray sqdiffsacc, cv::InputArray count, cv::InputArray err, cv::OutputArray stddev)
{
    int nchannels = sqdiffsacc.getMat().channels();
    
    // Compute the corrected standard deviation a sample
    cv::Mat sqdiffsacc_32F, count_32F;
    sqdiffsacc.getMat().convertTo(sqdiffsacc_32F, CV_MAKETYPE(CV_32F, nchannels));
    count.getMat().convertTo(count_32F, CV_32F);
    
    cv::Mat multiplier = 1 / (count_32F - 1); // basically: 1/(N-1)
    if (err.kind() > 0 && !err.getMat().empty()) multiplier.setTo(0, err.getMat());
    
    cv::Mat aux;
    cv::multiply(replicate(multiplier, nchannels), sqdiffsacc_32F, aux);
    cv::sqrt(aux, stddev);
}

template<typename T>
int cvx::match(cv::Mat m, cv::Mat query, bool debug)
{
    assert (m.cols == query.cols || m.rows == query.rows);
    
    bool colwise = (m.cols == query.cols);
    
    if (debug) std::cout << query << std::endl;
    
    bool bFound = false;
    int match = -1;
    for (int i = 0; (colwise ? (i < m.rows) : (i < m.cols)) && !bFound; i++)
    {
        if (debug) std::cout << m.row(i) << std::endl;
        
        bFound = true;
        for (int p = 0; (colwise ? (p < m.cols) : (p < m.rows)) && bFound; p++)
        {
            T t1 = colwise ? m.at<T>(i,p) : m.at<T>(p,i);
            T t2 = colwise ? query.at<T>(0,p) : query.at<T>(p,0);
            
            if (debug) std::cout << t1 << " " << t2 << std::endl;
            
            bFound &= (t1 == t2);
            if (debug) std::cout << bFound << std::endl;
        }
        
        if (bFound) match = i;
    }
    
    return match;
}

void cvx::match(cv::Mat src, cv::Mat queries, cv::Mat& matched, cv::Mat& pending)
{
    std::vector<cv::Mat> matchedRows;
    std::vector<cv::Mat> pendingRows;
    
    for (int i = 0; i < queries.rows; i++)
    {
        bool bMatch = false;
        for (int j = 0; j < src.rows && !bMatch; j++)
        {
            cv::Mat nonZeros;
            cv::findNonZero(src.row(j) == queries.row(i), nonZeros);
            bMatch = (nonZeros.rows == queries.cols);
        }
        
        if (bMatch)
            matchedRows.push_back(queries.row(i));
        else
            pendingRows.push_back(queries.row(i));
    }
    
    cvx::vconcat(matchedRows, matched);
    cvx::vconcat(pendingRows, pending);
}

void cvx::sortIdx(cv::InputArray src, cv::OutputArray ind, int flags, cv::OutputArray dst)
{
    cv::Mat _src = src.getMat();
    cv::Mat _ind;
    
    cv::sortIdx(_src, _ind, flags);
    
    if (dst.kind() > 0)
    {
        cv::Mat _dst (_src.rows, _src.cols, _src.type(), cv::Scalar(0));
        
        bool colwise = (flags & CV_SORT_EVERY_COLUMN) > 0;
        for (int j = 0; j < (colwise ? _ind.cols : _ind.rows); j++)
        {
            for (int i = 0; i < (colwise ? _ind.rows : _ind.cols); i++)
            {
                int idx = (colwise ? _ind.at<int>(i,j) : _ind.at<int>(j,i));
                
                if (colwise) _src.col(j).row(idx).copyTo(_dst.col(j).row(i));
                else _dst.row(j).col(i) = _src.row(j).col(idx);
            }
        }
        
        dst.getMatRef() = _dst;
    }
    
    ind.getMatRef() = _ind;
}

void cvx::harmonicMean(cv::InputArray src, cv::OutputArray dst, int dim)
{
    cv::Mat _src = src.getMat();
    cv::Mat _dst (dim > 0 ? _src.rows : 1,
                  dim > 0 ? 1 : _src.cols,
                  src.type());
    
    for (int i = 0; i < (dim > 0 ? _src.rows : _src.cols); i++) // colwise mean ?
    {
        cv::Mat invacc;
        cv::reduce(1.f / (dim > 0 ? _src.row(i) : _src.col(i)), invacc, dim, CV_REDUCE_SUM);
        
        double card = dim > 0 ? _src.cols : _src.rows;
        cv::Mat hm = card / invacc;
        hm.copyTo(_dst.row(i));
    }
    
    dst.getMatRef() = _dst;
}


template<typename T>
void cvx::convert(cv::Mat mat, std::list<std::vector<T> >& lv)
{
    lv.clear();
    
    for (int i = 0; i < mat.rows; i++)
    {
        std::vector<T> v (mat.cols);
        
        for (int j = 0; j < mat.cols; j++)
            v[j] = mat.at<T>(i,j);
        
        lv.push_back(v);
    }
}

template<typename T>
void cvx::convert(cv::Mat mat, std::vector<std::vector<T> >& vv)
{
    vv.clear();
    
    vv.resize(mat.rows);
    for (int i = 0; i < mat.rows; i++)
    {
        vv[i].resize(mat.cols);
        
        for (int j = 0; j < mat.cols; j++)
            vv[i][j] = mat.at<T>(i,j);
    }
}

template<typename T>
void cvx::convert(const std::vector<std::vector<T> >& vv, cv::Mat& mat)
{
    if (vv.size() == 0 || vv[0].size() == 0)
        return;
    
    mat.create(vv.size(), vv[0].size(), cv::DataType<T>::type);
    for (int i = 0; i < mat.rows; i++)
    {
        assert( vv[i].size() == mat.cols );
        for (int j = 0; j < mat.cols; j++)
            mat.at<T>(i,j) = vv[i][j];
    }
}


template<typename T>
void cvx::convert(const std::list<std::vector<T> >& lv, cv::Mat& mat)
{
    if (lv.size() == 0 || lv.front().size() == 0)
        return;
    
    mat.create(lv.size(), lv.front().size(), cv::DataType<T>::type);
    
    typename std::list<std::vector<T> >::const_iterator it = lv.begin();
    int i = 0;
    for ( ; it != lv.end(); ++it, ++i)
    {
        assert( it->size() == mat.cols );
        for (int j = 0; j < mat.cols; j++)
            mat.at<T>(i,j) = it->at(j);
    }
}

template<typename T>
cv::Mat cvx::convert(const std::vector<std::vector<T> >& vv)
{
    cv::Mat m;
    
    convert<T>(vv,m);
    
    return m;
}

template<typename T>
cv::Mat cvx::convert(const std::list<std::vector<T> >& lv)
{
    cv::Mat m;
    
    convert<T>(lv,m);
    
    return m;
}

void cvx::vconcat(std::vector<cv::Mat> array, cv::Mat& mat)
{
    if (array.size() > 0)
    {
        int length = 0;
        for (int i = 0; i < array.size(); i++)
            length += array[i].rows;
        
        mat.create(length, array[0].cols, array[0].type());
        
        int ptr = 0;
        for (int i = 0; i < array.size(); i++)
        {
            assert(array[i].cols == array[0].cols);
            assert(array[i].type() == array[0].type());
            
            cv::Mat roi (mat, cv::Rect(0, ptr, array[i].cols, array[i].rows));
            array[i].copyTo(roi);
            
            ptr += array[i].rows;
        }
    }
}

void cvx::hconcat(std::vector<cv::Mat> array, cv::Mat& mat)
{
    if (array.size())
    {
        int length = 0;
        for (int i = 0; i < array.size(); i++)
             length += array[i].cols;
        
        mat.create(array[0].rows, length, array[0].type());
        
        int ptr = 0;
        for (int i = 0; i < array.size(); i++)
        {
            assert(array[i].rows == array[0].rows);
            assert(array[i].type() == array[0].type());
            
            cv::Mat roi (mat, cv::Rect(ptr, 0, array[i].cols, array[i].rows));
            array[i].copyTo(roi);
            
            ptr += array[i].cols;
        }
    }
}

cv::Point cvx::diffIdx(cv::Mat src1, cv::Mat src2)
{
    assert (src1.rows == src2.rows && src1.cols == src2.cols && src1.channels() == src2.channels());
    
    for (int i = 0; i < src1.rows; i++) for (int j = 0; j < src1.cols; j++)
    {
        cv::Mat inequality = ( src1.row(i).col(j) != src2.row(i).col(j) );
        if (inequality.at<uchar>(0,0))
            return cv::Point(j,i);
    }
    
    return cv::Point(-1,-1);
}

void cvx::unique(cv::Mat M, int dim, cv::Mat& U, cv::Mat& I)
{
    std::vector<cv::Mat> elements;
    std::vector<int> indices;
    for (int i = 0; i < ((dim == 0) ? M.rows : M.cols); i++)
    {
        cv::Mat e = ((dim == 0) ? M.row(i) : M.col(i));
        
        if (i == 0)
        {
            elements.push_back(e);
            indices.push_back(elements.size() - 1);
        }
        else
        {
            int matchIdx = -1;
            for (int j = 0; j < elements.size() && matchIdx < 0; j++)
            {
                if ( cv::sum(elements[j] - e).val[0] == 0 )
                    matchIdx = j;
            }
            
            if (matchIdx < 0)
            {
                // New unique
                elements.push_back(e);
                indices.push_back(elements.size() - 1); // set new idx to the new element
            }
            else
            {
                indices.push_back(matchIdx); // set match idx
            }
        }
    }
    
    U.create(((dim == 0) ? elements.size() : M.rows), ((dim == 0) ? M.cols : elements.size()), M.type());
    I.create(((dim == 0) ? indices.size() : 1), ((dim == 0) ? 1 : elements.size()), cv::DataType<int>::type);
    
    for (int i = 0; i < elements.size(); i++)
        elements[i].copyTo( ((dim == 0) ? U.row(i) : U.col(i)) );
    for (int i = 0; i < indices.size(); i++)
        I.at<int>((dim == 0) ? i : 0, (dim == 0) ? 0 : i) = indices[i];
}

void cvx::findRowsToColsAssignation(cv::Mat scores, cv::Mat& matches)
{
    cv::Mat M (scores.size(), CV_8U, cv::Scalar(0)); // matches
    cv::Mat A = M.clone(); // assignations
    
    for (int i = 0; i < scores.rows && i < scores.cols; i++)
    {
        double minVal, maxVal;
        cv::Point minIdx, maxIdx;
        cv::minMaxLoc(scores, &minVal, &maxVal, &minIdx, &maxIdx, ~A);

        M.at<uchar>(minIdx.y, minIdx.x) = 255;
        A.col(minIdx.x) = 255;
        A.row(minIdx.y) = 255;
    }
    
    // Get the matches indices
    cv::findNonZero(M, matches);
}

void cvx::combine(cv::InputArray src1, cv::InputArray src2, int dim, cv::OutputArray dst)
{
    cv::Mat _src1 = src1.getMat();
    cv::Mat _src2 = src2.getMat();
    
    assert( _src1.type() == _src2.type() );
    
    cv::Mat _dst;
    if (dim == 0)
        _dst.create(_src1.rows + _src2.rows, _src1.cols * _src2.cols, _src1.type());
    else
        _dst.create(_src1.rows * _src2.rows, _src1.cols + _src2.cols, _src1.type());
    
    
    for (int i = 0; i < ((dim == 0) ? _src1.cols : _src1.rows); i++)
    {
        for (int j = 0; j < ((dim == 0) ? _src2.cols : _src2.rows); j++)
        {
            if (dim == 0)
            {
                _src1.col(i).copyTo(_dst.col(i * _src2.cols + j).colRange(0, _src1.rows));
                _src2.col(j).copyTo(_dst.col(i * _src2.cols + j).colRange(_src1.rows, _src1.rows + _src2.rows));
            }
            else
            {
                _src1.row(i).copyTo(_dst.row(i * _src2.rows + j).colRange(0, _src1.cols));
                _src2.row(j).copyTo(_dst.row(i * _src2.rows + j).colRange(_src1.cols, _src1.cols + _src2.cols));
            }
        }
    }
    
    dst.getMatRef() = _dst;
}

cv::Rect cvx::rectangleIntersection(cv::Rect a, cv::Rect b)
{
    int aib_min_x = std::max(a.x, b.x);
    int aib_min_y = std::max(a.y, b.y);
    int aib_max_x = std::min(a.x + a.width - 1, b.x + b.width - 1);
    int aib_max_y = std::min(a.y + a.height - 1, b.y + b.height - 1);
    
    int width = aib_max_x - aib_min_x + 1;
    int height = aib_max_y - aib_min_y + 1;
    cv::Rect aib (aib_min_x,
                  aib_min_y,
                  width > 0 ? width : 0,
                  height > 0 ? height : 0);
    return aib;
}

// Auxiliary function for varianceSubsampling(...)
float cvx::rectangleOverlap(cv::Rect a, cv::Rect b)
{
    cv::Rect aib = cvx::rectangleIntersection(a,b);
    
    float overlap = ((float) aib.area()) / (a.area() + b.area() - aib.area());
    
    return overlap;
}

// Auxiliary function for getTrainingCloudjects(...)
// Check the amount of inclusion of "a" in "b"
float cvx::rectangleInclusion(cv::Rect a, cv::Rect b)
{
    cv::Rect aib = cvx::rectangleIntersection(a,b);
    
    float inclusion = ((float) aib.area()) / std::min(a.area(),b.area());
    
    return inclusion;
}

// Weighted overlap between two Region objects. It uses the rectangles
// for the efficient computation of the intersetion. If the intersection
// is not empty, compute in the intersection region the overlap measure.
// Eventually, weight this overlap by the minimum size of the two
// non-cropped overlapping regions.
float cvx::weightedOverlapBetweenRegions(cv::Mat src1, cv::Mat src2, cv::Rect r1, cv::Rect r2)
{
    cv::Rect irect = cvx::rectangleIntersection(r1, r2);
    
    // RETURN 0 (no intersection)
    if ( !(irect.width > 0 && irect.height > 0) )
        return .0f;
    
    // Roi the intersection in the two patches
    cv::Mat roi1 ( src1, cv::Rect(irect.x - r1.x, irect.y - r1.y,
                                    irect.width, irect.height) );
    cv::Mat roi2 ( src2, cv::Rect(irect.x - r2.x, irect.y - r2.y,
                                    irect.width, irect.height) );
    assert (roi1.rows == roi2.rows && roi1.cols == roi2.cols);
    
    cv::Mat iroi = roi1 & roi2;
    
    int count1 = cv::countNonZero(src1); // A
    int count2 = cv::countNonZero(src2); // B
    int icount = cv::countNonZero(iroi); // A,B , i.e. A and B
    
    // Overlap calculion based on Jaccard index: |A,B|/|AUB|
    float ovl = ((float) icount) / (count1 + count2 - icount); // {AUB} = A + B - {A,B}
    float weight = (((float) icount) / std::min(count1,count2)); // scale in proportion to min(A,B)
    
    float wovl = ovl * weight;
    
    // RETURN the weighted overlap
    return wovl;
}

template void cvx::convert(cv::Mat mat, std::vector<std::vector<uchar> >& vv);
template void cvx::convert(cv::Mat mat, std::vector<std::vector<ushort> >& vv);
template void cvx::convert(cv::Mat mat, std::vector<std::vector<int> >& vv);
template void cvx::convert(cv::Mat mat, std::vector<std::vector<float> >& vv);
template void cvx::convert(cv::Mat mat, std::vector<std::vector<double> >& vv);

template void cvx::convert(cv::Mat mat, std::list<std::vector<uchar> >& lv);
template void cvx::convert(cv::Mat mat, std::list<std::vector<ushort> >& lv);
template void cvx::convert(cv::Mat mat, std::list<std::vector<int> >& lv);
template void cvx::convert(cv::Mat mat, std::list<std::vector<float> >& lv);
template void cvx::convert(cv::Mat mat, std::list<std::vector<double> >& lv);

template void cvx::convert(const std::vector<std::vector<uchar> >& vv, cv::Mat& mat);
template void cvx::convert(const std::vector<std::vector<ushort> >& vv, cv::Mat& mat);
template void cvx::convert(const std::vector<std::vector<int> >& vv, cv::Mat& mat);
template void cvx::convert(const std::vector<std::vector<float> >& vv, cv::Mat& mat);
template void cvx::convert(const std::vector<std::vector<double> >& vv, cv::Mat& mat);

template void cvx::convert(const std::list<std::vector<uchar> >& lv, cv::Mat& mat);
template void cvx::convert(const std::list<std::vector<ushort> >& lv, cv::Mat& mat);
template void cvx::convert(const std::list<std::vector<int> >& lv, cv::Mat& mat);
template void cvx::convert(const std::list<std::vector<float> >& lv, cv::Mat& mat);
template void cvx::convert(const std::list<std::vector<double> >& lv, cv::Mat& mat);

template cv::Mat cvx::convert(const std::vector<std::vector<uchar> >& vv);
template cv::Mat cvx::convert(const std::vector<std::vector<ushort> >& vv);
template cv::Mat cvx::convert(const std::vector<std::vector<int> >& vv);
template cv::Mat cvx::convert(const std::vector<std::vector<float> >& vv);
template cv::Mat cvx::convert(const std::vector<std::vector<double> >& vv);

template cv::Mat cvx::convert(const std::list<std::vector<uchar> >& lv);
template cv::Mat cvx::convert(const std::list<std::vector<ushort> >& lv);
template cv::Mat cvx::convert(const std::list<std::vector<int> >& lv);
template cv::Mat cvx::convert(const std::list<std::vector<float> >& lv);
template cv::Mat cvx::convert(const std::list<std::vector<double> >& lv);

template int cvx::match<int>(cv::Mat m, cv::Mat query, bool debug);
template int cvx::match<float>(cv::Mat m, cv::Mat query, bool debug);
template int cvx::match<double>(cv::Mat m, cv::Mat query, bool debug);

//template cv::Mat cvx::matlabread<int>(std::string file);
//template cv::Mat cvx::matlabread<float>(std::string file);
//template cv::Mat cvx::matlabread<double>(std::string file);
//
//template void cvx::matlabread<int>(std::string file, cv::Mat& mat);
//template void cvx::matlabread<float>(std::string file, cv::Mat& mat);
//template void cvx::matlabread<double>(std::string file, cv::Mat& mat);
