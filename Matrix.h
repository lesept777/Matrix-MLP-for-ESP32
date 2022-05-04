/* 
    Matrix linear algebre library for ML applications
    Inspred by: https://www.quantstart.com/articles/Matrix-Classes-in-C-The-Source-File/
    Some errors were corrected, methods and constructors added

    (c) Lesept dec. 2021  lesept777@gmail.com

*/


#ifndef __ML_MATRIX_H
#define __ML_MATRIX_H

#include <Arduino.h>

template <typename T> class MLMatrix {
 private:
    std::vector<std::vector<T> > mat;
    unsigned rows;
    unsigned cols;

 public:
    MLMatrix();
    MLMatrix(unsigned _rows, unsigned _cols, const T& _initial = 0);    // constant matrix
    MLMatrix(unsigned _rows, unsigned _cols, const T min, const T max); // random matrix
    MLMatrix(const MLMatrix<T>& rhs);
    MLMatrix(const MLMatrix<T>& rhs, const T& _initial); // copy the size of a matrix, fill with initial
    MLMatrix(const std::vector<T>& rhs);
    MLMatrix(const T& rhs, const unsigned dim);
    virtual ~MLMatrix();

    // Identity matrix
    MLMatrix<T>& Id(const unsigned dim);

    // Operator overloading, for "standard" mathematical matrix operations                                                                                                                                                          
    MLMatrix<T>& operator=(const MLMatrix<T>& rhs);  // copy a matrix
    MLMatrix<T>& operator=(const std::vector<T>& rhs); // copy a vector to a matrix (n,1)
    MLMatrix<T>& operator=(const T rhs[]); // copy from 1D array 

    // Matrix mathematical operations                                                                                                                                                                                               
    MLMatrix<T>  operator+ (const MLMatrix<T>& rhs) const;
    MLMatrix<T>& operator+=(const MLMatrix<T>& rhs);
    MLMatrix<T>  operator- (const MLMatrix<T>& rhs) const;
    MLMatrix<T>& operator-=(const MLMatrix<T>& rhs);
    MLMatrix<T>  operator* (const MLMatrix<T>& rhs) const;
    MLMatrix<T>& operator*=(const MLMatrix<T>& rhs);
    MLMatrix<T>  transpose();
    MLMatrix<T>  Hadamard (const MLMatrix<T>& rhs, bool clip=false);

    // Matrix/scalar operations                                                                                                                                                                                                     
    MLMatrix<T>  operator+(const T& rhs);
    MLMatrix<T>& operator+=(const T& rhs);
    MLMatrix<T>  operator-(const T& rhs);
    MLMatrix<T>& operator-=(const T& rhs);
    MLMatrix<T>  operator*(const T& rhs);
    MLMatrix<T>& operator*=(const T& rhs);
    MLMatrix<T>  operator/(const T& rhs);
    MLMatrix<T>& operator/=(const T& rhs);

    // Matrix/vector operations                                                                                                                                                                                                     
    std::vector<T> operator*(const std::vector<T>& rhs);
    std::vector<T> diag_vec();
    MLMatrix<T> applySelf(T (*function)(T));
    MLMatrix<T> apply(T (*function)(T)) ;

    // Vector operations
    T MdotProd(const MLMatrix<T>& rhs, bool clip=false);
    MLMatrix<T> times(const MLMatrix<T>& rhs, bool clip=false);

    // Matrix comparison
    const bool  operator==( const MLMatrix<T> &rhs ) const;
    const bool  operator!=( const MLMatrix<T> &rhs ) const;
    MLMatrix<T> operator< ( const MLMatrix<T> &rhs );
    MLMatrix<T> operator>=( const MLMatrix<T> &rhs );

    // Access the individual elements: mat(i,j)
    T& operator()(const unsigned& row, const unsigned& col);
    const T& operator()(const unsigned& row, const unsigned& col) const;

    // Access the row and column sizes                                                                                                                                                                                              
    unsigned get_rows() const;
    unsigned get_cols() const;

    // Norms
    int L0Norm();
    T L1Norm();
    T L2Norm();
    T max() const;
    T min() const;
    float mean() const;
    float stdev(const float) const;

    // Extract row or col
    MLMatrix<T> row(const uint32_t);
    MLMatrix<T> col(const uint32_t);
    MLMatrix<T> subMatrix(const uint32_t, const uint32_t, const uint32_t, const uint32_t);

    // Display the matrix
    void print();
    void printSize(); // Only display the size (rows, cols)

    // Find min and max values index
    void indexMin(int &, int &);
    void indexMax(int &, int &);

    // Misc
    MLMatrix<T> randomChange(const float);
    MLMatrix<T> normScale (float, bool &);
    void normScale2 (float, bool &);
    int clipToZero (float);
    int clipMin (float);
    int clipMax (float);
    MLMatrix<T> sgn();
    MLMatrix<T> randomNormal(const float, const float);
};

#include "matrix.cpp"

/* Generate Identity matrix
    Use like this:
        MLMatrix<int> mat = Id<int>(4);
        MLMatrix<float> mat = Id<float>(4);
*/

template<typename T>
inline MLMatrix<T> Id(const int dim) {
    MLMatrix<T> result(dim, dim, 0);
    for (unsigned i=0; i<dim; i++) result(i,i) = T(1);
    return result;
}

// Vector dot product
/* Example usage
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {4, 5, 6};
  int P = dotProd(v1, v2, false);
*/
template<typename T>
T dotProd(const std::vector<T>& a, const std::vector<T>& b, bool clip=false)
{
    if (a.size() != b.size()) {
        Serial.printf("Dot product error: dimensions do not match (%d, %d)\n",a.size(), b.size());
        while(1);
    }
    if (! clip) {
        T sum = std::inner_product(std::begin(a), std::end(a), std::begin(b), 0);
        return sum;
    } else { // Clip the result at min and max value of type T
        float sum = std::inner_product(std::begin(a), std::end(a), std::begin(b), 0.0);
        if (sum < std::numeric_limits<T>::min()) sum = std::numeric_limits<T>::min();
        if (sum > std::numeric_limits<T>::max()) sum = std::numeric_limits<T>::max();
        return T(sum);
    }
}

#endif