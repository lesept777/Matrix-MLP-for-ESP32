#ifndef __ML_MATRIX_CPP
#define __ML_MATRIX_CPP

#include "matrix.h"

/***************************************************************************************************
  Constructors
*/
// Parameter Constructor
template<typename T>
MLMatrix<T>::MLMatrix() {
//
}

// Create and fill a matrix (with 0s if not specified)
template<typename T>
MLMatrix<T>::MLMatrix(unsigned _rows, unsigned _cols, const T& _initial) {
  mat.resize(_rows);
  for (unsigned i=0; i<mat.size(); ++i) {
    mat[i].resize(_cols, _initial);
  }
  rows = _rows;
  cols = _cols;
}

// Copy Constructor
template<typename T>
MLMatrix<T>::MLMatrix(const MLMatrix<T>& rhs) {
  mat = rhs.mat;
  rows = rhs.get_rows();
  cols = rhs.get_cols();
}

// Copy Constructor (copy the dimensions, fill with constant value)
template<typename T>
MLMatrix<T>::MLMatrix(const MLMatrix<T>& rhs, const T& _initial) {
  rows = rhs.get_rows();
  cols = rhs.get_cols();
  mat.resize(rows);
  for (unsigned i=0; i<mat.size(); ++i) {
    mat[i].resize(cols, _initial);
  }
}

template<typename T>
MLMatrix<T>::MLMatrix(const std::vector<T>& rhs) {
  rows = rhs.size();
  cols = 1;
  for (unsigned i=0; i<rows; ++i)
    mat[i][0] = rhs[i];
}

template<typename T>
MLMatrix<T>::MLMatrix(const T& rhs, const unsigned dim) {
  rows = dim;
  cols = 1;
  mat.resize(rows);
}

// random matrix Constructor
template <typename T>
MLMatrix<T>::MLMatrix(unsigned _rows, unsigned _cols, const T min, const T max)
{
  mat.resize(_rows);
  for (unsigned i=0; i<mat.size(); ++i) {
    mat[i].resize(_cols);
    for (unsigned j=0; j<mat[i].size(); ++j) {
      if (std::is_floating_point<T>::value) { // float
        float x = esp_random();
        x /= UINT32_MAX;
        mat[i][j] = min + x * (max - min);
      } else {
        mat[i][j] = random(min, max);
      }
    }
  }
  rows = _rows;
  cols = _cols;
}

// (Virtual) Destructor
template<typename T>
MLMatrix<T>::~MLMatrix() {}

/***************************************************************************************************
  Initialization methods
*/
// Assignment Operator
template<typename T>
MLMatrix<T>& MLMatrix<T>::operator=(const MLMatrix<T>& rhs) {
  if (&rhs == this)
    return *this;

  unsigned new_rows = rhs.get_rows();
  unsigned new_cols = rhs.get_cols();

  mat.resize(new_rows);
  for (unsigned i=0; i<mat.size(); ++i) {
    mat[i].resize(new_cols);
  }

  for (unsigned i=0; i<new_rows; ++i) {
    for (unsigned j=0; j<new_cols; ++j) {
      mat[i][j] = rhs(i, j);
    }
  }
  rows = new_rows;
  cols = new_cols;

  return *this;
}

// Assignment from vector
template<typename T>
MLMatrix<T>& MLMatrix<T>::operator=(const std::vector<T>& rhs) {
  unsigned new_rows = rhs.size();
  unsigned new_cols = 1;

/* ???
  mat.resize(0);
  mat[0].resize(new_rows);
*/
  mat.resize(new_rows);
  for (unsigned i=0; i<new_rows; ++i) mat[i][0] = rhs[i];
  rows = new_rows;
  cols = new_cols;

  return *this;
}

// Assignment from 1D array
template<typename T>
MLMatrix<T>& MLMatrix<T>::operator=(const T rhs[]) {
  unsigned new_rows = sizeof(rhs) / sizeof(T);
  unsigned new_cols = 1;

  mat.resize(new_rows);
  for (unsigned i=0; i<new_rows; ++i) mat[i][0] = rhs[i];
  rows = new_rows;
  cols = new_cols;

  return *this;
}


/***************************************************************************************************
  Operators
*/
// Addition of two matrices
template<typename T>
MLMatrix<T> MLMatrix<T>::operator+(const MLMatrix<T>& rhs) const {
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Addition error: dimensions do not match (%d, %d)+(%d, %d)", cols, rows, rhs.get_cols(), rhs.get_rows());
    while(1);
  }

  MLMatrix result(rows, cols, 0);

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      result(i,j) = this->mat[i][j] + rhs(i,j);
    }
  }
  return result;
}

// Cumulative addition of this matrix and another
template<typename T>
MLMatrix<T>& MLMatrix<T>::operator+=(const MLMatrix<T>& rhs) {
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Addition error: dimensions do not match (%d, %d)+(%d, %d)", cols, rows, rhs.get_cols(), rhs.get_rows());
    while(1);
  }

  unsigned rows = rhs.get_rows();
  unsigned cols = rhs.get_cols();

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      this->mat[i][j] += rhs(i,j);
    }
  }
  return *this;
}

// Subtraction of this matrix and another
template<typename T>
MLMatrix<T> MLMatrix<T>::operator-(const MLMatrix<T>& rhs) const {
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Substraction error: dimensions do not match (%d, %d)-(%d, %d)", rows, cols, rhs.get_rows(), rhs.get_cols());
    while(1);
  }

  unsigned rows = rhs.get_rows();
  unsigned cols = rhs.get_cols();
  MLMatrix result(rows, cols, 0);

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      result(i,j) = this->mat[i][j] - rhs(i,j);
      // result(i,j) = mat[i][j] - rhs(i,j);
    }
  }
  return result;
}

// Cumulative subtraction of this matrix and another
template<typename T>
MLMatrix<T>& MLMatrix<T>::operator-=(const MLMatrix<T>& rhs) {
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Substraction error: dimensions do not match (%d, %d)-(%d, %d)", rows, cols, rhs.get_rows(), rhs.get_cols());
    while(1);
  }

  unsigned rows = rhs.get_rows();
  unsigned cols = rhs.get_cols();

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      this->mat[i][j] -= rhs(i,j);
    }
  }
  return *this;
}

// Left multiplication of this matrix and another
template<typename T>
MLMatrix<T> MLMatrix<T>::operator*(const MLMatrix<T>& rhs) const {
  if (cols != rhs.get_rows()) {
    Serial.printf("Multiplication error: dimensions do not match (%d, %d).(%d, %d)", rows, cols, rhs.get_rows(), rhs.get_cols());
    while(1);
  }

  // unsigned rows = rhs.get_rows();
  unsigned cols = rhs.get_cols();
  MLMatrix result(rows, cols, 0);

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      for (unsigned k=0; k<rhs.get_rows(); k++) {
        result(i,j) += this->mat[i][k] * rhs(k,j);
      }
    }
  }
  return result;
}

// Cumulative left multiplication of this matrix and another
template<typename T>
MLMatrix<T>& MLMatrix<T>::operator*=(const MLMatrix<T>& rhs) {
  if (cols != rhs.get_rows()) {
    Serial.printf("Multiplication error: dimensions do not match (%d, %d).(%d, %d)", rows, cols, rhs.get_rows(), rhs.get_cols());
    while(1);
  }

  MLMatrix result = (*this) * rhs;
  (*this) = result;
  return *this;
}

// Calculate a transpose of this matrix
template<typename T>
MLMatrix<T> MLMatrix<T>::transpose() {
  MLMatrix result(cols, rows, 0);

  for (unsigned i=0; i<cols; ++i) {
    for (unsigned j=0; j<rows; ++j) {
      result(i,j) = this->mat[j][i];
    }
  }
  return result;
}

// Matrix/scalar addition
template<typename T>
MLMatrix<T> MLMatrix<T>::operator+(const T& rhs) {
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Addition error: dimensions do not match (%d, %d)+(%d, %d)", cols, rows, rhs.get_cols(), rhs.get_rows());
    while(1);
  }
  MLMatrix result(rows, cols, 0);

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      result(i,j) = this->mat[i][j] + rhs;
    }
  }
  return result;
}

// Matrix/scalar addition
template<typename T>
MLMatrix<T>& MLMatrix<T>::operator+=(const T& rhs) {
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Addition error: dimensions do not match (%d, %d)+(%d, %d)", cols, rows, rhs.get_cols(), rhs.get_rows());
    while(1);
  }

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      this->mat[i][j] += rhs;
    }
  }
  return *this;
}

// Matrix/scalar substraction
template<typename T>
MLMatrix<T> MLMatrix<T>::operator-(const T& rhs) {
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Substraction error: dimensions do not match (%d, %d)-(%d, %d)", cols, rows, rhs.get_cols(), rhs.get_rows());
    while(1);
  }

  MLMatrix result(rows, cols, 0);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      result(i,j) = this->mat[i][j] - rhs;
    }
  }
  return result;
}

// Matrix/scalar substraction
template<typename T>
MLMatrix<T>& MLMatrix<T>::operator-=(const T& rhs) {
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Substraction error: dimensions do not match (%d, %d)-(%d, %d)", cols, rows, rhs.get_cols(), rhs.get_rows());
    while(1);
  }

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      this->mat[i][j] -= rhs;
    }
  }
  return *this;
}

// Matrix/scalar multiplication
template<typename T>
MLMatrix<T> MLMatrix<T>::operator*(const T& rhs) {
  MLMatrix result(rows, cols, 0);

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      result(i,j) = this->mat[i][j] * rhs;
    }
  }
  return result;
}

// Matrix/scalar multiplication
template<typename T>
MLMatrix<T>& MLMatrix<T>::operator*=(const T& rhs) {
  // MLMatrix result(rows, cols, 0);

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      this->mat[i][j] *= rhs;
    }
  }
  return *this;
}

// Matrix/scalar division
template<typename T>
MLMatrix<T> MLMatrix<T>::operator/(const T& rhs) {
  if (rhs == 0) {
    Serial.printf("Division by 0 error");
    while(1);
  }

  MLMatrix result(rows, cols, 0);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      result(i,j) = this->mat[i][j] / rhs;
    }
  }
  return result;
}

// Matrix/scalar division
template<typename T>
MLMatrix<T>& MLMatrix<T>::operator/=(const T& rhs) {
  if (rhs == 0) {
    Serial.printf("Division by 0 error");
    while(1);
  }

  // MLMatrix<T> result(rows, cols, 0);

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      this->mat[i][j] /= rhs;
    }
  }
  return *this;
}

// Multiply a matrix with a vector
template<typename T>
std::vector<T> MLMatrix<T>::operator*(const std::vector<T>& rhs) {
  if (cols != rhs.size()) {
    Serial.printf("Multiplication error: dimensions do not match (%d, %d).%d", rows, cols, rhs.size());
    while(1);
  }

  std::vector<T> result(rhs.size(), 0);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      result[i] += this->mat[i][j] * rhs[j];
    }
  }
  return result;
}

/***************************************************************************************************
  Accessors
*/
// Access the individual elements
template<typename T>
T& MLMatrix<T>::operator()(const unsigned& row, const unsigned& col) {
  return this->mat[row][col];
}

// Access the individual elements (const)                                                                                                                                     
template<typename T>
const T& MLMatrix<T>::operator()(const unsigned& row, const unsigned& col) const {
  return this->mat[row][col];
}

/***************************************************************************************************
  Comparison operators
*/
// Determine if two arrays are equal and return true, otherwise return false.
template <typename T>
const bool MLMatrix<T>::operator==( const MLMatrix<T> &rhs ) const
{
  if ( rows != rhs.rows || cols != rhs.cols ) return false; // matrices of different sizes

  for ( unsigned i = 0; i < rows; ++i )
    for ( unsigned j = 0; j < cols; ++j )
      if ( this->mat[i][i] != rhs.mat[i][i] ) return false; 
  return true; // matrices are equal
}

template <typename T>
const bool MLMatrix<T>::operator!=( const MLMatrix<T> &rhs ) const
{
  return !(*this == rhs);
}

// Compare 2 matrices element wise
template <typename T>
MLMatrix<T> MLMatrix<T>::operator<( const MLMatrix<T> &rhs )
{
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Comparison error: dimensions do not match (%d, %d)<(%d, %d)", cols, rows, rhs.get_cols(), rhs.get_rows());
    while(1);
  }
  MLMatrix result(rows, cols, 0);
  for ( unsigned i = 0; i < rows; ++i )
    for ( unsigned j = 0; j < cols; ++j )
      result(i,j) = T(mat[i][j] < rhs.mat[i][j] ? true : false); 
  return result;
}

template <typename T>
MLMatrix<T> MLMatrix<T>::operator>=( const MLMatrix<T> &rhs )
{
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Comparison error: dimensions do not match (%d, %d)>=(%d, %d)", cols, rows, rhs.get_cols(), rhs.get_rows());
    while(1);
  }
  MLMatrix result(rows, cols, 0);
  for ( unsigned i = 0; i < rows; ++i )
    for ( unsigned j = 0; j < cols; ++j )
      result(i,j) = T(mat[i][j] >= rhs.mat[i][j] ? true : false); 
  return result;
}


/***************************************************************************************************
  Misc methods
*/
// Get the number of rows of the matrix
template<typename T>
unsigned MLMatrix<T>::get_rows() const { return this->rows; }

// Get the number of columns of the matrix                                                                                                                                    
template<typename T>
unsigned MLMatrix<T>::get_cols() const { return this->cols; }

// Display the matrix
// usage: mat.print();
template <typename T>
void MLMatrix<T>::print()
{
  Serial.printf("%d rows, %d cols\n",rows, cols);
  for (int i = 0; i < rows; ++i) {
    if (std::is_floating_point<T>::value) { // float
      for (int j = 0; j < cols - 1; ++j) Serial.printf("%8.3f, ", this->mat[i][j]);
        Serial.printf("%8.3f\n", this->mat[i][cols - 1]);
    }  else { // integer
      for (int j = 0; j < cols - 1; ++j) Serial.printf("%5d, ", this->mat[i][j]);
      Serial.printf("%5d\n", this->mat[i][cols - 1]);
    }  
  }
}

// Display the matrix size
// usage: mat.printSize();
template <typename T>
void MLMatrix<T>::printSize()
{
  Serial.printf("(%d, %d)",rows, cols);
}

// Find the place of the minimum value
template <typename T>
void MLMatrix<T>::indexMin(int &indexRow, int &indexCol)
{
  T minVal = this->mat[0][0];
    for (int i = 0; i < rows; ++i) 
      for (int j = 0; j < cols; ++j)
        if (this->mat[i][j] < minVal) {
          minVal = this->mat[i][j];
          indexRow = i;
          indexCol = j;
      }
}


// Find the place of the maximum value
template <typename T>
void MLMatrix<T>::indexMax(int &indexRow, int &indexCol)
{
  indexRow = 0;
  indexCol = 0;
  T maxVal = this->mat[0][0];
  for (int i = 0; i < rows; ++i) 
    for (int j = 0; j < cols; ++j)
      if (this->mat[i][j] > maxVal) {
        maxVal = this->mat[i][j];
        indexRow = i;
        indexCol = j;
      }
}

// Obtain a vector of the diagonal elements
// Usage :  std::vector<int> Diag = mat.diag_vec();
template<typename T>
std::vector<T> MLMatrix<T>::diag_vec() {
  std::vector<T> result(rows, 0);

  for (unsigned i=0; i<rows; ++i) {
    result[i] = this->mat[i][i];
  }
  return result;
}

// Vector dot product, using matrices
/*
Example usage:
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {4, 5, 6};
  MLMatrix<int> mv1(3, 1, 0);
  mv1 = v1;
  MLMatrix<int> mv2(3, 1, 0);
  mv2 = v2;
  int P = mv1.MdotProd(mv2, true);
*/
template<typename T>
T MLMatrix<T>::MdotProd(const MLMatrix<T>& rhs, bool clip)
{
    if (rhs.cols !=1 || cols != 1) {
      Serial.printf("Dot product error: please use matrices with 1 column\n");
      while(1);
    }
    if (rhs.rows != rows) {
        Serial.printf("Dot product error: dimensions do not match (%d, %d)\n",rhs.rows, rows);
        while(1);
    }
    float sum = 0.0f;
    for (unsigned i=0; i<rows; ++i) sum += mat[i][0] * rhs.mat[i][0];

    if (clip) { // Clip the result at min and max value of type T
        // float sum = std::inner_product(std::begin(a.mat), std::end(a.mat), std::begin(mat), 0.0);;
        if (sum < std::numeric_limits<T>::min()) sum = std::numeric_limits<T>::min();
        if (sum > std::numeric_limits<T>::max()) sum = std::numeric_limits<T>::max();
    }
    return T(sum);
}

/* Faster matrice multiplication
Example usage :

  MLMatrix<int> a(10, 50, 0, 10);
  MLMatrix<int> b(50, 10, 0, 10);
Standard multiplication:
  MLMatrix<int> c = a * b;
Fast multiplication:
  c = a.times(b.transpose());

*/
template<typename T>
MLMatrix<T> MLMatrix<T>::times(const MLMatrix<T>& rhs, bool clip)
{
  if (cols != rhs.get_cols()) {
    Serial.printf("Multiplication error: dimensions do not match (%d, %d).(%d, %d)", rows, cols, rhs.get_rows(), rhs.get_cols());
    while(1);
  }

  unsigned rows2 = rhs.get_rows();
  MLMatrix result(rows, rows2, 0);

  if (! clip) {
    for (unsigned i=0; i<rows; ++i) {
      for (unsigned j=0; j<rows2; ++j) {
        result(i,j) = std::inner_product(std::begin(mat[i]), std::end(mat[i]), std::begin(rhs.mat[j]), 0);
      }
    }
  }else {
    for (unsigned i=0; i<rows; ++i) {
      for (unsigned j=0; j<rows2; ++j) {
        float R = std::inner_product(std::begin(mat[i]), std::end(mat[i]), std::begin(rhs.mat[j]), 0);
        if (R < std::numeric_limits<T>::min()) R = std::numeric_limits<T>::min();
        if (R > std::numeric_limits<T>::max()) R = std::numeric_limits<T>::max();
        result(i,j) = T(R);
      }
    }
  }
  return result;
}

/* 
  Hadamard (element-wise) product
  Usage
    MLMatrix<int> a(10, 50, 0, 10); // define the first matrix
    MLMatrix<int> b(10, 50, 0, 10); // define the second matrix, same dimensions
    a = a.Hadamard(b);

*/
template<typename T>
MLMatrix<T> MLMatrix<T>::Hadamard(const MLMatrix<T>& rhs, bool clip)
{
  if ( rows != rhs.rows || cols != rhs.cols ) { // matrices of different sizes
    Serial.printf("Hadamard product error: dimensions do not match (%d, %d).(%d, %d)", rows, cols, rhs.get_rows(), rhs.get_cols());
    while(1);
  }
  MLMatrix result(rows, cols, 0);

  if (!clip) {
    for ( unsigned i = 0; i < rows; ++i )
      for ( unsigned j = 0; j < cols; ++j )
        result(i,j) = mat[i][j] * rhs.mat[i][j];
  } else {
    for ( unsigned i = 0; i < rows; ++i )
      for ( unsigned j = 0; j < cols; ++j ) {
        float R = mat[i][j] * rhs.mat[i][j];
        if (R < std::numeric_limits<T>::min()) R = std::numeric_limits<T>::min();
        if (R > std::numeric_limits<T>::max()) R = std::numeric_limits<T>::max();
        result(i,j) = T(R);       
      }
  }
  return result;
}

/*
  Apply a given function to the elements of a matrix (element-wise)
  The function must be written as: 
    T function(T x) { ... }
  For example :
    int plusOne (int x) { return x+1; }
  
  Usage:
    MLMatrix<int> a(10, 50, 0, 10); // define the first matrix
    a.applySelf( &function ); // changes the matrix
    MLMatrix<int> b = a.apply( &function ); // does not change the matrix
*/
template<typename T>
MLMatrix<T> MLMatrix<T>::applySelf(T (*function)(T))
{
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      mat[i][j] = function(mat[i][j]);
    }
  }
  return *this;
}

template<typename T>
MLMatrix<T> MLMatrix<T>::apply(T (*function)(T)) 
{
  MLMatrix result(rows, cols, 0);

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      result(i,j) = function(this->mat[i][j]);
    }
  }
  return result;
}


/* Various normS of a matrix
  Usage:
    MLMatrix<float> a(10, 50, 0.0f, 10.0f);
    float normL0 = a.L0Norm();
    float normL1 = a.L1Norm();
    float normL2 = a.L2Norm();
*/
template<typename T>
T MLMatrix<T>::L2Norm()
{
  T L2 = T(0);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      L2 += pow(this->mat[i][j], 2);
    }
  }
  return sqrt(L2);
}

template<typename T>
T MLMatrix<T>::L1Norm()
{
  T L1 = T(0);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      float L = abs(this->mat[i][j]);
      L1 = (L > L1)? L: L1;
    }
  }
  return L1;
}

template<typename T>
int MLMatrix<T>::L0Norm() // number of non zero elements
{
  int L0 = 0;
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (this->mat[i][j] != 0) ++L0;
    }
  }
  return L0;
}

// Max, min, mean, std deviation
template<typename T>
T MLMatrix<T>::max() const
{
  T max = std::numeric_limits<T>::min();
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (this->mat[i][j] > max) max = this->mat[i][j];
    }
  }
  return max;
}

template<typename T>
T MLMatrix<T>::min() const
{
  T min = std::numeric_limits<T>::max();
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (this->mat[i][j] < min) min = this->mat[i][j];
    }
  }
  return min;
}

template<typename T>
float MLMatrix<T>::mean() const
{
  float mean = 0.0f;
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      mean += this->mat[i][j];
    }
  }
  mean /= float(rows * cols);
  return mean;
}

template<typename T>
float MLMatrix<T>::stdev(const float mean) const
{
  float stdev = 0.0f;
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      stdev += pow(this->mat[i][j] - mean, 2);
    }
  }
  stdev /= float(rows * cols);
  stdev = sqrt(stdev);
  return stdev;
}

// Scale the norm to a given value
// usage: Y = X.normScale(val, zeroNorm);
template <typename T>
MLMatrix<T> MLMatrix<T>::normScale (float value, bool & zeroNorm)
{
  zeroNorm = false;
  value = abs(value);
  MLMatrix result(rows, cols, 0);
  for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j) {
      result(i,j) = this->mat[i][j];
    }
  float L2 = result.L2Norm();
  if (L2 != 0.0f) result = result / L2 * value;
  else zeroNorm = true;
  return result;
}

// Scale the norm to a given value
// usage: X.normScale2(val, zeroNorm);
template <typename T>
void MLMatrix<T>::normScale2 (float value, bool & zeroNorm)
{
  zeroNorm = false;
  value = abs(value);
  float L2 = this->L2Norm();
  if (L2 == 0.0f) zeroNorm = true;
  else {
    float coef = value / L2;
    for (unsigned i=0; i<rows; ++i)
      for (unsigned j=0; j<cols; ++j)
        mat[i][j] *= coef;
  }
}

// Clip all values less than threshold to zero
// Leads to:  |abs(value)| > threshold or zero
template <typename T>
int MLMatrix<T>::clipToZero (float threshold)
{
  int nbClip = 0;
  threshold = abs(threshold);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (abs(mat[i][j]) <= threshold) mat[i][j] = 0.0f;
      ++nbClip;
    }
  }
  return nbClip;
}

// Set all values less than threshold to threshold
template <typename T>
int MLMatrix<T>::clipMin (float threshold)
{
  int nbClip = 0;
  threshold = abs(threshold);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (abs(mat[i][j]) < threshold && mat[i][j] >= 0) mat[i][j] = threshold;
      if (abs(mat[i][j]) < threshold && mat[i][j] < 0)  mat[i][j] = -threshold;
      ++nbClip;
    }
  }
  return nbClip;
}

// Set all values greater than threshold to threshold
// Leads to:     -threshold < value < threshold
template <typename T>
int MLMatrix<T>::clipMax (float threshold)
{
  int nbClip = 0;
  threshold = abs(threshold);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (mat[i][j] >  threshold) mat[i][j] =  threshold;
      if (mat[i][j] < -threshold) mat[i][j] = -threshold;
      ++nbClip;
    }
  }
  return nbClip;
}

// Create a matrix with the sign (+1 or -1) of each element of an input matrix
template<typename T>
MLMatrix<T> MLMatrix<T>::sgn()
{
  MLMatrix<T> S(rows, cols, 0);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      S(i,j) = (0 > mat[i][j]) ? T(-1) : T(1);
    }
  }
  return S;
}

// Extract a row or a column from a matrix
template<typename T>
MLMatrix<T> MLMatrix<T>::row(const uint32_t rowNumber)
{
  MLMatrix<T> result(1, cols, 0);
  if (rowNumber > rows) { 
    Serial.printf("Row extraction error: row %d greater than %d\n", rowNumber, rows);
    while(1);
  }
  for (unsigned j=0; j<cols; ++j) result(j, 0) = mat(rowNumber, j);
  return result;
}

template<typename T>
MLMatrix<T> MLMatrix<T>::col(const uint32_t colNumber)
{
  MLMatrix<T> result(rows, 1, 0);
  if (colNumber > cols) { 
    Serial.printf("Column extraction error: col %d greater than %d\n", colNumber, cols);
    while(1);
  }
  for (unsigned i=0; i<rows; ++i) result(i, 0) = mat(i, colNumber);
  return result;
}

/* Extract a submatrix
      row0 <= row < row0 + nrows
      col0 <= col < col0 + ncols
*/
template<typename T>
MLMatrix<T> MLMatrix<T>::subMatrix(const uint32_t row0, const uint32_t nrows, const uint32_t col0, const uint32_t ncols)
{
  if (row0 + nrows > rows || col0 + ncols > cols) {
    Serial.printf ("Submatrix extraction error (rows from %d to %d, cols from %d to %d)", row0, row0+nrows, col0, col0+ncols);
    while(1);
  }
  MLMatrix<T> result(nrows, ncols, 0);
  for (unsigned i=0; i<nrows; ++i) 
    for (unsigned j=0; j<ncols; ++j) 
      result(i, j) = mat[row0 + i][col0 + j];
  return result;
}

// Apply a random change to all elements of a matrix
template<typename T>
MLMatrix<T> MLMatrix<T>::randomChange(const float amplitude)
{
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      // random number between -1 and +1
      float rand = float(random(10000)) / 10000.0f * 2.0f - 1.0f;
      mat[i][j] = mat[i][j] * (1.0f + rand * amplitude);
    }
  }
  return *this;
}

/*  Generate a random matrix with normal distribution
    using the polar form of Box Muller algorithm
    usage:    MLMatrix<float> u(30, 30, 0.0f);
              u.randomNormal(0.0f,1.0f);
*/
template<typename T>
MLMatrix<T> MLMatrix<T>::randomNormal(const float mean, float std_dev)
{
  std_dev = abs(std_dev);
  int dim = rows * cols;
  float eps = 0.001f;
  MLMatrix<T> N(rows, cols, 0);
  MLMatrix<T> C(dim, 1, 0.0f);
  for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j) {
      float r = 1.0f;
      float u;
      do {
        u = 2 * float(random(100000)) / 100000.0f - 1;
        float v = 2 * float(random(100000)) / 100000.0f - 1;
        r = u * u + v * v;
      } while (r > 1.0f && r != 0.0f);
      mat[i][j] = u * sqrt(-2.0f * log(r) / r);
      mat[i][j] = mat[i][j] * std_dev - mean;   
    }
  return *this;
}

#endif