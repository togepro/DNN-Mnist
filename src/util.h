#ifndef UTIL_H
#define UTIL_H

#include <cstddef>
#include <inttypes.h>
#include <cstring>

#define TRAINNUM 50000
#define TESTNUM 10000
#define IMGSIZE 784

using namespace std;

void sigmoid(double* z, double* res, size_t n);
void sigmoidPrime(double* z, size_t n);
void MulSigmoidPrime(double* z, double* a, size_t n);
double* dot(double* w, double* a, int row, int col);
void dotAdd(double* w, double* a, double* b, double* res, int row, int col);
double* dotMul(double* w, double* a, int row, int col);
void normaldist(double* data, size_t n);
uint32_t swapEndian(uint32_t number);
int getMaxIndex(double* a, size_t n);
struct Image
{
    double pixel[IMGSIZE];
    int label;
};



#endif
