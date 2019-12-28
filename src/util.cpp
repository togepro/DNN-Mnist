#include <cmath>
#include <iostream>
#include <random>
#include "util.h"
using namespace std;

// Miscellaneous functions
//The sigmoid function.
void sigmoid(double* z, double* res, size_t n){
	for(int i=0; i<n; i++){
		res[i] = 1.0/(1.0+exp(-1*z[i]));
	}
}

//Derivative of the sigmoid function.
void sigmoidPrime(double*z, size_t n){
	for(int i=0; i<n;i++){
	  int res = 1.0/(1.0+exp(-1*z[i]));
	  z[i] = res*(1-res);
	}
}

void MulSigmoidPrime(double*z, double* a, size_t n){
    double res;
	for(int i=0; i<n;i++){
	  res = 1.0/(1.0+exp(-1*z[i]));
	  a[i] = res*(1-res)*a[i];
	}
}
// The dot function between matrix and vector
double* dot(double* w, double* a, int row, int col){
	double* res = new double[row];
	for(int i=0; i<row; i++){
		double sum = 0;
		for(int j=0; j<col; j++){
			sum += w[i*col+j]*a[j];
		}
		res[i] = sum;
	}
	return res;
}

void dotAdd(double* w, double* a, double* b, double* res, int row, int col){
	for(int i=0; i<row; i++){
		double sum = 0;
		for(int j=0; j<col; j++){
			sum += w[i*col+j]*a[j];
		}
		res[i] = sum + b[i];
	}
}



void normaldist(double *data, size_t n){
    normal_distribution<double> dis(0, 1);
    default_random_engine random(time(NULL)); 
	for(int i=0; i<n; i++){
		data[i] = dis(random);
	}
}

uint32_t swapEndian(uint32_t number){
    //uint32_t val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	//return (val << 16) | (val >> 16);
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = number & 255;
    ch2 = (number >> 8) & 255;
    ch3 = (number >> 16) & 255;
    ch4 = (number >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

int getMaxIndex(double* a, size_t n){
    if(n < 1){
        return -1;
    }
    int index = 0;
    double maxNumber = a[0];
    for(int i =1; i < n; i++){
        if(a[i] > maxNumber){
            index = i;
            maxNumber = a[i];
        }
    }
    return index;
}
