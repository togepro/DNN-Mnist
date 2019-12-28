#ifndef MNIST_H
#define MNIST_H

#include <iostream>
#include <fstream>
#include <string>
#include "util.h"

using namespace std;

class Mnist{
    public:
        Mnist();
        Mnist(const string& trainImg, const string& trainLabel);
        ~Mnist();
        void loadData();
        Image* getTrainData();
        Image* getTestData();
    private:
        string trainImg;
        string trainLabel;
        Image* trainingData;
        Image* testData;
};


#endif
