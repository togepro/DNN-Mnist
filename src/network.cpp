#include "network.h"
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

Network::Network(int* sizes, int num){
    this->numLayers = num;
    this->sizes = new int[num];
    memcpy(this->sizes,sizes,num*sizeof(int));
    biases = new double*[num-1];
    weights = new double*[num-1];

    nablaB = new double*[num-1];
    nablaW = new double*[num-1];
    deltaNablaB = new double*[num-1];
    deltaNablaW = new double*[num-1];

    zs = new double*[num-1];
    activations = new double*[num];

    for(int i=0; i<num; i++){
        activations[i] = new double[sizes[i]];
    }

    for(int i=1; i<num; i++){
	biases[i-1] = new double[sizes[i]];
	weights[i-1] = new double[sizes[i-1]*sizes[i]];

        nablaB[i-1] = new double[sizes[i]];
        nablaW[i-1] = new double[sizes[i-1]*sizes[i]];
        deltaNablaB[i-1] = new double[sizes[i]];
        deltaNablaW[i-1] = new double[sizes[i-1]*sizes[i]];

        zs[i-1] = new double[sizes[i]];

        fill_n(deltaNablaB[i-1], sizes[i], 0);
        fill_n(deltaNablaW[i-1], sizes[i]*sizes[i-1], 0);
        fill_n(nablaB[i-1], sizes[i], 0.0);
        fill_n(nablaW[i-1], sizes[i]*sizes[i-1], 0);

	normaldist(biases[i-1], sizes[i]);
	normaldist(weights[i-1], sizes[i-1]*sizes[i]);
	}
}

Network::Network(){}

Network::~Network(){
    for(int i=0; i<numLayers-1; i++){	
	delete biases[i];
	delete weights[i];
	delete nablaB[i];
        delete nablaW[i];
        delete deltaNablaB[i];
        delete deltaNablaW[i];
        delete zs[i];
        delete activations[i];
    }
    delete activations[numLayers-1];
    delete biases;
    delete weights;
    delete nablaB;
    delete nablaW;
    delete deltaNablaB;
    delete deltaNablaW;
}

/* 前向算法 */
void Network::feedForward(double* a){
    for(int i=0;i<sizes[0];i++){
        activations[0][i] = a[i];
    }
    for(int i = 0; i<numLayers-1; i++){
	for(int j=0;j<sizes[i+1];j++){
	    double sum = 0;
	    int index = j*sizes[i];
	    for(int k=0;k<sizes[i];k++){
		sum += weights[i][index+k]*activations[i][k];
	    }
	    activations[i+1][j] = 1.0/(1.0+exp(-1.0*(sum+biases[i][j])));
	}
	
    }
}

/* 异步随机梯度下降算法 */
void Network::SGD(int epochs, int miniBatchSize, double eta, Image* trainData, Image* testData,int trainCount, int testCount){
    for(int i = 0; i < epochs; i++){
	random_shuffle(&trainData[0], &trainData[49999]);

        updateMiniBatch(trainData, miniBatchSize, eta);
        cout<<"Epoch "<<i<<" "<<evaluate(testData)<<"/"<<testCount<<endl;
    }
}

/* 更新权重及偏置 */
void Network::updateMiniBatch(Image* trainData, int miniBatchSize, double eta){
    int batchCount = TRAINNUM / miniBatchSize;

    for(int i = 0; i<batchCount; i++){
        for(int j=0; j<numLayers-1; j++){
            fill_n(nablaB[j], sizes[j+1], 0);
            fill_n(nablaW[j], sizes[j]*sizes[j+1], 0);
        }
        int index = i * miniBatchSize;
        for(int j = 0; j < miniBatchSize; j++){

            backProp(trainData[index+j].pixel, trainData[index+j].label);
            for(int k=0; k<numLayers-1; k++){
                for(int l=0; l<sizes[k+1];l++){
                    nablaB[k][l] += deltaNablaB[k][l];
                }
                int length = sizes[k+1]*sizes[k];
                for(int l=0; l<length;l++){
                    nablaW[k][l] += deltaNablaW[k][l];
                }
            }
        }
        for(int k=0; k<numLayers-1; k++){
            for(int l=0; l<sizes[k+1];l++){
                biases[k][l] -= eta/miniBatchSize*nablaB[k][l];
            }
            int length = sizes[k+1]*sizes[k];
            for(int l=0; l<length;l++){
                weights[k][l] -= eta/miniBatchSize*nablaW[k][l];
            }
        }
    }
}

/* 反向传播算法 */
void Network::backProp(double *x, int y){
    for(int k=0; k<numLayers-1; k++){
        fill_n(deltaNablaB[k], sizes[k+1], 0);
        fill_n(deltaNablaW[k], sizes[k]*sizes[k+1], 0);
    }
    
    memcpy(activations[0], x, sizes[0]*sizeof(double));
    for(int i = 0; i<numLayers-1; i++){
	dotAdd(weights[i], activations[i], biases[i],zs[i],sizes[i+1], sizes[i]);
	sigmoid(zs[i],activations[i+1],sizes[i+1]);
    }
    costDerivative(activations[numLayers-1], y, deltaNablaB[numLayers-2], sizes[numLayers-1]);
    MulSigmoidPrime(zs[numLayers-2], deltaNablaB[numLayers-2], sizes[numLayers-1]);
    //dot
    for(int i = 0; i<sizes[numLayers-1]; i++){
        for(int j = 0;j<sizes[numLayers-2];j++){
            deltaNablaW[numLayers-2][i*sizes[numLayers-2] + j] = deltaNablaB[numLayers-2][i] * activations[numLayers-2][j];
        }
    }

    for(int i=numLayers-3; i>=0; i--){
        for(int k=0;k<sizes[i+1];k++){
            double sum = 0;
            for(int j=0;j<sizes[i+2];j++){
                sum += weights[i+1][k+sizes[i+1]*j] * deltaNablaB[i+1][j];
            }
            deltaNablaB[i][k] = sum;
        }

        MulSigmoidPrime(zs[i],deltaNablaB[i],sizes[i+1]);
        for(int j = 0; j<sizes[i+1]; j++){
            for(int k = 0;k<sizes[i];k++){
                deltaNablaW[i][j*sizes[i]+k] = deltaNablaB[i][j] * activations[i][k];
            }
        }
    }
}

int Network::evaluate(Image* testData){
    int count = 0;
    for(int i=0 ; i<TESTNUM; i++){
        feedForward(testData[i].pixel);
        int index = getMaxIndex(activations[numLayers-1],sizes[numLayers-1]);
        if(testData[i].label == index){
            count++;
        }
    }
    return count;
}

void Network::costDerivative(double* a, int y, double* res, size_t n){
    memcpy(res, a, sizeof(double)*n);
    res[y] = a[y] - 1.0;
}
