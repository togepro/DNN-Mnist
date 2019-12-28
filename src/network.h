#ifndef NETWORK_H
#define NETWORK_H
#include "util.h"

class Network{
	public:
		Network();					//默认构造函数
		Network(int* sizes, int num);					//带参构造函数
		~Network();					//析构函数
		void feedForward(double a[]);	//前向传播算法
		void SGD(int epochs, int miniBatchSize, double eta, Image* trainData, Image* testData, int trainCount=TRAINNUM, int testCount = TESTNUM);					//小批量随机梯度下降算法
		void updateMiniBatch(Image* trainData, int miniBatchSize, double eta);		//
		void backProp(double *x, int y);			//反向传播算法
		int evaluate(Image* testData);			//
		void costDerivative(double* a, int y, double* res, size_t n);	//

	private:
		int numLayers; 		//整型，神经网络层数
		int* sizes;			//整型数组，神经网络每层神经元个数
		double** biases;	//双精度二维数组，每个神经元上的偏置
		double** weights;	//双精度二维数组，权重（每层一个二维数组）
        double** nablaB;
        double** nablaW;
        double** deltaNablaB;
        double** deltaNablaW;
        double** zs;
        double** activations;
};

#endif
