#include <iostream>
#include "network.h"
#include "mnist.h"

using namespace std;

int main(){
    Mnist mnist("../data/trainImg","../data/trainLabel");
    mnist.loadData();
    double eta = 3.0;
    int sizes[3]={784,30,10};
    int epochs = 30;
    int miniBatchSize = 10;
    Network net(sizes,3);

//    Image* data = mnist.getTrainData();
//    for(int i =0;i<5000;i++)
//    	net.feedForward(data[i].pixel);
    net.SGD(epochs, miniBatchSize, eta, mnist.getTrainData(), mnist.getTestData());
//	double* res=net.feedForward(a);
//	cout<<res[0]<<endl;
    return 0;
}
