#include "mnist.h"
#include "util.h"
#include <string.h>
Mnist::Mnist(const string& trainImg, const string& trainLabel){
    this->trainImg = trainImg;
    this->trainLabel = trainLabel;
    trainingData = new Image[TRAINNUM];
    testData = new Image[TESTNUM];
}

Mnist::~Mnist(){
    delete trainingData;
    delete testData;
}

void Mnist::loadData(){
	ifstream image(trainImg.c_str(), ios::in | ios::binary);
	ifstream label(trainLabel.c_str(), ios::in | ios::binary);
	if (!image.is_open())
	{
		cout << "open mnist image file error!" << endl;
		return;
	}
	if (!label.is_open())
	{
		cout << "open mnist label file error!" << endl;
		return;
	}
	uint32_t magic;     //magic number
	uint32_t numItems; //mnist图像集文件中的图像数目
	uint32_t numLabel; //mnist标签集文件中的标签数目
	uint32_t rows;
	uint32_t cols;
	image.read(reinterpret_cast<char*>(&magic), 4);
	magic = swapEndian(magic);
	if (magic != 2051)
	{
		cout<< " this is not the mnist image file" << endl;
		return;
	}
	label.read(reinterpret_cast<char*>(&magic), 4);
	magic = swapEndian(magic);
	if (magic != 2049)
	{
		cout << "this is not the mnist label file" << endl;
		return;
	}
 	image.read(reinterpret_cast<char*>(&numItems), 4);
	numItems = swapEndian(numItems);
	label.read(reinterpret_cast<char*>(&numLabel), 4);
	numLabel = swapEndian(numLabel);
	if (numItems != numLabel)
	{
		cout << "the image file and label file are not a pair" << endl;
        return;
	}
    image.read(reinterpret_cast<char*>(&rows), 4);
	rows = swapEndian(rows);
	image.read(reinterpret_cast<char*>(&cols), 4);
	cols = swapEndian(cols);

    int total = rows * cols;
    if(total != IMGSIZE){
        cout << "the image size is not corret" << endl;
        return;
    }
	unsigned char* pixels = new unsigned char[numItems * total];
    unsigned char* labels = new unsigned char[numItems];

    image.read((char*)pixels, numItems * total);
    label.read((char*)labels, numItems);

    for(int i = 0; i < TRAINNUM; i++){
        int index = i * total;
        for(int j = 0; j < total; j++){
            trainingData[i].pixel[j] = (double)pixels[index+j] / 255.0;
        }
    }
    for(int i = 0; i<TESTNUM; i++){
        int index =(TRAINNUM + i) * total;
        for(int j = 0; j < total; j++){
            testData[i].pixel[j] = (double)pixels[index + j] / 255.0;
        }
    }

    for(int i=0; i<TRAINNUM; i++){
        trainingData[i].label = (int)labels[i];
    }
    for(int i=0; i<TESTNUM; i++){
        testData[i].label =  (int)labels[i+TRAINNUM];
    }
    delete pixels;
    delete labels;
}

Image* Mnist::getTrainData(){
    return trainingData;
}

Image* Mnist::getTestData(){
    return testData;
}
