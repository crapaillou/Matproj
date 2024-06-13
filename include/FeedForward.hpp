#pragma once
#include <vector>
#include <functional>
#include <string>
#include "opencv2/opencv.hpp"

using namespace std;

class convlayer{
    public:
    vector<size_t> shapeKernel; //size of the kernel and number of kernels.
    vector<vector<cv::Mat>> veckernels;
    vector<cv::Mat> vecbias;
    vector<vector<cv::Mat>> ZTensor;
    vector<vector<cv::Mat>> ATensor;
    vector<vector<cv::Mat>> NablaTensor;
    vector<vector<cv::Mat>> PoolBuffer;
    function<void(vector<cv::Mat>, vector<vector<cv::Mat>>, size_t)> PoolFunc;

    
    convlayer(size_t inputsize, size_t inputdepth , vector<size_t> shapeKernel , string layerType, string Pooling, function<void(vector<cv::Mat>, vector<vector<cv::Mat>>, size_t)> PoolFunc);
    void convol(vector<cv::Mat> input, size_t batchnumber);
    void Relu(size_t batchnumber);
    
    
    
};

class FullyConnectedLayer{
    public:
    function<void(vector<cv::Mat>, size_t)> activation;
    vector<cv::Mat> vecWeights;
    vector<cv::Mat> vecBiases;
    vector<vector<cv::Mat>> ZTensor;
    vector<vector<cv::Mat>> ATensor;
    vector<vector<cv::Mat>> NablaTensor;
    
    FullyConnectedLayer(size_t batchsize, vector<size_t> shape, vector<size_t> inputsizevec, 
    vector<size_t> veckernel);

    
};

class Network {
    public:
    vector<convlayer> veclayer;
    FullyConnectedLayer fullco;
    vector<vector<vector<cv::Mat>>*> resultptr; 

    Network(size_t batchsize, size_t imsize, std::vector<vector<string>> Plan, vector<size_t> sizefliter, 
        vector<size_t> veckernel, vector<size_t> inputsizevec, vector<size_t> shapefullco);
    void forward(size_t batchnumber, vector<cv::Mat> input);
    void reshapeflat(size_t batchnumber, size_t veconeDsize, size_t tenoneDsize, vector<cv::Mat> input);
};

void MaxPooling(vector<cv::Mat> A, vector<vector<cv::Mat>> PoolBuffer, size_t batchnumber);
void IndPooling(vector<cv::Mat> A, vector<vector<cv::Mat>> PoolBuffer, size_t batchnumber);


    

