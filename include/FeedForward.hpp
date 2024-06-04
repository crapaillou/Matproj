#pragma once
#include <vector>
#include <functional>
#include <string>
#include "opencv2/opencv.hpp"

using namespace std;

class Network {
    // Empty base class
};


class convlayer : public Network {
    public:
    vector<size_t> shapeKernel; //size of the kernel and number of kernels.
    function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>, vector<cv::Mat>, vector<cv::Mat>)> convolution;
    function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>)> activation;
    vector<vector<cv::Mat>> veckernels;
    vector<cv::Mat> vecbias;
    vector<vector<cv::Mat>> ZTensor;
    vector<vector<cv::Mat>> ATensor;
    vector<vector<cv::Mat>> NablaTensor;
    vector<vector<cv::Mat>> PoolBuffer;

    
    convlayer(size_t inputsize, size_t inputdepth , vector<size_t> shapeKernel , string layerType, string Pooling,
            function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>, vector<cv::Mat>, vector<cv::Mat>)> convolution, 
            function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>)> activation);
    void forward(vector<cv::Mat> input, size_t batchnumber);
    
};

class PoolLayer : public Network  {
    public:
    function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>)> pooling;
    vector<vector<cv::Mat>> ATensor;
    
    PoolLayer(size_t batchsize, size_t numkernels , size_t inputsize ,function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>)> pooling);
};

class FullyConnectedLayer : public Network  {
    public:
    function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>)> activation;
    vector<cv::Mat> vecWeights;
    vector<cv::Mat> vecBiases;
    vector<vector<cv::Mat>> ZTensor;
    vector<vector<cv::Mat>> ATensor;
    vector<vector<cv::Mat>> NablaTensor;
    
    FullyConnectedLayer(size_t batchsize, vector<size_t> shape, function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>)> activation);
    
};



vector<vector<cv::Mat>> Relu(vector<vector<cv::Mat>> Zvec);
vector<vector<cv::Mat>> MaxPooling(vector<vector<cv::Mat>> result);
vector<vector<cv::Mat>> ConvValid(vector<vector<cv::Mat>> input, vector<cv::Mat> kernels, vector<cv::Mat> bias);
    

