#include "FeedForward.hpp"
#include <iostream>
#include "Loader.hpp"

using namespace std;


convlayer::convlayer(size_t inputsize, size_t inputdepth , vector<size_t> shapeKernel , string layerType, string Pooling, function<void(vector<cv::Mat>, vector<vector<cv::Mat>>, size_t)> PoolFunc){
    this->shapeKernel = shapeKernel;
    this->PoolFunc = PoolFunc;
    
    
    size_t SizeBatch = shapeKernel[0];
    size_t numfilter = shapeKernel[1];
    size_t Sizefilter = shapeKernel[2];

    for (size_t i = 0; i <numfilter ; i++){
        vector<cv::Mat> Kernels;
        size_t sizeB = inputsize;
        cv::Mat bias = cv::Mat::zeros(sizeB, sizeB, CV_32F);
        vecbias.push_back(bias);
        for (size_t j = 0; j < inputdepth; j++){
            cv::Mat filter(Sizefilter, Sizefilter, CV_32F);
            cv::randu(filter, -1, 1);
            filter = filter / (float)(sizeB * sizeB);
            Kernels.push_back(filter);
        }
        veckernels.push_back(Kernels);
    }
        vector<cv::Mat> Zvec;
        vector<cv::Mat> Avec;
        vector<cv::Mat> Nabla;

    for (size_t i = 0; i < SizeBatch; i++){   
        
        if (layerType == "same"){
            size_t sizeBZA = inputsize;
            
            for (size_t j = 0; j < numfilter; j++){
                cv::Mat Z = cv::Mat::zeros(sizeBZA, sizeBZA, CV_32F);
                Zvec.push_back(Z);
                cv::Mat A = cv::Mat::zeros(sizeBZA, sizeBZA, CV_32F);
                Avec.push_back(A);
                cv::Mat N = cv::Mat::zeros(sizeBZA, sizeBZA, CV_32F);
                Nabla.push_back(N);
            }
            
            
        }
        else{
            cout << "only same convolution is implemented" << endl;
        }
        if ( Pooling == "true"){
            size_t sizeBZA = Zvec[0].rows;
            vector<cv::Mat> Pool;
            for (size_t j = 0; j < numfilter; j++){
                cv::Mat pool = cv::Mat::zeros(int(sizeBZA / 2), int(sizeBZA / 2), CV_32F);
                Pool.push_back(pool);
            }
            PoolBuffer.push_back(Pool);
            
        }
        ZTensor.push_back(Zvec);
        ATensor.push_back(Avec);
        NablaTensor.push_back(Nabla); 
    }
    
}

void convlayer::convol(vector<cv::Mat> input, size_t batchnumber){
    cv::Mat temp = cv::Mat::zeros(ZTensor[batchnumber][0].rows, ZTensor[batchnumber][0].cols, CV_32F);
    for (size_t i = 0; i < veckernels[batchnumber].size(); i++){
        for (size_t j = 0; j < veckernels.size(); j++){
            cv::filter2D(input[i], temp, -1, veckernels[j][i]);
            float debugtempval = temp.at<float>(0, 0);
            float debugvecbias = vecbias[j].at<float>(0, 0);
            ZTensor[batchnumber][j] += temp + vecbias[j];
            float debugZTensorval = ZTensor[batchnumber][j].at<float>(0, 0);
        }
        
    }
}

FullyConnectedLayer::FullyConnectedLayer(size_t batchsize, vector<size_t> shape , vector<size_t> inputsizevec, 
    vector<size_t> veckernel){
    this->activation = activation;
    size_t flattenvec = inputsizevec.back() * inputsizevec.back() * veckernel.back();
    shape.insert(shape.begin(), flattenvec);
    size_t lengthnet = shape.size();
    for (size_t i = 0; i < lengthnet - 1; i++){
        cv::Mat weight = cv::Mat::zeros(shape[i], shape[i + 1], CV_32F);
        cv::randu(weight, -1, 1);
        vecWeights.push_back(weight);
        cv::Mat bias = cv::Mat::zeros(1, shape[i + 1], CV_32F);
        vecBiases.push_back(bias);
        vector<cv::Mat> Zvec;
        vector<cv::Mat> Avec;
        vector<cv::Mat> Nabla;
        for (size_t j = 0; j < batchsize; j++){
            cv::Mat Z = cv::Mat::zeros(1, shape[i], CV_32F);
            Zvec.push_back(Z);
            cv::Mat A = cv::Mat::zeros(1, shape[i], CV_32F);
            Avec.push_back(A);
            cv::Mat N = cv::Mat::zeros(1, shape[i], CV_32F);
            Nabla.push_back(N);
        }
        ZTensor.push_back(Zvec);
        ATensor.push_back(Avec);
        NablaTensor.push_back(Nabla);
    }
        
}

Network::Network(size_t batchsize, size_t imsize, std::vector<vector<string>> Plan, vector<size_t> sizefliter, vector<size_t> veckernel, vector<size_t> inputsizevec, vector<size_t> shapefullco)
    :fullco(batchsize, shapefullco, inputsizevec, veckernel)
    {
    if (Plan[0][1] == "true"){
        convlayer layer(inputsizevec[0], batchsize, {batchsize, veckernel[0], sizefliter[0]}, Plan[0][0], Plan[0][1], MaxPooling);
        this->veclayer.emplace_back(layer);
        
        
    }
    else{
        convlayer layer(inputsizevec[0], batchsize, {batchsize, veckernel[0], sizefliter[0]}, Plan[0][0], Plan[0][1], IndPooling);
        this->veclayer.emplace_back(layer);
        
    }
    for (size_t i = 1; i < Plan.size(); i++)
    {
        if (Plan[i][1] == "true"){
            convlayer layerloop(inputsizevec[i],veckernel[i-1], {batchsize, veckernel[i], sizefliter[i]}, Plan[i][0], Plan[i][1], MaxPooling);
            this->veclayer.emplace_back(layerloop);
            
        }
        else{
            convlayer layerloop(inputsizevec[i],veckernel[i-1], {batchsize, veckernel[i], sizefliter[i]}, Plan[i][0], Plan[i][1], IndPooling);
            this->veclayer.emplace_back(layerloop);
            
        }
    }
    for (size_t i = 0; i < veclayer.size(); i++){
        if (Plan[i][1] == "true"){
            resultptr.push_back(&veclayer[i].PoolBuffer);
        }
        else{
            resultptr.push_back(&veclayer[i].ATensor);
        }
    }
}

void convlayer::Relu(size_t batchnumber){
    for (size_t i = 0; i < ZTensor[batchnumber].size(); i++){
        ATensor[batchnumber][i] = cv::max(ZTensor[batchnumber][i], 0);
    }
}

void MaxPooling(vector<cv::Mat> A, vector<vector<cv::Mat>> PoolBuffer, size_t batchnumber){
    cv::Mat buffer = cv::Mat::zeros(2, 2, CV_32F);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    for (size_t i = 0; i < A.size(); i++){
        for (size_t j = 0; j < A[i].rows; j += 2){
            for (size_t k = 0; k < A[i].cols; k += 2){
                cv::Rect roi = cv::Rect(j, k, 2, 2);
                buffer = A[i](roi);
                cv::minMaxLoc(buffer, &minVal, &maxVal, &minLoc, &maxLoc);
                size_t debugsizerow = PoolBuffer[batchnumber][i].rows;
                size_t debugsizecol = PoolBuffer[batchnumber][i].cols;
                PoolBuffer[batchnumber][i].at<float>(j / 2, k / 2) = maxVal;
            }
        }
    }
}

void IndPooling(vector<cv::Mat> A, vector<vector<cv::Mat>> PoolBuffer, size_t batchnumber){
    std::cout << "do nothing" << std::endl;
}

void Network::forward(size_t batchnumber, vector<cv::Mat> input){
    veclayer[0].convol(input, batchnumber);
    veclayer[0].Relu(batchnumber);
    veclayer[0].PoolFunc(veclayer[0].ATensor[batchnumber], veclayer[0].PoolBuffer, batchnumber);

    for (size_t i = 1; i < veclayer.size(); i++){
        
        veclayer[i].convol((*resultptr[i-1])[batchnumber], batchnumber);
        veclayer[i].Relu(batchnumber);
        veclayer[i].PoolFunc(veclayer[i].ATensor[batchnumber], veclayer[i].PoolBuffer, batchnumber);
         
    }
}

void Network::reshapeflat(size_t batchnumber, size_t veconeDsize, size_t tenoneDsize, vector<cv::Mat> input){
    size_t startpoint = batchnumber * veconeDsize;
    vector<cv::Mat> vecimg = (*resultptr[3])[batchnumber];
    for (size_t i = startpoint; i < tenoneDsize; i++){
        
    }
}
