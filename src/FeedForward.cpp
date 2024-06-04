#include "FeedForward.hpp"
#include <iostream>

using namespace std;


convlayer::convlayer(size_t inputsize, size_t inputdepth, vector<size_t> shapeKernel, string layerType, string Pooling,
            function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>, vector<cv::Mat>, vector<cv::Mat>)> convolution, 
            function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>)> activation){
    this->shapeKernel = shapeKernel;
    this->convolution = convolution;
    this->activation = activation;
    
    
    size_t SizeBatch = shapeKernel[0];
    size_t numfilter = shapeKernel[1];
    size_t Sizefilter = shapeKernel[2];

    for (size_t i = 0; i <numfilter ; i++){
        vector<cv::Mat> Kernels;
        size_t sizeB = inputsize - Sizefilter + 1;
        cv::Mat bias = cv::Mat::zeros(sizeB, sizeB, CV_32F);
        vecbias.push_back(bias);
        for (size_t j = 0; j < inputdepth; j++){
            cv::Mat filter(Sizefilter, Sizefilter, CV_32F);
            cv::randu(filter, 0, 1);
            Kernels.push_back(filter);
        }
        veckernels.push_back(Kernels);
    }
        vector<cv::Mat> Zvec;
        vector<cv::Mat> Avec;
        vector<cv::Mat> Nabla;

    for (size_t i = 0; i < SizeBatch; i++){   
        if (layerType == "valid"){
            size_t sizeBZA = inputsize - Sizefilter + 1;
            for (size_t j = 0; j < numfilter; j++){
                cv::Mat Z = cv::Mat::zeros(sizeBZA, sizeBZA, CV_32F);
                Zvec.push_back(Z);
                cv::Mat A = cv::Mat::zeros(sizeBZA, sizeBZA, CV_32F);
                Avec.push_back(A);
                cv::Mat N = cv::Mat::zeros(sizeBZA, sizeBZA, CV_32F);
                Nabla.push_back(N);

            }
        }
        else if (layerType == "same"){
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
        else if (layerType == "full"){
            size_t sizeBZA = inputsize + Sizefilter - 1;
            
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
            cout << "Invalid layer type" << endl;
        }
        if ( Pooling == "Max"){
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

void convlayer::forward(vector<cv::Mat> input, size_t batchnumber){
    cv::Mat temp = cv::Mat::zeros(ZTensor[batchnumber][0].rows, ZTensor[batchnumber][0].cols, CV_32F);
    for (size_t i = 0; i < veckernels.size(); i++){
        for (size_t j = 0; j < veckernels[i].size(); j++){
            cv::filter2D(input[i], temp, -1, veckernels[i][j]);
            ZTensor[batchnumber][i] += temp;
        }
    ZTensor[batchnumber][i] += vecbias[i];
    }
}


PoolLayer::PoolLayer(size_t batchsize, size_t numkernels , size_t inputsize, function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>)> pooling){
    this->pooling = pooling;
    for (size_t i = 0; i < batchsize; i++){
        vector<cv::Mat> Avec;
        for (size_t j = 0; j < numkernels; j++){
            size_t sizeBZA = int(inputsize / 2); //ensure that the inputsize is divisible by 2.
            cv::Mat A = cv::Mat::zeros(sizeBZA, sizeBZA, CV_32F);
            Avec.push_back(A);
        }
        ATensor.push_back(Avec);
    }
}

FullyConnectedLayer::FullyConnectedLayer(size_t batchsize, vector<size_t> shape , function<vector<vector<cv::Mat>>(vector<vector<cv::Mat>>)> activation){
    this->activation = activation;
    size_t lengthnet = shape.size();
    for (size_t i = 0; i < lengthnet - 1; i++){
        cv::Mat weight = cv::Mat::zeros(shape[i], shape[i + 1], CV_32F);
        cv::randu(weight, 0, 1);
        vecWeights.push_back(weight);
        cv::Mat bias = cv::Mat::zeros(1, shape[i + 1], CV_32F);
        vecBiases.push_back(bias);
        vector<cv::Mat> Zvec;
        vector<cv::Mat> Avec;
        vector<cv::Mat> Nabla;
        for (size_t j = 0; j < batchsize; j++){
            cv::Mat Z = cv::Mat::zeros(1, shape[i + 1], CV_32F);
            Zvec.push_back(Z);
            cv::Mat A = cv::Mat::zeros(1, shape[i + 1], CV_32F);
            Avec.push_back(A);
            cv::Mat N = cv::Mat::zeros(1, shape[i + 1], CV_32F);
            Nabla.push_back(N);
        }
        ZTensor.push_back(Zvec);
        ATensor.push_back(Avec);
        NablaTensor.push_back(Nabla);
    }
        
}



vector<vector<cv::Mat>> Relu(vector<vector<cv::Mat>> Zvec){
    vector<vector<cv::Mat>> result;
    for (size_t i = 0; i < Zvec.size(); i++){
        vector<cv::Mat> resultbatch;
        for (size_t j = 0; j < Zvec[i].size(); j++){
            cv::Mat resultmat = Zvec[i][j].clone();
            for (int k = 0; k < resultmat.rows; k++){
                for (int l = 0; l < resultmat.cols; l++){
                    if (resultmat.at<float>(k, l) < 0){
                        resultmat.at<float>(k, l) = 0;
                    }
                }
            }
            resultbatch.push_back(resultmat);
        }
        result.push_back(resultbatch);
    }
    return result;
}

vector<vector<cv::Mat>> MaxPooling(vector<vector<cv::Mat>> result){
    vector<vector<cv::Mat>> resultpool;
    for (size_t i = 0; i < result.size(); i++){
        vector<cv::Mat> resultbatch;
        for (size_t j = 0; j < result[i].size(); j++){
            cv::Mat resultmat = result[i][j].clone();
            cv::Mat resultpoolmat = cv::Mat::zeros(resultmat.rows / 2, resultmat.cols / 2, CV_32F);
            for (int k = 0; k < resultmat.rows; k += 2){
                for (int l = 0; l < resultmat.cols; l += 2){
                    float max = resultmat.at<float>(k, l);
                    for (int m = k; m < k + 2; m++){
                        for (int n = l; n < l + 2; n++){
                            if (resultmat.at<float>(m, n) > max){
                                max = resultmat.at<float>(m, n);
                            }
                        }
                    }
                    resultpoolmat.at<float>(k / 2, l / 2) = max;
                }
            }
            resultbatch.push_back(resultpoolmat);
        }
        resultpool.push_back(resultbatch);
    }
    return resultpool;
}

vector<vector<cv::Mat>> ConvValid(vector<vector<cv::Mat>> input, vector<cv::Mat> kernels, vector<cv::Mat> bias){
    vector<vector<cv::Mat>> result;
    for (size_t i = 0; i < input.size(); i++){
        vector<cv::Mat> resultbatch;
        for (size_t j = 0; j < kernels.size(); j++){
            cv::Mat resultmat = cv::Mat::zeros(input[i][0].rows - kernels[j].rows + 1, input[i][0].cols - kernels[j].cols + 1, CV_32F);
            for (size_t k = 0; k < input[i].size(); k++){
                cv::Mat conv = cv::Mat::zeros(input[i][0].rows - kernels[j].rows + 1, input[i][0].cols - kernels[j].cols + 1, CV_32F);
                cv::filter2D(input[i][k], conv, -1, kernels[j]);
                resultmat += conv;
            }
            resultmat += bias[j];
            resultbatch.push_back(resultmat);
        }
        result.push_back(resultbatch);
    }
    return result;
}
