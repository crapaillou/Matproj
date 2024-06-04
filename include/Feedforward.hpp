#pragma once

#include <vector>
#include <functional>

using namespace std;

class Layer{
    private:
    
    size_t numKernels;
    size_t kerneldepth;
    size_t KernelDim;
    function<vector<vector<float>>(vector<vector<float>>)> activationFunction;
    function<vector<vector<float>>(vector<vector<float>>)> poolfunction;
    vector<vector<vector<float>>> kernels;
    vector<vector<vector<float>>> biases;
    vector<vector<vector<float>>> Zvec;
    vector<vector<vector<float>>> Avec;
    vector<vector<vector<float>>> error;
    vector<vector<vector<float>>> intput;
    vector<vector<vector<float>>> output;
    
    public:

    Layer(size_t numKernels, size_t kerneldepth, size_t KernelDim, 
        function<vector<vector<float>>(vector<vector<float>>)> activationFunction, 
        function<vector<vector<float>>(vector<vector<float>>)> poolfunction);

    int getkerneldepth();
};

vector<vector<float>> Relu(vector<vector<float>> vec);

vector<vector<float>> MaxPool(vector<vector<float>> vec);

class test{
    int a;
    int b;
    public:
    test(int a, int b);
    int geta();
};

