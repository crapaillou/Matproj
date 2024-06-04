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
        function<vector<vector<float>>(vector<vector<float>>)> poolfunction){
        
        // Initialize the layer
        this->numKernels = numKernels;
        this->kerneldepth = kerneldepth;
        this->KernelDim = KernelDim;
        this->activationFunction = activationFunction;
        this->poolfunction = poolfunction;

        // Initialize kernels and biases
        for (size_t i = 0; i < numKernels; i++){
            vector<vector<float>> kernel;
            vector<vector<float>> bias;
            for (size_t j = 0; j < kerneldepth; j++){
                vector<float> row;
                for (size_t k = 0; k < KernelDim; k++){
                    row.push_back(0.0);
                }
                kernel.push_back(row);
                bias.push_back(row);
            }
            kernels.push_back(kernel);
            biases.push_back(bias);
        }
    }
    int getkerneldepth(){
        return kerneldepth;
    }
};

vector<vector<float>> Relu(vector<vector<float>> vec){
    vector<vector<float>> result;
    for (size_t i = 0; i < vec.size(); i++){
        vector<float> row;
        for (size_t j = 0; j < vec[i].size(); j++){
            row.push_back(max(0.0f, vec[i][j]));
        }
        result.push_back(row);
    }
    return result;
}

vector<vector<float>> MaxPool(vector<vector<float>> vec){
    vector<vector<float>> result;
    for (size_t i = 0; i < vec.size(); i+=2){
        vector<float> row;
        for (size_t j = 0; j < vec[i].size(); j+=2){
            float maxval = vec[i][j];
            maxval = max(maxval, vec[i][j+1]);
            maxval = max(maxval, vec[i+1][j]);
            maxval = max(maxval, vec[i+1][j+1]);
            row.push_back(maxval);
        }
        result.push_back(row);
    }
    return result;
}

class test{
    int a;
    int b;
    public:
    test(int a, int b){
        this->a = a;
        this->b = b;
    }
    int geta(){
        return a;
    }
};

