//std includes
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <math.h>
#include <tuple>

//libs includes.
#include "MatProjConfig.h"
#include "Loader.hpp"
#include "FeedForward.hpp"
#include "matplot/matplot.h"
#include "opencv2/opencv.hpp"


int main(int argc, char* argv[]){
    //Program parameters
    size_t ImSize = 28;
    size_t BatchSize = 10;
    
    //print project version and name
    std::cout << argv[0] << "Project version : "
    <<  MatProj_VERSION_MAJOR << "."
    << MatProj_VERSION_MINOR << "."
    << MatProj_VERSION_PATCH << std::endl;

    //path to files 
    const char* LabelPath = "../Data/label.bin";
    const char* DataPath = "../Data/data.bin";

    // load label and data to vector.
    auto dataTuple = LoadData(LabelPath, DataPath);
    std::vector<uint8_t> Veclabel = std::get<0>(dataTuple);
    std::vector<uint8_t> VecData = std::get<1>(dataTuple);
    
    //sperate veclabel and vecdata
    std::vector<uint8_t> LabelTraining, LabelValidation, LabelControl;
    SplitLabel(Veclabel, 100000, 110000, LabelTraining, LabelValidation, LabelControl);
    
    int born1 = 100000 * pow(ImSize,2);
    int born2 = 110000 * pow(ImSize,2);
    std::vector<uint8_t> DataTraining, DataValidation, DataControl;
    SplitLabel(VecData, born1, born2, DataTraining, DataValidation, DataControl);

    //put the image of the binary vecdata into a vector<vector<Mat>>
    std::vector<std::vector<cv::Mat>> BatchTraining = VecToBacths(DataTraining, BatchSize, ImSize, ImSize);
    std::vector<std::vector<cv::Mat>> BatchValidation = VecToBacths(DataValidation, BatchSize, ImSize, ImSize);
    std::vector<std::vector<cv::Mat>> BatchControl = VecToBacths(DataControl, BatchSize, ImSize, ImSize);
    
    //create a layer
    //calculate the size of the input of the network.
    std::vector<size_t> sizeFilters = {3, 3, 3, 3};
    std::vector<std::vector<std::string>> Plan = {{"valid", "false"}, {"valid", "true"}, {"valid", "false"}, {"valid", "false"}};
    std::vector<size_t> Size = InputCalcSize(ImSize, sizeFilters, Plan);
    

    //first layer
    std::vector<size_t> shapekernel  = {BatchSize, 32, sizeFilters[0]};
    convlayer conv1(Size[0], 1, shapekernel, "valid", "false", ConvValid, Relu);

    //second layer
    std::vector<size_t> shapekernel2  = {BatchSize, 32, sizeFilters[1]};
    convlayer conv2(Size[1], 32, shapekernel2, "valid", "Max", ConvValid, Relu);

    //third layer
    std::vector<size_t> shapekernel3  = {BatchSize, 64, sizeFilters[2]};
    convlayer conv3(Size[2], 64, shapekernel3, "valid", "false", ConvValid, Relu);

    //fourth layer
    std::vector<size_t> shapekernel4  = {BatchSize, 64, sizeFilters[3]};
    convlayer conv4(Size[3], 64, shapekernel4, "valid", "false", ConvValid, Relu);

    //fully connected layer
    size_t linSize = Size[4] * Size[4] * 64;
    FullyConnectedLayer fc1(BatchSize, {linSize, 300, 10}, Relu);
    
    //linking the layers
    //std::cout << "kernel 1 : " << conv1.veckernels[0][0] << std::endl;
    //std::cout << "imge : " << BatchTraining[0][0] << std::endl;

    
    
    
    return 0;
}



