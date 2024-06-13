//std includes
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include "MatProjConfig.h"
#include "Loader.hpp"
#include "FeedForward.hpp"
#include "opencv2/opencv.hpp"







int main(int argc, char* argv[]){
    //Program parameters
    size_t ImSize = 28;
    size_t BatchSize = 20;
    
    //print project version and name
    std::cout << argv[0] << "Project version : "
    <<  MatProj_VERSION_MAJOR << "."
    << MatProj_VERSION_MINOR << "."
    << MatProj_VERSION_PATCH << std::endl;

    //path to files 
    const char* LabelPath = "../Data/label.bin";
    const char* DataPath = "../Data/data.bin";

    
    //creat a layer
    std::vector<vector<string>> Plan = {{"same","true"},{"same", "false"},
                                        {"same","false"},{"same","true"}};
    vector<size_t> sizefliter = {3,3,3,3};
    vector<size_t> veckernel = {32,32,64,64};
    vector<size_t> shapekernel = {BatchSize, 32,3};
    vector<size_t> inputsizevec = InputCalcSize(ImSize, sizefliter, Plan);
    vector<size_t> shapefullco = {300,100,10};

    Network net(BatchSize, ImSize, Plan, sizefliter, veckernel, inputsizevec, shapefullco);
    
    size_t numimg = 20;
    vector<cv::Mat> imagevec;
    for (size_t i = 0; i < numimg; i++){
        cv::Mat image(ImSize, ImSize, CV_32F);
        cv::randu(image, 0, 255);
        imagevec.push_back(image);
    }
    
    /*net.veclayer[0].convol(imagevec, 0);
    net.veclayer[0].Relu(0);
    net.veclayer[0].PoolFunc(net.veclayer[0].ATensor[0], *net.resultptr[0], 0);
    

    net.veclayer[1].convol((*net.resultptr[0])[0], 0);
    net.veclayer[1].Relu(0);
    
    net.veclayer[2].convol((*net.resultptr[1])[0], 0);
    net.veclayer[2].Relu(0);
    
    net.veclayer[3].convol((*net.resultptr[2])[0], 0);
    net.veclayer[3].Relu(0);
    net.veclayer[3].PoolFunc((*net.resultptr[2])[0], *net.resultptr[3], 0);
    std::cout << "layer 3 pool : " << (*net.resultptr[3])[0][0] << std::endl;*/
    net.forward(0, imagevec);
    return 0;
}



