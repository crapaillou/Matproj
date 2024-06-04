#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include "MatProjConfig.h"
#include "Load.hpp"
#include "Feedforward.hpp"
#include "opencv2/opencv.hpp"







int main(int argc, char* argv[]){
    //print project version and name
    std::cout << argv[0] << "Project version : "
    <<  MatProj_VERSION_MAJOR << "."
    << MatProj_VERSION_MINOR << "."
    << MatProj_VERSION_PATCH << std::endl;

    //path to files 
    const char* LabelPath = "../Data/label.bin";
    const char* DataPath = "../Data/data.bin";

    
    //generate to cv random matrix and display it.
    cv::Mat cvmatrix = cv::Mat::ones(10, 10, CV_32F);
    cv::Mat cvmatrix2 = cv::Mat::ones(10, 10, CV_32F);
    cv::Mat res = cvmatrix + cvmatrix2;
    std::cout << res << std::endl;
    
    return 0;
}


