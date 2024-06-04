#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstdint>
#include <tuple>
#include <stdexcept>
#include "opencv2/opencv.hpp"

std::tuple<int, int> getfileshape(){
    //open the file and creat a string to put the first line in.
    std::ifstream shapeFile("../Data/data_shape.txt");
    std::string shapeStr;
    
    //get the first line.
    std::getline(shapeFile, shapeStr);
    
    std::stringstream ss(shapeStr);
    std::string item;
    std::vector<int> ShapeData;
    while (std::getline(ss, item, ',')) {
        ShapeData.push_back(std::stoi(item));
        
    }
    
    // Read the second line from the file into shapeStr
    std::getline(shapeFile, shapeStr);

    // Reset ss for the new line
    ss.clear(); // Clear any error flags
    ss.str(shapeStr); // Set the new string for the second line
    
    std::vector<int> ShapeLabel;
    while (std::getline(ss, item, ',')){
        ShapeLabel.push_back(std::stoi(item));
    }
    
    int DataTotalSize = ShapeData[0] * ShapeData[1] * ShapeData[2];
    int LabelTotalSize = ShapeLabel[0];
    
    

    return std::make_tuple(DataTotalSize, LabelTotalSize);
}


std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> LoadData(const char* labelfilepath, const char* datafilepath){
    std::cout << "begin loading data" << std::endl;
    std::ifstream LabelFile(labelfilepath, std::ios::binary);
    if (!LabelFile) {
        std::cerr << "Unable to open file label file";
        exit(1);
    }
    
    std::vector<uint8_t> labels;
    uint8_t Labelvalue;
    while (LabelFile.read(reinterpret_cast<char*>(&Labelvalue), sizeof(uint8_t))) {
        labels.push_back(Labelvalue);
    }
    
    LabelFile.close();
    
    //load the data file
    std::ifstream DataFile(datafilepath, std::ios::binary);
    if (!DataFile) {
        std::cerr << "Unable to open file data file";
        exit(1);  // call system to stop
    }

    std::vector<uint8_t> data;
    uint8_t Datavalue;
    while (DataFile.read(reinterpret_cast<char*>(&Datavalue), sizeof(uint8_t))) {
        data.push_back(Datavalue);
    }

    DataFile.close();
    std::cout << "data loaded successfully!" << std::endl;
    return std::make_tuple(labels, data);
}

void SplitLabel(std::vector<uint8_t>& input, size_t split1, size_t split2, 
                  std::vector<uint8_t>& training, std::vector<uint8_t>& validation, std::vector<uint8_t>& control) {
    // Clear the output vectors
    training.clear();
    validation.clear();
    control.clear();
    
    // Ensure the split points are valid
    if (split1 > split2 || split2 > input.size()) {
        throw std::invalid_argument("Invalid split points");
    }
    
    // Populate the training vector
    training.insert(training.end(), input.begin(), input.begin() + split1);
    
    // Populate the validation vector
    validation.insert(validation.end(), input.begin() + split1, input.begin() + split2);
    
    // Populate the control vector
    control.insert(control.end(), input.begin() + split2, input.end());

}

std::vector<std::vector<cv::Mat>> VecToBacths(std::vector<uint8_t> input, size_t BatchSize, size_t ImWidth, size_t ImgHeight){
    //declare vector<vector<Mat>> called vecbatches
    std::vector<std::vector<cv::Mat>> vecbatches;
    
    
    int ImSize = ImgHeight * ImWidth;
    int elements = ImSize * BatchSize;
    int NumBatch = input.size() / elements;
    size_t vecsize = input.size();
    if(input.size() % elements != 0){
        throw std::invalid_argument("batch size must be a multiple of the total num of image");
    }
    
    int index = 0;
    for (size_t i = 0; i<NumBatch; i++){
        std::vector<cv::Mat> batch;
        for (size_t j = 0; j<BatchSize; j++){
            index = i * elements + j * ImSize;
            cv::Mat image(ImWidth, ImgHeight, CV_8UC1, input.data() + index);
            batch.push_back(image.clone());
        }
        vecbatches.push_back(batch);
    }
    return vecbatches;
}
