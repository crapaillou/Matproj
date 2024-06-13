#include <vector>
#include <cstdint>
#include <string>
#include "opencv2/opencv.hpp"

std::vector<std::vector<int>> DataToMatrix(std::vector<uint8_t> DataArray){
    std::vector<std::vector<int>> matrix(28, std::vector<int>(28));
    for (int i = 0; i < 28; i++){
        for (int j = 0; j < 28; j++){
            int index = 28 * i + j;
            matrix[i][j] = DataArray[index];
        } 
    }
    return matrix;
}

void resizeAndDisplay(cv::Mat image, int newWidth, int newHeight) {
    // Specify the new size for the image
    cv::Size size(newWidth, newHeight);

    // Create a new cv::Mat object for the resized image
    cv::Mat resizedImage;

    // Resize the image
    cv::resize(image, resizedImage, size);

    // Display the resized image
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", resizedImage);
    cv::waitKey(0);
}

std::vector<size_t> InputCalcSize(size_t Imsize, std::vector<size_t> sizeFilters, std::vector<std::vector<std::string>> Plan){
    std::vector<size_t> Size;
    Size.push_back(Imsize);
    if (Plan.size() != sizeFilters.size()){
        throw std::invalid_argument("The size of the plan and the size of the filters must be the same");
    }
    
    size_t currentSize = Imsize;
    for (size_t i = 0; i < Plan.size(); i++){
        if (Plan[i][0] == "valid"){
            if (Plan[i][1] == "false"){
                currentSize = currentSize - sizeFilters[i] + 1;
            }
            else if (Plan[i][1] == "true"){
                currentSize = currentSize - sizeFilters[i] + 1;
                currentSize = currentSize / 2;
            }
            else {
                throw std::invalid_argument("plan second element must be true or false");
                }
            }
        else if (Plan[i][0] == "same"){
            if (Plan[i][1] == "false"){
                currentSize = currentSize;
            }
            else if (Plan[i][1] == "true"){
                currentSize = currentSize / 2;
            }
            else {
                throw std::invalid_argument("plan second element must be true or false");
                }
            }
        else if (Plan[i][0] == "full"){
            if (Plan[i][1] == "false"){
                currentSize = currentSize + sizeFilters[i] - 1;
            }
            else if (Plan[i][1] == "true"){
                currentSize = currentSize + sizeFilters[i] - 1;
                currentSize = currentSize / 2;
            }
            else {
                throw std::invalid_argument("plan second element must be true or false");
                }
            }
        else {
            throw std::invalid_argument("The plan must be valid, same or full");
            }
        Size.push_back(currentSize);
        }
    return Size;
}       
