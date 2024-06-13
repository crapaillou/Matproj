#pragma once
#include <string>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

//get files shape to load them afetward.
std::tuple<int, int> getfileshape();

//load the binary data into a tuple of label and data where data is the file containing image.
std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> LoadData(const char* labelfilepath, const char* datafilepath);

//put the first 784 element of a 1D vector into 28 * 28 2D vector.
std::vector<std::vector<int>> DataToMatrix(std::vector<uint8_t> DataArray);

//separate label into 3 array training validation control.
void SplitLabel(std::vector<uint8_t>& input, size_t split1, size_t split2, 
                  std::vector<uint8_t>& training, std::vector<uint8_t>& validation, std::vector<uint8_t>& control);
//put the image of the binary vecdata into a vector<vector<Mat>>
std::vector<std::vector<cv::Mat>> VecToBacths(std::vector<uint8_t> input, size_t batchSize, size_t ImWidth, size_t ImgHeight);

//resize the image to a specific size and display it.
void resizeAndDisplay(cv::Mat image, int newWidth, int newHeight);

//calculate the size of the input of the network.
std::vector<size_t> InputCalcSize(size_t Imsize, std::vector<size_t> sizeFilters, std::vector<std::vector<std::string>> Plan);



