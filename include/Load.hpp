#pragma once
#include <string>
#include <iostream>
#include <string>
#include <vector>



//hello world test function.
int PrintMessage(std::string message);

//get files shape to load them afetward.
std::tuple<int, int> getfileshape();

//load the binary data into a tuple of label and data where data is the file containing image.
std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> LoadData(const char* labelfilepath, const char* datafilepath);

//put the first 784 element of a 1D vector into 28 * 28 2D vector.
std::vector<std::vector<int>> DataToMatrix(std::vector<uint8_t> DataArray);

//separate label into 3 array training validation control.
void SplitLabel(std::vector<uint8_t> original, std::vector<int> training, std::vector<int> validation, std::vector<int> control, int split1, int split2);




