#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstdint>
#include <tuple>



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
    
    //print label to verify.
    for (int i = 0; i<10; i++){
        std::cout << "label[" << i << "] = " << int(labels[i]) << std::endl;
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
    std::cout << "data loaded with no error!" << std::endl;
    return std::make_tuple(labels, data);
}

void SplitLabel(std::vector<uint8_t> original, std::vector<int> training, std::vector<int> validation, std::vector<int> control, int split1, int split2){
    // Check if split points are valid
    if (split1 >= split2 || split2 > original.size()) {
        throw std::invalid_argument("Invalid split points.");
    }

    // Clear any existing data in the output vectors
    training.clear();
    validation.clear();
    control.clear();

    // Split the original vector into three parts
    training.insert(training.end(), original.begin(), original.begin() + split1);
    validation.insert(validation.end(), original.begin() + split1, original.begin() + split2);
    control.insert(control.end(), original.begin() + split2, original.end());
}



