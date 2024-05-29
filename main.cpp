#include <iostream>
#include <string>
#include <vector>
#include "MatProjConfig.h"
#include "header.hpp"
#include "matplot/matplot.h"





int main(int argc, char* argv[]){
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

    
    std::vector<int> training, validation, control;
    int split1 = 99999; // Index to end the training set
    int split2 = 100000; // Index to end the validation set and start the control set

    try {
        SplitLabel(Veclabel, training, validation, control, split1, split2);

        // Print the results
        std::cout << "Training: ";
        for (int i = 0; i<10; i++) {
            std::cout << training[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Validation: ";
        for (int i = 0; i<10; i++) {
            std::cout << validation[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Control: ";
        for (int i = 0; i<10; i++) {
            std::cout << control[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}


