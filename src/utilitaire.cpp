#include <vector>
#include <cstdint>

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