//
// Created by root on 8/23/24.
//


#include "header.h"
#include <iostream>
#include <vector>
#include <chrono>

double Predict()
{
    std::vector<float> input = {3117., 2922., 1654. , 238. , 402.};

    int p = 1;

    Entry en[5];
    for (int i = 0; i < 5; ++ i) {
        en[i].fvalue = input[i];
    }

    std::vector<double> out(1, 0);
    auto* out_result = static_cast<double*>(out.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    predict(en, 0, out_result);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "predict cost: \t" << std::chrono::duration<double>(t2 - t1).count() << std::endl;

    return out[0];
    /*I know the above return statement is completely insignificant. But i wanted to use the loaded model to predict the data points further.*/
}

int main() {
    std::cout << Predict() << "\n";

    std::cout << "Ok complete!"<< std::endl;
    return 0;
}
