//
// Created by root on 8/16/24.
//


#include "vsag/PQCodes.h"
#include <random>
#include <iostream>
#include <chrono>

static void
Normalize(float* input_vector, int64_t dim) {
    float magnitude = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        magnitude += input_vector[i] * input_vector[i];
    }
    magnitude = std::sqrt(magnitude);

    for (int64_t i = 0; i < dim; ++i) {
        input_vector[i] = input_vector[i] / magnitude;
    }
}

static std::vector<float>
GenVector(int64_t num_vectors, int64_t dim) {
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    std::vector<float> vectors(dim * num_vectors);
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    for (int64_t i = 0; i < num_vectors; ++i) {
        Normalize(vectors.data() + i * dim, dim);
    }

    return vectors;
}

void TestPQCodesTrain()
{
    int baseCount = 100000;
    int queryCount = 100;
    int M = 120;
    int dim = 960;
    PQCodes pqCodes(M, dim);

    std::vector<float> base = GenVector(baseCount, dim);
    std::cout << "start train\n";
    pqCodes.Train(base.data(), baseCount);
    std::vector<uint8_t> codes;
    int64_t N = 10000;
    std::vector<std::vector<float>> newData;
    std::vector<std::vector<uint8_t>> newCodes(N);
    for (int i = 0; i < N; ++ i) {
        newData.emplace_back(GenVector(64, dim));
        pqCodes.BatchEncode(newData[i].data(), 64, newCodes[i]);
        pqCodes.Packaged(newCodes[i]);
    }

    std::vector<float> query = GenVector(100, dim);
    std::vector<float> dist(64);
    for (int i = 0; i < 100; ++ i) {
        PQScanner scanner(&pqCodes);
        scanner.SetQuery(query.data() + i * dim);
        for (int j = 0; j < 1000; ++ j) {
            auto idx = random() % N;
            auto t1 = std::chrono::steady_clock::now();
            scanner.ScanCodes(newCodes[idx].data(), dist);
            auto t2 = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration<double, std::nano>(t2 - t1).count() << "\n";
        }
    }

}


int main()
{
    TestPQCodesTrain();
}