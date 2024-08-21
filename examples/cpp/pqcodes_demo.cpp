//
// Created by root on 8/16/24.
//


#include "vsag/PQCodes.h"
#include "vsag/vsag.h"
#include <random>
#include <iostream>
#include <chrono>
#include "fmt/format-inl.h"

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
    int baseCount = 10000;
    int queryCount = 100;
    int M = 32;
    int dim = 32;
    auto baseVec = GenVector(baseCount, dim);
    auto queryVec = GenVector(queryCount, dim);
    std::string algo_name = "hnsw";
    constexpr const char* BUILD_PARAM = R"(
    {{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "hnsw": {{
            "max_degree": 32,
            "ef_construction": 200,
            "sq_num_bits": {}
        }}
    }}
    )";
    auto* base_id = new int64_t[baseCount];
    for (int64_t i = 0; i < baseCount; i++) {
        base_id[i] = i;
    }
    std::string build_parameters = fmt::format(BUILD_PARAM, dim, 12);
    std::cout << build_parameters << std::endl;
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(baseCount)
            ->Dim(dim)
            ->Ids(base_id)
            ->Float32Vectors(baseVec.data())
            ->Owner(false);

    auto index = vsag::Factory::CreateIndex(algo_name, build_parameters).value();
    index->Build(dataset);
    int k = 10;
    constexpr auto search_parameters_json = R"(
        {{
            "hnsw": {{
                "ef_search": {}
            }}
        }}
        )";
    auto search_parameters = fmt::format(search_parameters_json, 10);
    for (int i = 0; i < queryCount; ++ i) {
        auto single_query = vsag::Dataset::Make();
        single_query->NumElements(1)->Dim(dim)->Owner(false);
        vsag::DatasetPtr ann_result;
        single_query->Float32Vectors(queryVec.data() + i * dim);
        index->KnnSearch(single_query, k, search_parameters);
    }

}


int main()
{
    TestPQCodesTrain();
}