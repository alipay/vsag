
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <limits>

#include "simd/simd.h"
#include "test_index.h"
#include "vsag/vsag.h"

const std::string tmp_dir = "/tmp/";

namespace vsag {

extern float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

extern float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

}  // namespace vsag

bool
is_path_belong_to(const std::string& a, const std::string& b) {
    return b.compare(0, a.size(), a) == 0;
}

std::string
create_random_string(bool is_full) {
    const std::vector<std::string> level1 = {"a", "b", "c"};
    const std::vector<std::string> level2 = {"d", "e"};
    const std::vector<std::string> level3 = {"f", "g", "h"};

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> distr;

    std::vector<std::string> selected_levels;

    if (is_full) {
        selected_levels.push_back(level1[distr(mt) % level1.size()]);
        selected_levels.push_back(level2[distr(mt) % level2.size()]);
        selected_levels.push_back(level3[distr(mt) % level3.size()]);
    } else {
        std::uniform_int_distribution<> dist(1, 3);
        int num_levels = dist(mt);

        if (num_levels >= 1) {
            selected_levels.push_back(level1[distr(mt) % level1.size()]);
        }
        if (num_levels >= 2) {
            selected_levels.push_back(level2[distr(mt) % level2.size()]);
        }
        if (num_levels == 3) {
            selected_levels.push_back(level3[distr(mt) % level3.size()]);
        }
    }

    std::string random_string = selected_levels.empty() ? "" : selected_levels[0];
    for (size_t i = 1; i < selected_levels.size(); ++i) {
        random_string += "/" + selected_levels[i];
    }

    return random_string;
}

TEST_CASE("pyramid", "[ft][index][pyramid]") {
    // TODO(inabao): Reconstruct the pyramid's tests using a new framework in the future.
    int64_t num_vectors = 1000;
    size_t dim = 128;
    int64_t topk = 10;

    // prepare ids and vectors
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    auto paths = new std::string[num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    for (int64_t i = 0; i < num_vectors; ++i) {
        paths[i] = create_random_string(true);
    }

    // create index
    auto pyramid_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "sub_index_type": "hnsw",
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("pyramid", pyramid_build_paramesters).value();
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids)
        ->Float32Vectors(vectors)
        ->Paths(paths)
        ->Owner(false);
    index->Build(base);

    // Serialize(multi-file)
    {
        if (auto bs = index->Serialize(); bs.has_value()) {
            index = nullptr;
            auto keys = bs->GetKeys();
            for (auto key : keys) {
                vsag::Binary b = bs->Get(key);
                std::ofstream file(tmp_dir + "pyramid.index." + key, std::ios::binary);
                file.write((const char*)b.data.get(), b.size);
                file.close();
            }
            std::ofstream metafile(tmp_dir + "pyramid.index._meta", std::ios::out);
            for (auto key : keys) {
                metafile << key << std::endl;
            }
            metafile.close();
        } else if (bs.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY) {
            std::cerr << "no enough memory to serialize index" << std::endl;
        }
    }

    // Deserialize(binaryset)
    {
        std::ifstream metafile(tmp_dir + "pyramid.index._meta", std::ios::in);
        std::vector<std::string> keys;
        std::string line;
        while (std::getline(metafile, line)) {
            keys.push_back(line);
        }
        metafile.close();

        vsag::BinarySet bs;
        for (auto key : keys) {
            std::ifstream file(tmp_dir + "pyramid.index." + key, std::ios::in);
            file.seekg(0, std::ios::end);
            vsag::Binary b;
            b.size = file.tellg();
            b.data.reset(new int8_t[b.size]);
            file.seekg(0, std::ios::beg);
            file.read((char*)b.data.get(), b.size);
            bs.Set(key, b);
        }
        index = vsag::Factory::CreateIndex("pyramid", pyramid_build_paramesters).value();
        index->Deserialize(bs);
    }

    // prepare a query vector
    // memory will be released by query the dataset since owner is set to true when creating the query.
    auto query_count = 100;
    auto query_vector = new float[dim * query_count];
    auto query_path = new std::string[query_count];
    for (int64_t i = 0; i < dim * query_count; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    for (int64_t i = 0; i < query_count; ++i) {
        query_path[i] = create_random_string(false);
    }

    // search on the index
    auto pyramid_search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    float final_recall = 0;
    for (int i = 0; i < query_count; ++i) {
        std::priority_queue<std::pair<float, int64_t>> ground_truths;
        for (int j = 0; j < num_vectors; ++j) {
            if (is_path_belong_to(query_path[i], paths[j])) {
                auto distance = vsag::L2Sqr(vectors + j * dim, query_vector + i * dim, &dim);
                ground_truths.push({distance, j});
            }
            if (ground_truths.size() > topk) {
                ground_truths.pop();
            }
        }

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(query_vector + i * dim)
            ->Paths(query_path + i)
            ->Owner(false);
        auto result = index->KnnSearch(query, topk, pyramid_search_parameters).value();
        auto result_ids = result->GetIds();

        std::unordered_set<int64_t> neighbors_set(result_ids, result_ids + result->GetDim());
        auto ground_size = ground_truths.size();
        float correct = 0;
        while (not ground_truths.empty()) {
            auto [distance, id] = ground_truths.top();
            ground_truths.pop();
            if (neighbors_set.find(id) != neighbors_set.end()) {
                correct += 1.0f;
            }
        }

        auto recall = correct / ground_size;
        final_recall += recall;
    }

    REQUIRE((final_recall / query_count) == 1.0f);
    // free memory
    delete[] ids;
    delete[] vectors;
    delete[] paths;
    delete[] query_path;
    delete[] query_vector;
}
