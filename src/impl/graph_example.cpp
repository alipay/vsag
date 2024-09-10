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

#include "graph.h"
int
main() {
    int64_t num_vectors = 1000;
    size_t dim = 128;
    int64_t max_degree = 20;

    // prepare ids and vectors
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    vsag::DistanceFunc dist = vsag::GetL2DistanceFunc(dim);
    std::vector<std::vector<std::pair<float, uint32_t>>> ground_truths(num_vectors);

    vsag::DatasetPtr dataset = vsag::Dataset::Make();
    dataset->NumElements(num_vectors)->Float32Vectors(vectors)->Dim(dim)->Owner(true);
    vsag::NNdescent graph(max_degree, 30, dist);
    graph.Build(dataset);

    //    auto extract_graph = graph.GetGraph();
    //
    //    float min_loss = 0;
    //    for (int i = 0; i < num_vectors; ++i) {
    //        for (int j = 0; j < num_vectors; ++j) {
    //            if (i != j) {
    //                ground_truths[i].emplace_back(dist(vectors + i * dim, vectors + j * dim, &dim), j);
    //            }
    //        }
    //        std::sort(ground_truths[i].begin(), ground_truths[i].end());
    //        std::cout << i << " ";
    //        for (int j = 0; j < max_degree; ++j) {
    //            std::cout << ground_truths[i][j].second << " ";
    //            min_loss += ground_truths[i][j].first;
    //        }
    //        std::cout << std::endl;
    //        std::cout << i << " ";
    //        for (int j = 0; j < max_degree; ++j) {
    //            std::cout << extract_graph[i][j] << " ";
    //        }
    //        std::cout << std::endl;
    //        std::cout << std::endl;
    //    }
    //    std::cout << "min_loss:" << min_loss / (num_vectors * max_degree) << std::endl;
}