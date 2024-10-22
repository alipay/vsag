
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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>

#include "local_file_reader.h"
#include "vsag/allocator.h"
#include "vsag/errors.h"
#include "vsag/vsag.h"

int
main() {
    std::string json_str = R"(
        {
          "index_type": "HGraph",
          "metric_type": "l2",
          "dim": 128,
          "data_type": "FP32",
          "index_param": {
            "use_reorder": false,
            "graph": {
              "io_type": "block_memory",
              "io_params": {
                "block_size": 134217728
              },
              "type": "NSW",
              "graph_params": {
                "max_degree": 64,
                "init_capacity": 1000000
              }
            },
            "base_codes": {
              "io_type": "block_memory",
              "io_params": {
                "block_size": 134217728
              },
              "codes_type": "flatten_codes",
              "codes_param": {
              },
              "quantization_type": "sq8",
              "quantization_params": {
                "subspace": 64,
                "nbits": 8
              }
            }
          }
        })";
    auto index = vsag::Factory::CreateIndex("hgraph", json_str, nullptr);

    auto dim = 128;
    auto max_elements = 10000;
    std::shared_ptr<int64_t[]> ids(new int64_t[max_elements]);
    std::shared_ptr<float[]> data(new float[dim * max_elements]);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int i = 0; i < max_elements; i++) ids[i] = i;
    for (int i = 0; i < dim * max_elements; i++) data[i] = distrib_real(rng);

    // Build index
    {
        std::cout << "start build" << std::endl;
        auto dataset = vsag::Dataset::Make();
        dataset->Dim(dim)
            ->NumElements(max_elements)
            ->Ids(ids.get())
            ->Float32Vectors(data.get())
            ->Owner(false);
        if (const auto num = index.value()->Build(dataset); num.has_value()) {
            std::cout << "After Build(), Index constains: " << index.value()->GetNumElements()
                      << std::endl;
        } else if (num.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "Failed to build index: internalError" << std::endl;
            exit(-1);
        }
    }
    return 0;
}