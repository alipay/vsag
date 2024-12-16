
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

TEST_CASE_PERSISTENT_FIXTURE(fixtures::TestIndex,
                             "pyramid Build & ContinueAdd Test",
                             "[ft][hnsw]") {
    auto dims = fixtures::get_common_used_dims(3);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    for (auto& dim : dims) {
        constexpr auto build_parameter_json = R"(
        {{
            "dtype": "float32",
            "metric_type": "{}",
            "dim": {},
            "index_param": {{
                "sub_index_type": "{}",
                "{}": {{
                    "max_degree": 24,
                    "ef_construction": 200
                }}
            }}
        }}
        )";
        auto param = fmt::format(build_parameter_json, metric_type, name, name, dim);
        auto index = TestFactory(name, param, true);
        TestBuildIndex(index, dim);
        TestContinueAdd(index, dim);
    }
}