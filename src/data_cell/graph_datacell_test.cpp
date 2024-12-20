
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

#include "graph_datacell.h"

#include "catch2/catch_template_test_macros.hpp"
#include "fmt/format-inl.h"
#include "graph_interface_test.h"
#include "io/io_headers.h"
#include "safe_allocator.h"
using namespace vsag;

template <typename IOTemp>
void
TestGraphDataCell(const JsonType& graph_param,
                  const JsonType& io_param,
                  const IndexCommonParam& param) {
    auto counts = {1000, 2000};
    auto max_id = 10000;
    for (auto count : counts) {
        auto graph = std::make_shared<GraphDataCell<IOTemp, false>>(graph_param, io_param, param);
        GraphInterfaceTest test(graph);
        auto other = std::make_shared<GraphDataCell<IOTemp, false>>(graph_param, io_param, param);
        test.BasicTest(max_id, count, other);
    }
}

TEST_CASE("graph basic test", "[ut][graph_datacell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dims = {32, 64};
    auto max_degrees = {5, 12, 24, 32, 64, 128};
    auto max_capacities = {1, 100, 10000, 10'000'000, 32'179'837};
    std::vector<JsonType> graph_params;
    constexpr const char* graph_param_temp = R"(
        {{
            "max_degree": {},
            "init_capacity": {}
        }}
        )";
    for (auto degree : max_degrees) {
        for (auto capacity : max_capacities) {
            auto str = fmt::format(graph_param_temp, degree, capacity);
            graph_params.emplace_back(JsonType::parse(str));
        }
    }
    auto io_param = JsonType::parse("{}");
    for (auto dim : dims) {
        IndexCommonParam param;
        param.dim_ = dim;
        param.allocator_ = allocator;
        for (auto& gp : graph_params) {
            TestGraphDataCell<MemoryIO>(gp, io_param, param);
            TestGraphDataCell<MemoryBlockIO>(gp, io_param, param);
        }
    }
}
