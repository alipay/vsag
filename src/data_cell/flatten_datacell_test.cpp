
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

#include "flatten_datacell.h"

#include <algorithm>

#include "catch2/catch_template_test_macros.hpp"
#include "default_allocator.h"
#include "fixtures.h"
#include "flatten_interface_test.h"
#include "io/io_headers.h"
#include "quantization/quantizer_headers.h"

using namespace vsag;

template <typename QuantTmpl, typename IOTmpl, MetricType metric>
void
TestFlattenDataCell(const IndexCommonParam& common_param,
                    const JsonType& quantizer_json,
                    const JsonType& io_json,
                    float error = 1e-5) {
    auto counts = {100, 1000};
    for (auto count : counts) {
        auto flatten = std::make_shared<FlattenDataCell<QuantTmpl, IOTmpl>>(
            quantizer_json, io_json, common_param);
        FlattenInterfaceTest test(flatten, metric);
        test.BasicTest(common_param.dim_, count, error);
        auto other = std::make_shared<FlattenDataCell<QuantTmpl, IOTmpl>>(
            quantizer_json, io_json, common_param);
        test.TestSerializeAndDeserialize(common_param.dim_, other, error);
    }
}

template <typename IOTmpl>
void
TestFlattenDataCellFP32(const IndexCommonParam& common_param,
                        const JsonType& quantizer_json,
                        const JsonType& io_json,
                        float error = 1e-5) {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    TestFlattenDataCell<FP32Quantizer<metrics[0]>, IOTmpl, metrics[0]>(
        common_param, quantizer_json, io_json, error);
    TestFlattenDataCell<FP32Quantizer<metrics[1]>, IOTmpl, metrics[1]>(
        common_param, quantizer_json, io_json, error);
    TestFlattenDataCell<FP32Quantizer<metrics[2]>, IOTmpl, metrics[2]>(
        common_param, quantizer_json, io_json, error);
}

TEST_CASE("fp32", "[ut][flatten_data_cell]") {
    auto allocator = std::make_shared<SafeAllocator>(DefaultAllocator::Instance());
    auto fp32_param = JsonType::parse("{}");
    auto io_param = JsonType::parse("{}");
    auto dims = {8, 64, 512};
    float error = 1e-5;
    for (auto dim : dims) {
        IndexCommonParam common_param;
        common_param.dim_ = dim;
        common_param.allocator_ = allocator;
        common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
        TestFlattenDataCellFP32<MemoryIO>(common_param, fp32_param, io_param, error);
        TestFlattenDataCellFP32<MemoryBlockIO>(common_param, fp32_param, io_param, error);
    }
}

template <typename IOTmpl>
void
TestFlattenDataCellSQ8(const IndexCommonParam& common_param,
                       const JsonType& quantizer_json,
                       const JsonType& io_json,
                       float error = 1e-5) {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    TestFlattenDataCell<SQ8Quantizer<metrics[0]>, IOTmpl, metrics[0]>(
        common_param, quantizer_json, io_json, error);
    TestFlattenDataCell<SQ8Quantizer<metrics[1]>, IOTmpl, metrics[1]>(
        common_param, quantizer_json, io_json, error);
    TestFlattenDataCell<SQ8Quantizer<metrics[2]>, IOTmpl, metrics[2]>(
        common_param, quantizer_json, io_json, error);
}

TEST_CASE("sq8", "[ut][flatten_data_cell]") {
    auto allocator = std::make_shared<SafeAllocator>(DefaultAllocator::Instance());
    auto sq8_param = JsonType::parse("{}");
    auto io_param = JsonType::parse("{}");
    auto dims = {32, 64, 512};
    auto error = 2e-2f;
    for (auto dim : dims) {
        IndexCommonParam common_param;
        common_param.dim_ = dim;
        common_param.allocator_ = allocator;
        common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
        TestFlattenDataCellSQ8<MemoryIO>(common_param, sq8_param, io_param, error);
        TestFlattenDataCellSQ8<MemoryBlockIO>(common_param, sq8_param, io_param, error);
    }
}
