
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


//template<typename QuantTmpl, typename IOTmpl>
void TestFlattenDataCell(std::shared_ptr<FlattenInterface> inter, int dim, MetricType metric, float error = 1e-5) {

}


TEST_CASE("fp32[ut][flatten_data_cell]") {
    int dim = 32;
    auto alloctor = std::make_shared<DefaultAllocator>();
    {
        auto data_cell = std::make_shared<FlattenDataCell<FP32Quantizer<>, MemoryIO>>();
        data_cell->SetQuantizer(std::make_shared<FP32Quantizer<>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor.get()));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_L2SQR);
    }
    {
        auto data_cell = std::make_shared<
            FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_IP>, MemoryIO>>();
        data_cell->SetQuantizer(
            std::make_shared<FP32Quantizer<vsag::MetricType::METRIC_TYPE_IP>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor.get()));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_IP);
    }
    {
        auto data_cell = std::make_shared<
            FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>, MemoryIO>>();
        data_cell->SetQuantizer(
            std::make_shared<FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor.get()));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_COSINE);
    }
}

TEST_CASE("sq8[ut][flatten_data_cell]") {
    int dim = 32;
    float error = 0.03;
    auto alloctor = new DefaultAllocator();
    {
        auto data_cell = std::make_shared<FlattenDataCell<SQ8Quantizer<>, MemoryIO>>();
        data_cell->SetQuantizer(std::make_shared<SQ8Quantizer<>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_L2SQR, error);
    }
    {
        auto data_cell = std::make_shared<
            FlattenDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_IP>, MemoryIO>>();
        data_cell->SetQuantizer(
            std::make_shared<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_IP>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_IP, error);
    }
    {
        auto data_cell = std::make_shared<
            FlattenDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>, MemoryIO>>();
        data_cell->SetQuantizer(
            std::make_shared<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>>(dim));
        data_cell->SetIO(std::make_shared<MemoryIO>(alloctor));
        TestFlattenDataCell(data_cell, dim, vsag::MetricType::METRIC_TYPE_COSINE, error);
    }
    delete alloctor;
}
