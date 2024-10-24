
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

#pragma once

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_set>

#include "fixtures/fixtures.h"
#include "fixtures/test_dataset.h"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/logger.h"
#include "vsag/options.h"
#include "vsag/vsag.h"

namespace vsag {

IndexPtr
TestFactory(const std::string& name, const std::string& build_param, bool expect_success = true);

void
TestBuildIndex(IndexPtr index, int64_t dim, const std::string& index_name = "no_name");

void
TestAddIndex(IndexPtr index, int64_t dim, const std::string& index_name = "no_name");

void
TestContinueAdd(IndexPtr index,
                int64_t dim,
                int64_t count = 100,
                const std::string& index_name = "no_name");

void
TestKnnSearch(IndexPtr index,
              std::shared_ptr<fixtures::TestDataset> dataset,
              const std::string& search_param,
              int topk = 10,
              float recall = 0.99);

}  // namespace vsag