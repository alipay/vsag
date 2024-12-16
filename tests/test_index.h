
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
#include <utility>

#include "fixtures/fixtures.h"
#include "fixtures/test_dataset.h"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/logger.h"
#include "vsag/options.h"
#include "vsag/vsag.h"

namespace fixtures {
class TestIndex {
public:
    using IndexPtr = vsag::IndexPtr;
    using DatasetPtr = vsag::DatasetPtr;

    enum class IndexStatus {
        Init = 0,
        Factory = 1,
        Build = 2,
        DeSerialize = 3,
    };

    TestIndex() {
        vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);
    }

protected:
    IndexPtr
    FastGeneralTest(const std::string& name,
                    const std::string& build_param,
                    const std::string& search_parameters,
                    const std::string& metric_type,
                    int64_t dim,
                    IndexStatus end_status = IndexStatus::DeSerialize) const;

    static IndexPtr
    TestFactory(const std::string& name,
                const std::string& build_param,
                bool expect_success = true);

    void
    TestBuildIndex(const IndexPtr& index, int64_t dim, bool expected_success = true) const;

    static void
    TestBuildIndex(const IndexPtr& index, const DatasetPtr& dataset, bool expected_success = true);

    void
    TestAddIndex(const IndexPtr& index, int64_t dim, bool expected_success = true) const;

    static void
    TestAddIndex(const IndexPtr& index, const DatasetPtr& dataset, bool expected_success = true);

    static void
    TestContinueAdd(const IndexPtr& index,
                    int64_t dim,
                    int64_t count = 100,
                    bool expected_success = true);

    static void
    TestKnnSearch(const IndexPtr& index,
                  const TestDatasetPtr& dataset,
                  const std::string& search_param,
                  int topk = 10,
                  float recall = 0.99,
                  bool expected_success = true);

    static void
    TestRangeSearch(const IndexPtr& index,
                    const TestDatasetPtr& dataset,
                    const std::string& search_param,
                    float radius = 0.01,
                    float recall = 0.99,
                    int64_t limited_size = -1,
                    bool expected_success = true);
    static void
    TestCalcDistanceById(const IndexPtr& index,
                         const TestDatasetPtr& dataset,
                         const std::string& metric_str,
                         float error = 1e-5);

    static void
    TestSerializeFile(const IndexPtr& index, const std::string& path, bool expected_success = true);

    static IndexPtr
    TestDeserializeFile(const std::string& path,
                        const std::string& name,
                        const std::string& param,
                        bool expected_success = true);

    static bool
    SetDataset(const std::string& key, TestDatasetPtr value) {
        if (datasets.find(key) != datasets.end()) {
            return false;
        }
        datasets[key] = std::move(value);
        return true;
    }

    static TestDatasetPtr
    GetDataset(const std::string& key) {
        auto iter = datasets.find(key);
        if (iter == datasets.end()) {
            return nullptr;
        }
        return iter->second;
    }

    static void
    DeleteDataset(const std::string& key) {
        auto iter = datasets.find(key);
        if (iter != datasets.end()) {
            datasets.erase(iter);
        }
    }

    template <typename T>
    static TestDatasetPtr
    GenerateAndSetDataset(int64_t dim, int64_t count, bool with_path = false) {
        std::string datatype = "float";
        if constexpr (std::is_same_v<T, int8_t>) {
            datatype = "int8";
        } else if constexpr (std::is_same_v<T, float>) {
            datatype = "float";
        } else {
            // TODO throw;
        }
        auto key = KeyGen(dim, count, datatype, "classic", with_path);
        if (datasets.find(key) == datasets.end()) {
            datasets[key] = GenerateDataset<T>(dim, count, with_path);
        }

        return datasets[key];
    }

    template <typename T>
    static TestDatasetPtr
    GenerateDataset(int64_t dim, int64_t count, bool with_path) {
        if constexpr (std::is_same_v<T, int8_t>) {
            return GenerateDatasetInt8(dim, count, with_path);
        } else if constexpr (std::is_same_v<T, float>) {
            return GenerateDatasetFloat(dim, count, with_path);
        } else {
            // TODO throw;
            return nullptr;
        }
    }

    static TestDatasetPtr
    GenerateDatasetFloat(int64_t dim, int64_t count, bool with_path);

    static TestDatasetPtr
    GenerateDatasetInt8(int64_t dim, int64_t count, bool with_path) {
        return nullptr;  // TODO
    };

    static std::string
    KeyGen(int64_t dim,
           uint64_t count,
           std::string datatype = "float",
           std::string name = "classic",
           bool with_path = false) {
        return fmt::format("[dim={}][count={}][type={}][dataset_name={}][with_path={}]",
                           dim,
                           count,
                           datatype,
                           name,
                           with_path);
    }

    static std::string
    KeyGenIndex(int64_t dim,
                uint64_t count,
                std::string index_name,
                std::string datatype = "float",
                std::string dataset_name = "classic",
                bool with_path = false) {
        auto str = KeyGen(dim, count, std::move(datatype), std::move(dataset_name), with_path);
        return str + fmt::format("[index_name={}]", index_name);
    }

    std::pair<IndexPtr, IndexStatus>
    LoadIndex(const std::string& key) const {
        if (indexes.find(key) == indexes.end()) {
            return {nullptr, IndexStatus::Init};
        }
        return indexes[key];
    }

    void
    SaveIndex(const std::string& key, const IndexPtr& index, IndexStatus status) const {
        indexes[key] = {index, status};
    }

    mutable int dataset_base_count{1000};

private:
    static std::unordered_map<std::string, TestDatasetPtr> datasets;

    mutable std::unordered_map<std::string, std::pair<IndexPtr, IndexStatus>> indexes;
};

}  // namespace fixtures
