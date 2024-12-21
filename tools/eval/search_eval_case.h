
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

#include "eval_case.h"
#include "monitor.h"

namespace eval {

class SearchEvalCase : public EvalCase {
public:
    using JsonType = Monitor::JsonType;

public:
    SearchEvalCase(const std::string& dataset_path,
                   const std::string& index_path,
                   vsag::IndexPtr index,
                   argparse::ArgumentParser& parser);

    ~SearchEvalCase() override = default;

    void
    Run() override;

private:
    enum SearchType {
        KNN,
        RANGE,
        KNN_FILTER,
        RANGE_FILTER,
    };

    void
    init_monitor();

    void
    init_latency_monitor();

    void
    init_recall_monitor();

    void
    init_memory_monitor();

    void
    deserialize();

    void
    do_knn_search();

    void
    do_range_search();

    void
    do_knn_filter_search();

    void
    do_range_filter_search();

    JsonType
    process_result();

private:
    std::vector<MonitorPtr> monitors_{};

    SearchType search_type_{SearchType::KNN};

    std::string search_param_{};

    argparse::ArgumentParser& parser_;
};
}  // namespace eval