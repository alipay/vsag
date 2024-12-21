
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

#include <iostream>
#include <string>

#include "argparse/argparse.hpp"
#include "eval/eval_case.h"

void
CheckArgs(argparse::ArgumentParser& parser) {
    auto mode = parser.get<std::string>("--type");
    if (mode == "search") {
        auto search_mode = parser.get<std::string>("--search_params");
        if (search_mode.empty()) {
            throw std::runtime_error(R"(When "--type" is "search", "--search_params" is required)");
        }
    }
}

void
ParseArgs(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument<std::string>("--datapath", "-d")
        .required()
        .help("The hdf5 file path for eval");
    parser.add_argument<std::string>("--type", "-t")
        .required()
        .choices("build", "search")
        .help(R"(The eval method to select, choose from {"build", "search"})");
    parser.add_argument<std::string>("--index_name", "-n")
        .required()
        .help("The name of index fot create index");
    parser.add_argument<std::string>("--create_params", "-c")
        .required()
        .help("The param for create index");
    parser.add_argument<std::string>("--indexpath", "-i")
        .default_value("/tmp/performance/index")
        .help("The index path for load or save");
    parser.add_argument<std::string>("--search_params", "-s")
        .default_value("")
        .help("The param for search");
    parser.add_argument<std::string>("--search_mode")
        .default_value("knn")
        .choices("knn", "range", "knn_filter", "range_filter")
        .help(
            "The mode supported while use 'search' type,"
            " choose from {\"knn\", \"range\", \"knn_filter\", \"range_filter\"}");
    parser.add_argument("--topk")
        .default_value(10)
        .help("The topk value for knn search or knn_filter search")
        .scan<'i', int>();
    parser.add_argument("--range")
        .default_value(0.5f)
        .help("The range value for range search or range_filter search")
        .scan<'f', float>();
    parser.add_argument("--recall").default_value(true).help("Enable average recall eval");
    parser.add_argument("--percent_recall")
        .default_value(true)
        .help("Enable percent recall eval, include p0, p10, p30, p50, p70, p90");
    parser.add_argument("--qps").default_value(true).help("Enable qps eval");
    parser.add_argument("--tps").default_value(true).help("Enable tps eval");
    parser.add_argument("--memory").default_value(true).help("Enable memory eval");
    parser.add_argument("--latency").default_value(true).help("Enable average latency eval");
    parser.add_argument("--percent_latency")
        .default_value(true)
        .help("Enable percent latency eval, include p50, p80, p90, p95, p99");

    try {
        parser.parse_args(argc, argv);
        CheckArgs(parser);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
    }
}

int
main(int argc, char** argv) {
    argparse::ArgumentParser program("eval_performance");
    ParseArgs(program, argc, argv);
    auto eval_case = eval::EvalCase::MakeInstance(program);
    eval_case->Run();
}
