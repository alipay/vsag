
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

#include "build_eval_case.h"

#include <algorithm>

#include "duration_monitor.h"
#include "memory_peak_monitor.h"

namespace eval {

BuildEvalCase::BuildEvalCase(const std::string& dataset_path,
                             const std::string& index_path,
                             vsag::IndexPtr index,
                             argparse::ArgumentParser& parser)
    : EvalCase(dataset_path, index_path, index), parser_(parser) {
    this->init_monitors();
}

void
BuildEvalCase::init_monitors() {
    if (parser_.get<bool>("--memory")) {
        auto memory_peak_monitor = std::make_shared<MemoryPeakMonitor>();
        this->monitors_.emplace_back(std::move(memory_peak_monitor));
    }
    if (parser_.get<bool>("--tps")) {
        auto duration_monitor = std::make_shared<DurationMonitor>();
        this->monitors_.emplace_back(std::move(duration_monitor));
    }
}

void
BuildEvalCase::Run() {
    this->do_build();
    this->serialize();
    auto result = this->process_result();
    this->PrintResult(result);
}
void
BuildEvalCase::do_build() {
    auto base = vsag::Dataset::Make();
    int64_t total_base = this->dataset_ptr_->GetNumberOfBase();
    std::vector<int64_t> ids(total_base);
    std::iota(ids.begin(), ids.end(), 0);
    base->NumElements(total_base)->Dim(this->dataset_ptr_->GetDim())->Ids(ids.data())->Owner(false);
    if (this->dataset_ptr_->GetTrainDataType() == vsag::DATATYPE_FLOAT32) {
        base->Float32Vectors((const float*)this->dataset_ptr_->GetTrain());
    } else if (this->dataset_ptr_->GetTrainDataType() == vsag::DATATYPE_INT8) {
        base->Int8Vectors((const int8_t*)this->dataset_ptr_->GetTrain());
    }
    for (auto& monitor : monitors_) {
        monitor->Start();
    }
    auto build_index = index_->Build(base);
    for (auto& monitor : monitors_) {
        monitor->Record();
        monitor->Stop();
    }
}
void
BuildEvalCase::serialize() {
    std::ofstream outfile(this->index_path_, std::ios::binary);
    this->index_->Serialize(outfile);
}

EvalCase::JsonType
BuildEvalCase::process_result() {
    JsonType result;
    JsonType eval_result;
    for (auto& monitor : this->monitors_) {
        const auto& one_result = monitor->GetResult();
        EvalCase::MergeJsonType(one_result, eval_result);
    }
    result = eval_result;
    result["tps"] = double(this->dataset_ptr_->GetNumberOfBase()) / double(result["duration(s)"]);
    EvalCase::MergeJsonType(this->basic_info_, result);
    result["index_info"] = JsonType::parse(parser_.get<std::string>("--create_params"));
    return result;
}

}  // namespace eval
