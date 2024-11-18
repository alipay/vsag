
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

#include "hgraph_zparameters.h"

#include "../utils.h"
#include "fmt/format-inl.h"
#include "inner_string_params.h"
#include "vsag/constants.h"

namespace vsag {

const std::unordered_map<std::string, std::vector<std::string>> HGraphParameters::EXTERNAL_MAPPING =
    {{HGRAPH_USE_REORDER, {HGRAPH_USE_REORDER_KEY}},
     {HGRAPH_BASE_QUANTIZATION_TYPE, {HGRAPH_BASE_CODES_KEY, QUANTIZATION_TYPE_KEY}},
     {HGRAPH_GRAPH_MAX_DEGREE, {HGRAPH_GRAPH_KEY, GRAPH_PARAMS_KEY, GRAPH_PARAM_MAX_DEGREE}}};

HGraphParameters::HGraphParameters(const IndexCommonParam& common_param, JsonType& hgraph_param)
    : common_param_(common_param) {
    this->check_common_param();
    this->refresh_json_by_string();
    this->ParseStringParam(hgraph_param);
    this->refresh_string_by_json();
}

void
HGraphParameters::check_common_param() const {
    if (this->common_param_.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw std::invalid_argument(fmt::format("HGraph not support {} datatype", DATATYPE_INT8));
    }
}

void
HGraphParameters::ParseStringParam(JsonType& hgraph_param) {
    for (const auto& [key, value] : hgraph_param.items()) {
        this->CheckAndSetKeyValue(key, value);
    }
    this->refresh_string_by_json();
}

void
HGraphParameters::CheckAndSetKeyValue(const std::string& key, JsonType& value) {
    const auto& iter = EXTERNAL_MAPPING.find(key);

    if (key == HGRAPH_BASE_QUANTIZATION_TYPE) {
        std::string value_str = value;
        if (value_str != QUANTIZATION_TYPE_VALUE_SQ8 && value_str != QUANTIZATION_TYPE_VALUE_FP32) {
            throw std::invalid_argument(fmt::format("parameters[{}] must be {} or {}, now is {}",
                                                    HGRAPH_BASE_QUANTIZATION_TYPE,
                                                    QUANTIZATION_TYPE_VALUE_SQ8,
                                                    QUANTIZATION_TYPE_VALUE_FP32,
                                                    value_str));
        }
    }

    if (iter != EXTERNAL_MAPPING.end()) {
        const auto& vec = iter->second;
        auto* json = &json_;
        for (const auto& str : vec) {
            json = &(json->operator[](str));
        }
        *json = value;
    } else {
        throw std::invalid_argument(fmt::format("HGraph have no config param: {}", key));
    }
}

const std::string HGraphParameters::DEFAULT_HGRAPH_PARAMS = format_map(
    R"(
    {
        "{HGRAPH_USE_REORDER_KEY}": false,
        "{HGRAPH_GRAPH_KEY}": {
            "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
            "{IO_PARAMS_KEY}": {
                "{BLOCK_IO_BLOCK_SIZE_KEY}": 134217728
            },
            "type": "nsw",
            "{GRAPH_PARAMS_KEY}": {
                "{GRAPH_PARAM_MAX_DEGREE}": 64,
                "{GRAPH_PARAM_INIT_MAX_CAPACITY}": 2000000
            }
        },
        "{HGRAPH_BASE_CODES_KEY}": {
            "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
            "{IO_PARAMS_KEY}": {
                "{BLOCK_IO_BLOCK_SIZE_KEY}": 134217728
            },
            "codes_type": "flatten_codes",
            "codes_param": {},
            "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_PQ}",
            "{QUANTIZATION_PARAMS_KEY}": {
                "subspace": 64,
                "nbits": 8
            }
        },
        "precise_codes": {
            "{IO_TYPE_KEY}": "aio_ssd",
            "{IO_PARAMS_KEY}": {},
            "codes_type": "flatten_codes",
            "codes_param": {},
            "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_SQ8}",
            "{QUANTIZATION_PARAMS_KEY}": {}
        },
        "build_params": {
            "ef_construction": 400,
            "{BUILD_THREAD_COUNT}": 5
        }
    })",
    DEFAULT_MAP);

}  // namespace vsag
