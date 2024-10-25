
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

#include "multi_index_zparameters.h"

#include <fmt/format-inl.h>

#include <nlohmann/json.hpp>

#include "../common.h"
#include "vsag/constants.h"
#include "vsag/factory.h"

namespace vsag {

CreateMultiIndexParameters
CreateMultiIndexParameters::FromJson(const std::string& json_string) {
    nlohmann::json params = nlohmann::json::parse(json_string);

    CHECK_ARGUMENT(params.contains(PARAMETER_INDEX_TYPE),
                   fmt::format("parameters must contains {}", PARAMETER_INDEX_TYPE));

    CreateMultiIndexParameters obj;
    obj.subindex_type = params[PARAMETER_INDEX_TYPE];
    obj.parameters = json_string;
    if (auto result = Factory::CreateIndex(obj.subindex_type, obj.parameters);
        not result.has_value()) {
        throw std::invalid_argument(result.error().message);
    }
    return obj;
}

}  // namespace vsag
