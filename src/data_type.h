
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
#include <string>
namespace vsag {
enum class DataTypes { DATA_TYPE_FLOAT = 0, DATA_TYPE_INT8 = 1, DATA_TYPE_FP16 = 2 };

inline std::string
datatype_to_str(DataTypes type) {
    switch (type) {
        case DataTypes::DATA_TYPE_FLOAT:
            return "float32";
        case DataTypes::DATA_TYPE_INT8:
            return "int8";
        case DataTypes::DATA_TYPE_FP16:
            return "float16";
    }
}

}  // namespace vsag
