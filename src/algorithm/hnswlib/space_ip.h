
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
#include "data_type.h"
#include "simd/simd.h"
#include "space_interface.h"

namespace hnswlib {

class InnerProductSpace : public SpaceInterface {
    DISTFUNC fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    explicit InnerProductSpace(size_t dim, vsag::DataTypes type) {
        dim_ = dim;
        if (type == vsag::DataTypes::DATA_TYPE_FLOAT) {
            fstdistfunc_ = vsag::GetInnerProductDistanceFunc(dim);
            data_size_ = dim * sizeof(float);
        } else if (type == vsag::DataTypes::DATA_TYPE_INT8) {
            fstdistfunc_ = vsag::GetINT8InnerProductDistanceFunc(dim);
            data_size_ = dim * sizeof(int8_t);
        } else {
            throw std::invalid_argument(
                fmt::format("no support for this metric: {}", datatype_to_str(type)));
        }
    }

    size_t
    get_data_size() override {
        return data_size_;
    }

    DISTFUNC
    get_dist_func() override {
        return fstdistfunc_;
    }

    void*
    get_dist_func_param() override {
        return &dim_;
    }

    ~InnerProductSpace() {
    }
};

}  // namespace hnswlib
