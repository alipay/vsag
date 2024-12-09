
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

#include "pyramid_zparameters.h"
#include "safe_allocator.h"
#include "typing.h"
#include "vsag/vsag.h"

namespace vsag {

struct IndexNode {
    std::shared_ptr<Index> index{nullptr};
    UnorderedMap<std::string, std::shared_ptr<IndexNode>> children;
    IndexNode(Allocator* allocator, IndexBuildFunction func = nullptr) : children(allocator) {
        if (nullptr != func) {
            index = func();
        }
    }
};

class Pyramid : public Index {
public:
    Pyramid(PyramidParameters pyramid_param, const IndexCommonParam commom_param)
        : indexes_(commom_param.allocator_),
          pyramid_param_(std::move(pyramid_param)),
          commom_param_(std::move(commom_param)) {
    }

    ~Pyramid() = default;

    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override;

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) override;

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override;

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override;

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override;

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override;

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override;

    tl::expected<BinarySet, Error>
    Serialize() const override;

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override;

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override;

    int64_t
    GetNumElements() const override;

    int64_t
    GetMemoryUsage() const override;

private:
    UnorderedMap<std::string, std::shared_ptr<IndexNode>> indexes_;
    PyramidParameters pyramid_param_;
    IndexCommonParam commom_param_;
};

}  // namespace vsag
