
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

#include "pyramid.h"

#include <iostream>

#include "../logger.h"
namespace vsag {

template <typename T>
using Deque = std::deque<T, vsag::AllocatorWrapper<T>>;

constexpr static const char PART_SLASH = '/';
std::vector<std::string>
split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    if (str.empty()) {
        throw std::runtime_error("fail to parse empty path");
    }

    while (end != std::string::npos) {
        std::string token = str.substr(start, end - start);
        if (token.empty()) {
            throw std::runtime_error("fail to parse path:" + str);
        }
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delimiter, start);
    }
    std::string lastToken = str.substr(start);
    if (lastToken.empty()) {
        throw std::runtime_error("fail to parse path:" + str);
    }
    tokens.push_back(str.substr(start, end - start));
    return tokens;
}

tl::expected<std::vector<int64_t>, Error>
Pyramid::Build(const DatasetPtr& base) {
    return this->Add(base);
}

tl::expected<std::vector<int64_t>, Error>
Pyramid::Add(const DatasetPtr& base) {
    auto path = base->GetPaths();
    int64_t data_num = base->GetNumElements();
    int64_t data_dim = base->GetDim();
    auto data_ids = base->GetIds();
    auto data_vectors = base->GetFloat32Vectors();
    for (int i = 0; i < data_num; ++i) {
        std::string current_path = path[i];
        auto result = split(current_path, PART_SLASH);
        if (indexes_.find(result[0]) == indexes_.end()) {
            indexes_[result[0]] = std::make_shared<IndexNode>(commom_param_.allocator_);
        }
        std::shared_ptr<IndexNode> node = indexes_.at(result[0]);
        DatasetPtr single_data = Dataset::Make();
        single_data->Owner(false)
            ->NumElements(1)
            ->Dim(data_dim)
            ->Float32Vectors(data_vectors + data_dim * i)
            ->Ids(data_ids + i);
        for (int j = 1; j < result.size(); ++j) {
            if (node->index) {
                node->index->Add(single_data);
            }
            if (node->children.find(result[j]) == node->children.end()) {
                node->children[result[j]] = std::make_shared<IndexNode>(commom_param_.allocator_);
            }
            node = node->children.at(result[j]);
        }
        node->CreateIndex(pyramid_param_.index_builder);
        node->index->Add(single_data);
    }
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::KnnSearch(const DatasetPtr& query,
                   int64_t k,
                   const std::string& parameters,
                   BitsetPtr invalid) const {
    auto path = query->GetPaths();

    std::string current_path = path[0];
    auto parsed_path = split(current_path, PART_SLASH);

    if (indexes_.find(parsed_path[0]) == indexes_.end()) {
        auto ret = Dataset::Make();
        ret->Dim(0)->NumElements(1);
        return ret;
    }
    std::shared_ptr<IndexNode> root = indexes_.at(parsed_path[0]);
    for (int j = 1; j < parsed_path.size(); ++j) {
        if (root->children.find(parsed_path[j]) == root->children.end()) {
            std::cout << "search:" << parsed_path[j] << std::endl;
            auto ret = Dataset::Make();
            ret->Dim(0)->NumElements(1);
            return ret;
        }
        root = root->children.at(parsed_path[j]);
    }
    Deque<std::shared_ptr<IndexNode>> candidate_indexes(commom_param_.allocator_);

    std::priority_queue<std::pair<float, int64_t>> results;
    candidate_indexes.push_back(root);
    while (not candidate_indexes.empty()) {
        auto node = candidate_indexes.front();
        candidate_indexes.pop_front();
        if (node->index) {
            auto result = node->index->KnnSearch(query, k, parameters);
            if (result.has_value()) {
                DatasetPtr r = result.value();
                for (int i = 0; i < r->GetDim(); ++i) {
                    results.emplace(r->GetDistances()[i], r->GetIds()[i]);
                }
            } else {
                auto error = result.error();
                LOG_ERROR_AND_RETURNS(error.type, error.message);
            }
        } else {
            for (const auto& item : node->children) {
                candidate_indexes.emplace_back(item.second);
            }
        }
        if (results.size() > k) {
            results.pop();
        }
    }

    // return result
    auto result = Dataset::Make();
    size_t target_size = results.size();
    if (results.size() == 0) {
        result->Dim(0)->NumElements(1);
        return result;
    }
    result->Dim(target_size)->NumElements(1)->Owner(true, commom_param_.allocator_);
    int64_t* ids = (int64_t*)commom_param_.allocator_->Allocate(sizeof(int64_t) * target_size);
    result->Ids(ids);
    float* dists = (float*)commom_param_.allocator_->Allocate(sizeof(float) * target_size);
    result->Distances(dists);
    for (int64_t j = results.size() - 1; j >= 0; --j) {
        if (j < target_size) {
            dists[j] = results.top().first;
            ids[j] = results.top().second;
        }
        results.pop();
    }
    return result;
}

tl::expected<DatasetPtr, Error>
Pyramid::KnnSearch(const DatasetPtr& query,
                   int64_t k,
                   const std::string& parameters,
                   const std::function<bool(int64_t)>& filter) const {
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::RangeSearch(const DatasetPtr& query,
                     float radius,
                     const std::string& parameters,
                     int64_t limited_size) const {
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::RangeSearch(const DatasetPtr& query,
                     float radius,
                     const std::string& parameters,
                     BitsetPtr invalid,
                     int64_t limited_size) const {
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::RangeSearch(const DatasetPtr& query,
                     float radius,
                     const std::string& parameters,
                     const std::function<bool(int64_t)>& filter,
                     int64_t limited_size) const {
    return {};
}

tl::expected<BinarySet, Error>
Pyramid::Serialize() const {
    return {};
}

tl::expected<void, Error>
Pyramid::Deserialize(const BinarySet& binary_set) {
    return {};
}

tl::expected<void, Error>
Pyramid::Deserialize(const ReaderSet& reader_set) {
    return {};
}

int64_t
Pyramid::GetNumElements() const {
    return 0;
}

int64_t
Pyramid::GetMemoryUsage() const {
    return 0;
}

}  // namespace vsag
