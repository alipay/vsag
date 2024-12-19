
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

#include "../logger.h"
namespace vsag {

// Function to convert BinarySet to a Binary
Binary
binaryset_to_binary(const BinarySet binarySet) {
    size_t totalSize = 0;
    auto keys = binarySet.GetKeys();

    for (const auto& key : keys) {
        totalSize += sizeof(size_t) + key.size();
        totalSize += sizeof(size_t);
        totalSize += binarySet.Get(key).size;
    }

    Binary result;
    result.data = std::shared_ptr<int8_t[]>(new int8_t[totalSize]);
    result.size = totalSize;

    size_t offset = 0;

    for (const auto& key : keys) {
        size_t keySize = key.size();
        memcpy(result.data.get() + offset, &keySize, sizeof(size_t));
        offset += sizeof(size_t);
        memcpy(result.data.get() + offset, key.data(), keySize);
        offset += keySize;

        Binary binary = binarySet.Get(key);
        memcpy(result.data.get() + offset, &binary.size, sizeof(size_t));
        offset += sizeof(size_t);
        memcpy(result.data.get() + offset, binary.data.get(), binary.size);
        offset += binary.size;
    }

    return result;
}

BinarySet
binary_to_binaryset(const Binary binary) {
    BinarySet binarySet;
    size_t offset = 0;

    while (offset < binary.size) {
        size_t keySize;
        memcpy(&keySize, binary.data.get() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        std::string key(reinterpret_cast<const char*>(binary.data.get() + offset), keySize);
        offset += keySize;

        size_t binarySize;
        memcpy(&binarySize, binary.data.get() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        Binary newBinary;
        newBinary.size = binarySize;
        newBinary.data = std::shared_ptr<int8_t[]>(new int8_t[binarySize]);
        memcpy(newBinary.data.get(), binary.data.get() + offset, binarySize);
        offset += binarySize;

        binarySet.Set(key, newBinary);
    }

    return binarySet;
}

ReaderSet
reader_to_readerset(std::shared_ptr<Reader> reader) {
    ReaderSet readerSet;
    size_t offset = 0;

    while (offset < reader->Size()) {
        size_t keySize;
        reader->Read(offset, sizeof(size_t), &keySize);
        offset += sizeof(size_t);
        std::shared_ptr<char[]> key_chars = std::shared_ptr<char[]>(new char[keySize]);
        reader->Read(offset, keySize, key_chars.get());
        std::string key(key_chars.get(), keySize);
        offset += keySize;

        size_t binarySize;
        reader->Read(offset, sizeof(size_t), &binarySize);
        offset += sizeof(size_t);

        auto newReader = std::shared_ptr<SubReader>(new SubReader(reader, offset, binarySize));
        offset += binarySize;

        readerSet.Set(key, newReader);
    }

    return readerSet;
}

template <typename T>
using Deque = std::deque<T, vsag::AllocatorWrapper<T>>;

constexpr static const char PART_SLASH = '/';
constexpr static const char PART_OCTOTHORPE = '#';
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
        if (node->index == nullptr) {
            node->CreateIndex(pyramid_param_.index_builder);
        }
        node->index->Add(single_data);
    }
    data_num_ += data_num;
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
            auto result = node->index->KnnSearch(query, k, parameters, invalid);
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
        while (results.size() > k) {
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
    BinarySet binary_set;
    for (const auto& root_index : indexes_) {
        std::string path = root_index.first;
        std::vector<std::pair<std::string, std::shared_ptr<IndexNode>>> need_serialize_indexes;
        need_serialize_indexes.emplace_back(path, root_index.second);
        while (not need_serialize_indexes.empty()) {
            auto [current_path, index_node] = need_serialize_indexes.back();
            need_serialize_indexes.pop_back();
            if (index_node->index) {
                auto serialize_result = index_node->index->Serialize();
                if (not serialize_result.has_value()) {
                    return tl::unexpected(serialize_result.error());
                }
                binary_set.Set(current_path, binaryset_to_binary(serialize_result.value()));
            }
            for (const auto& sub_index_node : index_node->children) {
                need_serialize_indexes.emplace_back(
                    current_path + PART_OCTOTHORPE + sub_index_node.first, sub_index_node.second);
            }
        }
    }
    return binary_set;
}

tl::expected<void, Error>
Pyramid::Deserialize(const BinarySet& binary_set) {
    auto keys = binary_set.GetKeys();
    for (const auto& path : keys) {
        const auto& binary = binary_set.Get(path);
        auto parsed_path = split(path, PART_OCTOTHORPE);
        if (indexes_.find(parsed_path[0]) == indexes_.end()) {
            indexes_[parsed_path[0]] = std::make_shared<IndexNode>(commom_param_.allocator_);
        }
        std::shared_ptr<IndexNode> node = indexes_.at(parsed_path[0]);
        for (int j = 1; j < parsed_path.size(); ++j) {
            if (node->children.find(parsed_path[j]) == node->children.end()) {
                node->children[parsed_path[j]] =
                    std::make_shared<IndexNode>(commom_param_.allocator_);
            }
            node = node->children.at(parsed_path[j]);
        }
        node->CreateIndex(pyramid_param_.index_builder);
        node->index->Deserialize(binary_to_binaryset(binary));
    }
    return {};
}

tl::expected<void, Error>
Pyramid::Deserialize(const ReaderSet& reader_set) {
    return {};
}

int64_t
Pyramid::GetNumElements() const {
    return data_num_;
}

int64_t
Pyramid::GetMemoryUsage() const {
    return 0;
}

}  // namespace vsag
