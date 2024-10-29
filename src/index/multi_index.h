
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
#include <iostream>

#include "../utils.h"
#include "safe_allocator.h"
#include "vsag/factory.h"
#include "vsag/index.h"

namespace vsag {

Binary
binaryset_to_binary(const BinarySet binarySet);
BinarySet
binary_to_binaryset(const Binary binary);
ReaderSet
reader_to_readerset(std::shared_ptr<Reader> reader);



class SubReader : public Reader {
public:
    SubReader(std::shared_ptr<Reader> parrent_reader, uint64_t start_pos, uint64_t size) : parrent_reader_(parrent_reader), size_(size), start_pos_(start_pos) {}

    void Read(uint64_t offset, uint64_t len, void* dest) override {
        if (offset + len > size_) throw std::out_of_range("Read out of range.");
        parrent_reader_->Read(offset + start_pos_, len, dest);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {

    }

    uint64_t Size() const override {
        return size_;
    }

private:
    std::shared_ptr<Reader> parrent_reader_;
    uint64_t size_;
    uint64_t start_pos_;
};



class MultiIndex : public Index {
public:
    MultiIndex(std::string sub_index_type, std::string build_params, Allocator* allocator = nullptr)
        : sub_index_type_(sub_index_type),
          build_params_(build_params),
          safe_allocator_(new SafeAllocator(allocator ? allocator : DefaultAllocator::Instance())),
          sub_indexes_(safe_allocator_.get()) {
    }

    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        std::vector<int64_t> failed_ids;
        auto tags = base->GetTags();
        int64_t data_num = base->GetNumElements();
        int64_t data_dim = base->GetDim();
        auto datas = base->GetFloat32Vectors();
        std::map<int64_t, std::vector<int64_t>> tag_lists;
        for (int i = 0; i < data_num; ++i) {
            tag_lists[tags[i]].push_back(i);
        }
        for (const auto& item : tag_lists) {
            auto current_tag = item.first;
            std::cout << current_tag << std::endl;
            auto new_index =
                Factory::CreateIndex(sub_index_type_, build_params_, safe_allocator_.get());
            if (not new_index.has_value()) {
                LOG_ERROR_AND_RETURNS(new_index.error().type, new_index.error().message);
            }
            sub_indexes_[current_tag] = new_index.value();
            const auto& current_data_ids = item.second;
            auto sub_data_num = current_data_ids.size();
            auto tmp_vector_memory = std::shared_ptr<float[]>(new float[sub_data_num * data_dim]);

            for (int i = 0; i < current_data_ids.size(); ++i) {
                memcpy(tmp_vector_memory.get() + i * data_dim, datas + data_dim * current_data_ids[i], data_dim * sizeof(float));
            }

            DatasetPtr sub_data = Dataset::Make();
            sub_data->Owner(false)
                ->Ids(current_data_ids.data())
                ->Float32Vectors(tmp_vector_memory.get())
                ->Dim(data_dim)
                ->NumElements(sub_data_num);
            auto insert_result = sub_indexes_[current_tag]->Build(sub_data);
            if (not insert_result.has_value()) {
                LOG_ERROR_AND_RETURNS(insert_result.error().type, insert_result.error().message);
            }
        }
        return failed_ids;
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        auto tags = query->GetTags();
        int64_t cur_tag = tags[0];
        if (sub_indexes_.find(cur_tag) != sub_indexes_.end()) {
            auto result = sub_indexes_[cur_tag]->KnnSearch(query, k, parameters, invalid);
            if (result.has_value()) {
                auto knn_results = result.value();
                return std::move(knn_results);
            } else {
                logger::error(result.error().message);
                LOG_ERROR_AND_RETURNS(result.error().type, result.error().message);
            }
        }
        return Dataset::Make();
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        auto tags = query->GetTags();
        int64_t cur_tag = tags[0];
        if (sub_indexes_.find(cur_tag) != sub_indexes_.end()) {
            auto result = sub_indexes_[cur_tag]->KnnSearch(query, k, parameters, filter);
            if (result.has_value()) {
                auto knn_results = result.value();
                return std::move(knn_results);
            } else {
                logger::error(result.error().message);
                LOG_ERROR_AND_RETURNS(result.error().type, result.error().message);
            }
        }
        return Dataset::Make();
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        return {};
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        return {};
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        return {};
    }

public:
    tl::expected<BinarySet, Error>
    Serialize() const override {
        BinarySet binary_set;
        for (const auto& item : sub_indexes_) {
            auto key = std::to_string(item.first);
            auto serialize_result = item.second->Serialize();
            BinarySet value = serialize_result.value();
            auto all_binary = binaryset_to_binary(value);
            binary_set.Set(key, all_binary);
        }
        return binary_set;
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        auto str_keys = binary_set.GetKeys();
        std::cout << str_keys.size() << std::endl;
        for (int i = 0; i < str_keys.size(); ++i) {
            auto key = std::stoll(str_keys[i]);
            auto new_index =
                Factory::CreateIndex(sub_index_type_, build_params_, safe_allocator_.get());
            if (not new_index.has_value()) {
                LOG_ERROR_AND_RETURNS(new_index.error().type, new_index.error().message);
            }
            sub_indexes_[key] = new_index.value();
            BinarySet sub_binary_set = binary_to_binaryset(binary_set.Get(str_keys[i]));
            sub_indexes_[key]->Deserialize(sub_binary_set);
        }
        return {};
    }

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        auto str_keys = reader_set.GetKeys();
        std::cout << str_keys.size() << std::endl;
        for (int i = 0; i < str_keys.size(); ++i) {
            auto key = std::stoll(str_keys[i]);
            auto new_index =
                Factory::CreateIndex(sub_index_type_, build_params_, safe_allocator_.get());
            if (not new_index.has_value()) {
                LOG_ERROR_AND_RETURNS(new_index.error().type, new_index.error().message);
            }
            sub_indexes_[key] = new_index.value();
            std::cout << "deserialize:" << key << std::endl;
            auto reader = reader_set.Get(str_keys[i]);
            auto sub_reader_set = reader_to_readerset(reader);
            sub_indexes_[key]->Deserialize(sub_reader_set);
        }
        return {};
    }

public:
    int64_t
    GetNumElements() const override {
        return 0;
    }

    int64_t
    GetMemoryUsage() const override {
        return 0;
    }

private:
    std::string sub_index_type_;
    std::string build_params_;

    std::shared_ptr<Allocator> safe_allocator_;
    mutable UnorderedMap<int64_t, std::shared_ptr<Index>> sub_indexes_;
};

}  // namespace vsag
