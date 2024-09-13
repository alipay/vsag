
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
#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "flatten_storage.h"
#include "graph_datacell.h"

namespace vsag {

enum class REDUNDANT_STRATEGY { ID_FIRST = 0, RANDOM_FIRST = 1, DEGREE_FIRST = 2 };

template <typename QuantTmpl, typename IOTmpl>
class MixDataCell : public FlattenStorage<QuantTmpl, IOTmpl> {
public:
    MixDataCell(std::shared_ptr<GraphDataCell<IOTmpl>> graph_data_cell)
        : FlattenStorage<QuantTmpl, IOTmpl>() {
        this->graph_data_cell_ = graph_data_cell;
    };

    explicit MixDataCell(const std::string& initializeJson);  // todo

    void
    MakeRedundant(float loading_factor = 1.0,
                  REDUNDANT_STRATEGY redundant_strategy = REDUNDANT_STRATEGY::ID_FIRST);

    template <typename IDType = uint64_t>
    void
    QueryLine(float* resultDists,
              const float* queryVector,
              uint64_t id);  // todo: add mask for visited

    template <typename IDType = uint64_t>
    void
    QueryLine(float* resultDists, std::shared_ptr<Computer<QuantTmpl>>& computer, uint64_t id);

    inline void
    SetIO(std::unique_ptr<BasicIO<IOTmpl>>& io) = delete;

    inline void
    SetIO(std::unique_ptr<BasicIO<IOTmpl>>&& io) = delete;

    inline void
    SetIO(std::unique_ptr<BasicIO<IOTmpl>>& io, std::unique_ptr<BasicIO<IOTmpl>>& redundant_io) {
        this->io_ = std::move(io);
        this->redundant_io_ = std::move(redundant_io);
    }

    inline void
    SetIO(std::unique_ptr<BasicIO<IOTmpl>>&& io, std::unique_ptr<BasicIO<IOTmpl>>&& redundant_io) {
        this->io_ = std::move(io);
        this->redundant_io_ = std::move(redundant_io);
    }

private:
    std::vector<uint64_t>
    SelectIds(float loading_factor, REDUNDANT_STRATEGY redundant_strategy);

    void
    RedundantInsertVector(uint64_t id,
                          const std::vector<uint64_t> neighbor_ids,
                          const std::vector<const uint8_t*> neighbor_codes);

    const uint64_t
    GetRedundantSizeById(uint64_t id) const;

    const uint8_t*
    GetNeighborCodesById(uint64_t id, uint64_t neighbor_j) const;

private:
    std::shared_ptr<GraphDataCell<IOTmpl>> graph_data_cell_;

    std::shared_ptr<BasicIO<IOTmpl>> redundant_io_{nullptr};

    uint64_t redundant_cur_offset_{0};
    std::unordered_map<uint64_t, uint64_t> redundant_offset_;
    uint64_t redundant_total_count_{0};
};

template <typename QuantTmpl, typename IOTmpl>
void
MixDataCell<QuantTmpl, IOTmpl>::MakeRedundant(float loading_factor,
                                              vsag::REDUNDANT_STRATEGY redundant_strategy) {
    std::vector<uint64_t> neighbor_ids;
    std::vector<const uint8_t*> neighbor_codes;
    auto selected_ids = SelectIds(loading_factor, redundant_strategy);
    for (auto id : selected_ids) {
        uint32_t size = graph_data_cell_->GetNeighborSize(id);
        graph_data_cell_->GetNeighbors(neighbor_ids);
        neighbor_codes.resize(size);
        for (uint32_t i = 0; i < size; i++) {
            neighbor_codes[i] = this->GetCodesById(neighbor_ids[i]);
        }
        RedundantInsertVector(id, neighbor_ids, neighbor_codes);
    }
}

template <typename QuantTmpl, typename IOTmpl>
std::vector<uint64_t>
MixDataCell<QuantTmpl, IOTmpl>::SelectIds(float loading_factor,
                                          REDUNDANT_STRATEGY redundant_strategy) {
    uint64_t data_size = graph_data_cell_.TotalCount();
    std::vector<uint64_t> ids(data_size);
    std::vector<uint64_t> selected_ids(loading_factor * data_size);
    std::iota(ids.begin(), ids.end(), 0);

    if (redundant_strategy == REDUNDANT_STRATEGY::DEGREE_FIRST) {
        std::vector<std::pair<uint64_t, uint32_t>> id_degree;
        for (uint64_t i = 0; i < ids.size(); i++) {
            id_degree.push_back(std::make_pair(ids[i], graph_data_cell_.GetNeighborSize(ids[i])));
        }
        std::sort(id_degree.begin(),
                  id_degree.end(),
                  [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                      return a.second > b.second;
                  });

        for (uint64_t i = 0; i < loading_factor * data_size; i++) {
            selected_ids[i] = id_degree[i].first;
        }
    } else if (redundant_strategy == REDUNDANT_STRATEGY::RANDOM_FIRST) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(ids.begin(), ids.end(), gen);
        selected_ids.assign(ids.begin(), ids.begin() + loading_factor * data_size);
    } else {
        selected_ids.assign(ids.begin(), ids.begin() + loading_factor * data_size);
    }

    return selected_ids;
}

template <typename QuantTmpl, typename IOTmpl>
void
MixDataCell<QuantTmpl, IOTmpl>::RedundantInsertVector(
    uint64_t id,
    const std::vector<uint64_t> neighbor_ids,
    const std::vector<const uint8_t*> neighbor_codes) {
    redundant_offset_[id] = redundant_cur_offset_;

    uint64_t size = neighbor_codes.size();

    redundant_io_->Write(&size, sizeof(size), redundant_cur_offset_);
    redundant_cur_offset_ += sizeof(size);

    for (uint64_t i = 0; i < size; i++) {
        redundant_io_->Write(neighbor_ids[i], sizeof(neighbor_ids[i]), redundant_cur_offset_);
        redundant_cur_offset_ += sizeof(neighbor_ids[i]);

        redundant_io_->Write(neighbor_codes[i], this->codeSize_, redundant_cur_offset_);
        redundant_cur_offset_ += this->codeSize_;
    }
}

template <typename QuantTmpl, typename IOTmpl>
const uint64_t
MixDataCell<QuantTmpl, IOTmpl>::GetRedundantSizeById(uint64_t id) const {
    auto iter = redundant_offset_.find(id);
    if (iter == redundant_offset_.end()) {
        return 0;
    }
    return *(uint64_t*)(redundant_io_->Read(sizeof(uint64_t), iter->second));
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
MixDataCell<QuantTmpl, IOTmpl>::GetNeighborCodesById(uint64_t id, uint64_t neighbor_i) const {
    auto iter = redundant_offset_.find(id);
    if (iter == redundant_offset_.end()) {
        return nullptr;
    }
    auto pos = iter->second + +neighbor_i * (this->codeSize_ + sizeof(uint64_t));
    return *(uint64_t*)(this->io_->Read(this->codeSize_, pos));
}

template <typename QuantTmpl, typename IOTmpl>
template <typename IDType>
void
MixDataCell<QuantTmpl, IOTmpl>::QueryLine(float* resultDists,
                                          const float* queryVector,
                                          uint64_t id) {
    std::shared_ptr<Computer<QuantTmpl>> computer = std::move(this->quantizer_->FactoryComputer());
    computer->SetQuery(queryVector);
    this->QueryLine(resultDists, computer, id);
}

template <typename QuantTmpl, typename IOTmpl>
template <typename IDType>
void
MixDataCell<QuantTmpl, IOTmpl>::QueryLine(float* resultDists,
                                          std::shared_ptr<Computer<QuantTmpl>>& computer,
                                          uint64_t id) {
    auto iter = redundant_offset_.find(id);
    if (iter == redundant_offset_.end()) {
        std::vector<uint64_t> neighbor_ids;
        graph_data_cell_->GetNeighbors(id, neighbor_ids);
        for (int i = 0; i < neighbor_ids.size(); i++) {
            const auto* codes = this->GetCodesById(id);
            computer->ComputeDist(codes, resultDists + i);
        }
    } else {
        auto neighbor_size = GetRedundantSizeById(id);
        for (int i = 0; i < neighbor_size; i++) {
            const auto* codes = this->GetNeighborCodesById(id, i);
            computer->ComputeDist(codes, resultDists + i);
        }
    };
}

}  // namespace vsag