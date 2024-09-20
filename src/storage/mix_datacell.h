
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
template <typename QuantTmpl, typename IOTmpl>
class MixDataCell : public FlattenStorage<QuantTmpl, IOTmpl> {
public:
    MixDataCell(std::shared_ptr<GraphDataCell<IOTmpl>> graph_data_cell)
        : FlattenStorage<QuantTmpl, IOTmpl>() {
        this->graph_data_cell_ = graph_data_cell;
    };

    explicit MixDataCell(const std::string& initializeJson);  // todo

    void
    MakeRedundant(double loading_factor = 1.0);

    template <typename IDType = uint64_t>
    void
    QueryLine(float* resultDists,
              const float* queryVector,
              uint64_t id,
              std::vector<uint32_t>& to_be_visit,
              uint32_t count_no_visit);

    template <typename IDType = uint64_t>
    void
    QueryLine(float* resultDists,
              std::unique_ptr<Computer<QuantTmpl>>& computer,
              uint64_t id,
              std::vector<uint32_t>& to_be_visit,
              uint32_t count_no_visit);

    void
    SetIO(std::unique_ptr<BasicIO<IOTmpl>>& io) = delete;

    void
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

    inline void
    SetPrefetchParameters(uint32_t neighbor_codes_num, uint32_t cache_line) {
        prefetch_neighbor_codes_num = neighbor_codes_num;
        prefetch_cache_line = cache_line;
    }

    inline uint64_t
    GetRedundantTotalCount() const {
        return redundant_total_count_;
    }

private:
    void
    redundant_insert_neighbors(uint64_t id,
                               std::vector<uint64_t> neighbor_ids,
                               std::vector<const uint8_t*> neighbor_codes);

    uint64_t
    get_codes_offset_by_id(uint64_t id) const;

    uint64_t
    get_redundant_size_by_id(uint64_t id) const;

    uint64_t
    get_neighbor_codes_offset_by_id(uint64_t id, uint64_t neighbor_j) const;

    uint64_t
    get_neighbor_id_by_id(uint64_t id, uint64_t neighbor_i) const;

private:
    std::shared_ptr<GraphDataCell<IOTmpl>> graph_data_cell_;

    std::shared_ptr<BasicIO<IOTmpl>> redundant_io_{nullptr};

    uint64_t redundant_cur_offset_{0};

    uint64_t redundant_total_count_{0};

    uint32_t prefetch_neighbor_codes_num{0};

    uint32_t prefetch_cache_line{0};
};

template <typename QuantTmpl, typename IOTmpl>
void
MixDataCell<QuantTmpl, IOTmpl>::MakeRedundant(double loading_factor) {
    std::vector<uint64_t> neighbor_ids;
    std::vector<const uint8_t*> neighbor_codes;
    redundant_total_count_ = loading_factor * this->TotalCount();

    uint64_t tmp = 0;
    for (uint64_t id = 0; id < redundant_total_count_; id++) {
        redundant_io_->Write(
            reinterpret_cast<uint8_t*>(&tmp), sizeof(uint64_t), redundant_cur_offset_);
        redundant_cur_offset_ += sizeof(uint64_t);
    }

    for (uint64_t id = 0; id < redundant_total_count_; id++) {
        uint32_t size = graph_data_cell_->GetNeighborSize(id);
        graph_data_cell_->GetNeighbors(id, neighbor_ids);
        neighbor_codes.resize(size);
        for (uint32_t i = 0; i < size; i++) {
            neighbor_codes[i] = this->GetCodesById(neighbor_ids[i]);
        }
        redundant_insert_neighbors(id, neighbor_ids, neighbor_codes);
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
MixDataCell<QuantTmpl, IOTmpl>::redundant_insert_neighbors(
    uint64_t id, std::vector<uint64_t> neighbor_ids, std::vector<const uint8_t*> neighbor_codes) {
    // storage structure
    // 1. offset(uint64_t): offset[0] offset[1] ... offset[redundant_total_count_ - 1]
    //
    // 2. codes: [offset[0]] [offset[1]] ... [offset[redundant_total_count_ - 1]]
    //
    // 3. for each [offset[id]]:
    // size:      uint64_t
    // neighbor1: id(uint64_t), code(code_size_)
    // neighbor2: id(uint64_t), code(code_size_)
    // ...

    redundant_io_->Write(reinterpret_cast<uint8_t*>(&redundant_cur_offset_),
                         sizeof(uint64_t),
                         id * sizeof(uint64_t));

    uint64_t size = neighbor_codes.size();

    redundant_io_->Write(reinterpret_cast<uint8_t*>(&size), sizeof(size), redundant_cur_offset_);
    redundant_cur_offset_ += sizeof(size);

    for (uint64_t i = 0; i < size; i++) {
        redundant_io_->Write(reinterpret_cast<uint8_t*>(&neighbor_ids[i]),
                             sizeof(neighbor_ids[i]),
                             redundant_cur_offset_);
        redundant_cur_offset_ += sizeof(neighbor_ids[i]);

        redundant_io_->Write(neighbor_codes[i], this->codeSize_, redundant_cur_offset_);
        redundant_cur_offset_ += this->codeSize_;
    }
}

template <typename QuantTmpl, typename IOTmpl>
uint64_t
MixDataCell<QuantTmpl, IOTmpl>::get_codes_offset_by_id(uint64_t id) const {
    if (id > redundant_total_count_) {
        return std::numeric_limits<uint64_t>::max();
    }
    return *(uint64_t*)(redundant_io_->Read(sizeof(uint64_t), id * sizeof(uint64_t)));
}

template <typename QuantTmpl, typename IOTmpl>
uint64_t
MixDataCell<QuantTmpl, IOTmpl>::get_redundant_size_by_id(uint64_t id) const {
    uint64_t code_offset = get_codes_offset_by_id(id);
    if (code_offset == std::numeric_limits<uint64_t>::max()) {
        return 0;
    }
    return *(uint64_t*)(redundant_io_->Read(sizeof(uint64_t), code_offset));
}

template <typename QuantTmpl, typename IOTmpl>
uint64_t
MixDataCell<QuantTmpl, IOTmpl>::get_neighbor_codes_offset_by_id(uint64_t id,
                                                                uint64_t neighbor_i) const {
    uint64_t code_offset = get_codes_offset_by_id(id);
    if (code_offset == std::numeric_limits<uint64_t>::max()) {
        return code_offset;
    }

    return code_offset + sizeof(uint64_t) + neighbor_i * (this->codeSize_ + sizeof(uint64_t)) +
           sizeof(uint64_t);
}

template <typename QuantTmpl, typename IOTmpl>
uint64_t
MixDataCell<QuantTmpl, IOTmpl>::get_neighbor_id_by_id(uint64_t id, uint64_t neighbor_i) const {
    uint64_t code_offset = get_codes_offset_by_id(id);
    if (code_offset == std::numeric_limits<uint64_t>::max()) {
        return 0;
    }
    auto pos = code_offset + sizeof(uint64_t) + neighbor_i * (this->codeSize_ + sizeof(uint64_t));
    return *(uint64_t*)redundant_io_->Read(sizeof(uint64_t), pos);
}

template <typename QuantTmpl, typename IOTmpl>
template <typename IDType>
void
MixDataCell<QuantTmpl, IOTmpl>::QueryLine(float* resultDists,
                                          const float* queryVector,
                                          uint64_t id,
                                          std::vector<uint32_t>& to_be_visit,
                                          uint32_t count_no_visit) {
    std::unique_ptr<Computer<QuantTmpl>> computer = std::move(this->quantizer_->FactoryComputer());
    computer->SetQuery(queryVector);
    this->QueryLine(resultDists, computer, id, to_be_visit, count_no_visit);
}

template <typename QuantTmpl, typename IOTmpl>
template <typename IDType>
void
MixDataCell<QuantTmpl, IOTmpl>::QueryLine(float* resultDists,
                                          std::unique_ptr<Computer<QuantTmpl>>& computer,
                                          uint64_t id,
                                          std::vector<uint32_t>& to_be_visit,
                                          uint32_t count_no_visit) {
    if (id >= redundant_total_count_) {
        std::vector<uint64_t> neighbor_ids;
        graph_data_cell_->GetNeighbors(id, neighbor_ids);
        for (uint32_t i = 0; i < count_no_visit; i++) {
            if (to_be_visit[i] >= neighbor_ids.size()) {
                continue;
            }
            const auto* codes = this->GetCodesById(neighbor_ids[to_be_visit[i]]);
            computer->ComputeDist(codes, resultDists + i);
        }
    } else {
        uint64_t code_offset;
        const uint8_t* codes;
        uint64_t neighbor_size = get_redundant_size_by_id(id);

        for (uint32_t i = 0; i < prefetch_neighbor_codes_num and i < count_no_visit; i++) {
            if (to_be_visit[i] >= neighbor_size) {
                continue;
            }
            code_offset = this->get_neighbor_codes_offset_by_id(id, to_be_visit[i]);
            redundant_io_->Prefetch(code_offset, prefetch_cache_line);
        }
        for (uint32_t i = 0; i < count_no_visit; i++) {
            if (to_be_visit[i] >= neighbor_size) {
                continue;
            }

            if (i + prefetch_neighbor_codes_num < count_no_visit and
                to_be_visit[i + prefetch_neighbor_codes_num] < neighbor_size) {
                code_offset = this->get_neighbor_codes_offset_by_id(
                    id, to_be_visit[i + prefetch_neighbor_codes_num]);
                redundant_io_->Prefetch(code_offset, prefetch_cache_line);
            }

            code_offset = this->get_neighbor_codes_offset_by_id(id, to_be_visit[i]);
            codes = redundant_io_->Read(this->GetCodeSize(), code_offset);
            computer->ComputeDist(codes, resultDists + i);
        }
    };
}

}  // namespace vsag