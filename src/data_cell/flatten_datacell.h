
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
#include <limits>
#include <memory>

#include "io/basic_io.h"
#include "quantization/quantizer.h"
namespace vsag {

template <typename QuantTmpl, typename IOTmpl>
class FlattenDataCell {
public:
    FlattenDataCell() = default;

    explicit FlattenDataCell(const std::string& initializeJson);  // todo

    void
    TrainQuantizer(const float* data, uint64_t size);

    void
    InsertVector(const float* vector, uint64_t idx = std::numeric_limits<uint64_t>::max());

    template <typename IDType = uint64_t>
    void
    BatchInsertVector(const float* vectors, uint64_t count, IDType* idx = nullptr);

    template <typename IDType = uint64_t>
    void
    Query(float* resultDists, const float* queryVector, const IDType* idx, uint64_t idCount);

    template <typename IDType = uint64_t>
    void
    Query(float* resultDists,
          std::unique_ptr<Computer<QuantTmpl>>& computer,
          const IDType* idx,
          uint64_t idCount);

    template <typename IDType = uint64_t>
    void
    BatchQuery(float* resultDists,
               const float* queryVector,
               const IDType* idx,
               uint64_t queryCount,
               uint64_t idCount);

    inline std::unique_ptr<Computer<QuantTmpl>>
    FactoryComputer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }

    float
    ComputePairVectors(uint64_t id1, uint64_t id2);

    inline void
    SetMaxCapacity(uint64_t capacity) {
        this->maxCapacity_ = std::max(capacity, this->totalCount_);  // TODO add warning
    }

    [[nodiscard]] const uint8_t*
    GetCodesById(uint64_t id) const;

    inline void
    SetQuantizer(std::unique_ptr<Quantizer<QuantTmpl>>& quantizer) {
        this->quantizer_.swap(quantizer);
        this->codeSize_ = quantizer_->GetCodeSize();
    }

    inline void
    SetQuantizer(std::unique_ptr<Quantizer<QuantTmpl>>&& quantizer) {
        this->quantizer_.swap(quantizer);
        this->codeSize_ = quantizer_->GetCodeSize();
    }

    inline void
    SetIO(std::unique_ptr<BasicIO<IOTmpl>>& io) {
        this->io_.swap(io);
    }

    inline void
    SetIO(std::unique_ptr<BasicIO<IOTmpl>>&& io) {
        this->io_.swap(io);
    }

    inline uint64_t
    TotalCount() {
        return this->totalCount_;
    }

public:
    std::unique_ptr<Quantizer<QuantTmpl>> quantizer_{nullptr};

    std::unique_ptr<BasicIO<IOTmpl>> io_{nullptr};

    uint64_t totalCount_{0};

    uint64_t maxCapacity_{1000000};

    uint64_t codeSize_{0};
};

template <typename QuantTmpl, typename IOTmpl>
FlattenDataCell<QuantTmpl, IOTmpl>::FlattenDataCell(const std::string& initializeJson) {
    // TODO
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::TrainQuantizer(const float* data, uint64_t size) {
    if (this->quantizer_) {
        this->quantizer_->Train(data, size);
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::InsertVector(const float* vector, uint64_t idx) {
    if (idx == std::numeric_limits<uint64_t>::max()) {
        idx = totalCount_;
        ++totalCount_;
    }
    auto* codes = new uint8_t[codeSize_];
    quantizer_->EncodeOne(vector, codes);
    io_->Write(codes, codeSize_, idx * codeSize_);
    delete[] codes;
}

template <typename QuantTmpl, typename IOTmpl>
template <typename IDType>
void
FlattenDataCell<QuantTmpl, IOTmpl>::BatchInsertVector(const float* vectors,
                                                      uint64_t count,
                                                      IDType* idx) {
    if (idx == nullptr) {
        auto* codes = new uint8_t[codeSize_ * count];
        quantizer_->EncodeBatch(vectors, codes, count);
        io_->Write(codes, codeSize_ * count, totalCount_ * codeSize_);
        totalCount_ += count;
        delete[] codes;
    } else {
        uint64_t dim = quantizer_->GetDim();
        for (uint64_t i = 0; i < count; ++i) {
            InsertVector(vectors + dim * i, idx[i]);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
template <typename IDType>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Query(float* resultDists,
                                          const float* queryVector,
                                          const IDType* idx,
                                          uint64_t idCount) {
    auto computer = quantizer_->FactoryComputer();
    computer->SetQuery(queryVector);
    this->Query(resultDists, computer, idx, idCount);
}

template <typename QuantTmpl, typename IOTmpl>
template <typename IDType>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Query(float* resultDists,
                                          std::unique_ptr<Computer<QuantTmpl>>& computer,
                                          const IDType* idx,
                                          uint64_t idCount) {
    for (uint64_t i = 0; i < idCount; ++i) {
        const auto* codes = GetCodesById(idx[i]);
        computer->ComputeDist(codes, resultDists + i);
    }
}

template <typename QuantTmpl, typename IOTmpl>
template <typename IDType>
void
FlattenDataCell<QuantTmpl, IOTmpl>::BatchQuery(float* resultDists,
                                               const float* queryVector,
                                               const IDType* idx,
                                               uint64_t queryCount,
                                               uint64_t idCount) {
    auto dim = quantizer_->GetDim();
    for (uint64_t i = 0; i < queryCount; ++i) {
        auto* result = resultDists + i * idCount;
        this->Query(result, queryVector + i * dim, idx, idCount);
    }
}

template <typename QuantTmpl, typename IOTmpl>
float
FlattenDataCell<QuantTmpl, IOTmpl>::ComputePairVectors(uint64_t id1, uint64_t id2) {
    const auto* codes1 = this->GetCodesById(id1);
    const auto* codes2 = this->GetCodesById(id1);
    return this->quantizer_->Compute(codes1, codes2);
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
FlattenDataCell<QuantTmpl, IOTmpl>::GetCodesById(uint64_t id) const {
    return io_->Read(codeSize_, id * codeSize_);
}

}  // namespace vsag
