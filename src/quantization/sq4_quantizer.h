
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
#include <cstring>
#include <limits>
#include <vector>

#include "quantizer.h"
#include "sq4_simd.h"

namespace vsag {

template <MetricType Metric = MetricType::METRIC_TYPE_L2SQR>
class SQ4Quantizer : public Quantizer<SQ4Quantizer<Metric>> {
public:
    explicit SQ4Quantizer(int dim);

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    inline float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    inline void
    ProcessQueryImpl(const DataType* query, Computer<SQ4Quantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<SQ4Quantizer>& computer, const uint8_t* codes, float* dists) const;

private:
    std::vector<DataType> lower_bound_;
    std::vector<DataType> diff_;
    uint32_t vector_encoded_size_;
};

template <MetricType Metric>
SQ4Quantizer<Metric>::SQ4Quantizer(int dim) : Quantizer<SQ4Quantizer<Metric>>(dim) {
    vector_encoded_size_ = (dim + 1) / 2;
    lower_bound_.resize(dim, std::numeric_limits<DataType>::max());
    diff_.resize(dim, std::numeric_limits<DataType>::min());
    this->codeSize_ = vector_encoded_size_;
    //    this->codeSize_ = vector_encoded_size_ + sizeof(norm_size_t);
}

template <MetricType Metric>
bool
SQ4Quantizer<Metric>::TrainImpl(const DataType* data, uint64_t count) {
    if (data == nullptr) {
        return false;
    }

    for (uint32_t i = 0; i < count; i++) {
        for (uint32_t d = 0; d < this->dim_; d++) {
            auto val = data[i * this->dim_ + d];
            if (val > diff_[d]) {
                diff_[d] = val;
            }
            if (val < lower_bound_[d]) {
                lower_bound_[d] = val;
            }
        }
    }

    for (uint32_t d = 0; d < this->dim_; d++) {
        diff_[d] -= lower_bound_[d];
    }

    this->isTrained_ = true;
    return true;
}

template <MetricType Metric>
bool
SQ4Quantizer<Metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    float delta;
    uint8_t scaled;

    //    norm_size_t norm = 0;
    for (uint32_t d = 0; d < this->dim_; d++) {
        delta = ((data[d] - lower_bound_[d]) / diff_[d]);
        if (delta < 0.0) {
            delta = 0;
        }
        if (delta > 0.999) {
            delta = 1;
        }
        scaled = 15 * delta;

        if (d & 1) {
            codes[d / 2] |= scaled << 4;
        } else {
            codes[d / 2] = 0;
            codes[d / 2] |= scaled;
        }
        //        norm += (norm_size_t)(data[d] * data[d]);
    }
    //    if (Metric == MetricType::METRIC_TYPE_L2SQR) {
    //        *(norm_size_t*)(codes + this->vector_encoded_size_) = norm;
    //    }

    return true;
}

template <MetricType Metric>
bool
SQ4Quantizer<Metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->codeSize_);
    }
    return true;
}

template <MetricType Metric>
bool
SQ4Quantizer<Metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (uint32_t d = 0; d < this->dim_; d++) {
        if (d & 1) {
            data[d] = ((codes[d / 2] & 0xf0) >> 4) / 15.0 * diff_[d] + lower_bound_[d];
            data[d] = data[d];
        } else {
            data[d] = (codes[d / 2] & 0x0f) / 15.0 * diff_[d] + lower_bound_[d];
            data[d] = data[d];
        }
    }

    return true;
}

template <MetricType Metric>
bool
SQ4Quantizer<Metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->codeSize_, data + i * this->dim_);
    }
    return true;
}

template <MetricType Metric>
inline float
SQ4Quantizer<Metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if (Metric == MetricType::METRIC_TYPE_L2SQR) {
        return SQ4ComputeCodesL2(codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
    } else if (Metric == MetricType::METRIC_TYPE_IP) {
        return SQ4ComputeCodesIP(codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
    } else if (Metric == MetricType::METRIC_TYPE_COSINE) {
        return SQ4ComputeCodesIP(codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
    } else {
        return 0;
    }
}

template <MetricType Metric>
void
SQ4Quantizer<Metric>::ProcessQueryImpl(const DataType* query,
                                       Computer<SQ4Quantizer>& computer) const {
    computer.buf_ = new uint8_t[this->codeSize_];
    this->EncodeOneImpl(query, computer.buf_);
}

template <MetricType Metric>
void
SQ4Quantizer<Metric>::ComputeDistImpl(Computer<SQ4Quantizer>& computer,
                                      const uint8_t* codes,
                                      float* dists) const {
    if (Metric == MetricType::METRIC_TYPE_L2SQR) {
        dists[0] =
            SQ4ComputeCodesL2(computer.buf_, codes, lower_bound_.data(), diff_.data(), this->dim_);
    } else if (Metric == MetricType::METRIC_TYPE_IP) {
        dists[0] =
            SQ4ComputeCodesIP(computer.buf_, codes, lower_bound_.data(), diff_.data(), this->dim_);
    } else if (Metric == MetricType::METRIC_TYPE_COSINE) {
        dists[0] =
            SQ4ComputeCodesIP(computer.buf_, codes, lower_bound_.data(), diff_.data(), this->dim_);
    } else {
        dists[0] = 0;
    }
}

}  // namespace vsag
