
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
#include <math.h>

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "quantizer.h"
#include "simd/sq4_uniform_simd.h"

namespace vsag {

using norm_type = uint64_t;
using sum_type = float;

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ4UniformQuantizer : public Quantizer<SQ4UniformQuantizer<metric>> {
public:
    explicit SQ4UniformQuantizer(int dim);

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
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    inline void
    ProcessQueryImpl(const DataType* query, Computer<SQ4UniformQuantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<SQ4UniformQuantizer>& computer,
                    const uint8_t* codes,
                    float* dists) const;

private:
    DataType lower_bound_{0};
    DataType diff_{0};

    /***
     * code layout: sq-code(fixed) + norm(opt) + sum(opt)
     * for L2 and COSINE, norm is needed for fast computation
     * for IP and COSINE, sum is needed for restoring original distance
     */
    uint64_t offset_code_{0};
    uint64_t offset_norm_{0};
    uint64_t offset_sum_{0};
};

template <MetricType metric>
SQ4UniformQuantizer<metric>::SQ4UniformQuantizer(int dim)
    : Quantizer<SQ4UniformQuantizer<metric>>(dim) {
    lower_bound_ = std::numeric_limits<DataType>::max();
    diff_ = std::numeric_limits<DataType>::lowest();

    this->code_size_ = 0;

    offset_code_ = this->code_size_;
    this->code_size_ += (dim + 1) >> 1 << 1;

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        offset_norm_ = this->code_size_;
        this->code_size_ += sizeof(norm_type);  // norm of vector
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        offset_sum_ = this->code_size_;
        this->code_size_ += sizeof(sum_type);  // sum of vector
    }
}

template <MetricType metric>
bool
SQ4UniformQuantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
    if (data == nullptr) {
        return false;
    }

    lower_bound_ = std::numeric_limits<DataType>::max();
    diff_ = std::numeric_limits<DataType>::lowest();

    for (uint64_t i = 0; i < count; i++) {
        for (uint64_t d = 0; d < this->dim_; d++) {
            auto val = data[i * this->dim_ + d];
            if (val > diff_) {
                diff_ = val;
            }
            if (val < lower_bound_) {
                lower_bound_ = val;
            }
        }
    }

    diff_ -= lower_bound_;
    if (diff_ < 1e-4) {
        diff_ = 1;
    }

    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
SQ4UniformQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    float delta = 0;
    uint8_t scaled = 0;
    norm_type norm = 0;
    sum_type sum = 0;

    for (uint64_t d = 0; d < this->dim_; d++) {
        delta = 1.0f * (data[d] - lower_bound_) / diff_;
        if (delta < 0.0) {
            delta = 0;
        } else if (delta > 0.999) {
            delta = 1;
        }
        scaled = 15 * delta;

        if (d & 1) {
            codes[offset_code_ + (d >> 1)] |= scaled << 4;
        } else {
            codes[offset_code_ + (d >> 1)] = 0;
            codes[offset_code_ + (d >> 1)] |= scaled;
        }
        norm += scaled * scaled;
        sum += data[d];
    }

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        *(norm_type*)(codes + offset_norm_) = norm;
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        *(sum_type*)(codes + offset_sum_) = sum;
    }

    return true;
}

template <MetricType metric>
bool
SQ4UniformQuantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
SQ4UniformQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (uint64_t d = 0; d < this->dim_; d++) {
        if (d & 1) {
            data[d] = ((codes[d >> 1] & 0xf0) >> 4) / 15.0 * diff_ + lower_bound_;
        } else {
            data[d] = (codes[d >> 1] & 0x0f) / 15.0 * diff_ + lower_bound_;
        }
    }

    return true;
}

template <MetricType metric>
bool
SQ4UniformQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
inline float
SQ4UniformQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    float result = 0;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        result = SQ4UniformComputeCodesIP(codes1, codes2, this->dim_);

        norm_type norm1 = *((norm_type*)(codes1 + offset_norm_));
        norm_type norm2 = *((norm_type*)(codes2 + offset_norm_));

        result = norm1 + norm2 - 2 * result;
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        result = SQ4UniformComputeCodesIP(codes1, codes2, this->dim_);

        sum_type sum1 = *((sum_type*)(codes1 + offset_sum_));
        sum_type sum2 = *((sum_type*)(codes2 + offset_sum_));

        result = lower_bound_ * (sum1 + sum2) + (diff_ / 15.0) * (diff_ / 15.0) * result -
                 lower_bound_ * lower_bound_;

    } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        result = SQ4UniformComputeCodesIP(codes1, codes2, this->dim_);

        sum_type sum1 = *((sum_type*)(codes1 + offset_sum_));
        sum_type sum2 = *((sum_type*)(codes2 + offset_sum_));

        result = lower_bound_ * (sum1 + sum2) + (diff_ / 15.0) * (diff_ / 15.0) * result -
                 lower_bound_ * lower_bound_;
    } else {
        logger::error("unsupported metric type");
        result = 0;
    }
    return result;
}

template <MetricType metric>
void
SQ4UniformQuantizer<metric>::ProcessQueryImpl(const DataType* query,
                                              Computer<SQ4UniformQuantizer>& computer) const {
    try {
        computer.buf_ = new uint8_t[this->code_size_];  // todo: replace with allocator
        this->EncodeOneImpl(query, computer.buf_);
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        logger::error("bad alloc when init computer buf");
    }
}

template <MetricType metric>
void
SQ4UniformQuantizer<metric>::ComputeDistImpl(Computer<SQ4UniformQuantizer>& computer,
                                             const uint8_t* codes,
                                             float* dists) const {
    dists[0] = this->ComputeImpl(computer.buf_, codes);
}

}  // namespace vsag
