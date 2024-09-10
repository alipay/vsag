
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

#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#include "quantizer.h"
#include "sq4_uniform_simd.h"

namespace vsag {

typedef uint64_t norm_type;
typedef float sum_type;

const char* OFFSET_KEY_CODE = "code";
const char* OFFSET_KEY_NORM = "norm";
const char* OFFSET_KEY_SUM = "sum";

template <MetricType Metric = MetricType::METRIC_TYPE_L2SQR>
class SQ4UniformQuantizer : public Quantizer<SQ4UniformQuantizer<Metric>> {
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
    DataType lower_bound_;
    DataType diff_;
    std::unordered_map<std::string, uint64_t> offsets_;
};

template <MetricType Metric>
SQ4UniformQuantizer<Metric>::SQ4UniformQuantizer(int dim)
    : Quantizer<SQ4UniformQuantizer<Metric>>(dim) {
    lower_bound_ = std::numeric_limits<DataType>::max();
    diff_ = std::numeric_limits<DataType>::min();

    this->codeSize_ = 0;

    offsets_[OFFSET_KEY_CODE] = this->codeSize_;
    this->codeSize_ += (dim + 1) / 2;

    if (Metric == MetricType::METRIC_TYPE_L2SQR or Metric == MetricType::METRIC_TYPE_COSINE) {
        offsets_[OFFSET_KEY_NORM] = this->codeSize_;
        this->codeSize_ += sizeof(norm_type);  // norm of vector
    }

    if (Metric == MetricType::METRIC_TYPE_IP or Metric == MetricType::METRIC_TYPE_COSINE) {
        offsets_[OFFSET_KEY_SUM] = this->codeSize_;
        this->codeSize_ += sizeof(sum_type);  // sum  of vector
    }
}

template <MetricType Metric>
bool
SQ4UniformQuantizer<Metric>::TrainImpl(const DataType* data, uint64_t count) {
    if (data == nullptr) {
        return false;
    }

    for (uint32_t i = 0; i < count; i++) {
        for (uint32_t d = 0; d < this->dim_; d++) {
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

    this->isTrained_ = true;
    return true;
}

template <MetricType Metric>
bool
SQ4UniformQuantizer<Metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    float delta;
    uint8_t scaled;
    norm_type norm = 0;
    sum_type sum = 0;

    for (uint32_t d = 0; d < this->dim_; d++) {
        delta = ((data[d] - lower_bound_) / diff_);
        if (delta < 0.0) {
            delta = 0;
        }
        if (delta > 0.999) {
            delta = 1;
        }
        scaled = 15 * delta;

        if (d & 1) {
            codes[offsets_.at(OFFSET_KEY_CODE) + d / 2] |= scaled << 4;
        } else {
            codes[offsets_.at(OFFSET_KEY_CODE) + d / 2] = 0;
            codes[offsets_.at(OFFSET_KEY_CODE) + d / 2] |= scaled;
        }
        norm += scaled * scaled;
        sum += data[d];
    }

    if (Metric == MetricType::METRIC_TYPE_L2SQR or Metric == MetricType::METRIC_TYPE_COSINE) {
        *(norm_type*)(codes + offsets_.at(OFFSET_KEY_NORM)) = norm;
    }

    if (Metric == MetricType::METRIC_TYPE_IP or Metric == MetricType::METRIC_TYPE_COSINE) {
        *(sum_type*)(codes + offsets_.at(OFFSET_KEY_SUM)) = sum;
    }

    return true;
}

template <MetricType Metric>
bool
SQ4UniformQuantizer<Metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->codeSize_);
    }
    return true;
}

template <MetricType Metric>
bool
SQ4UniformQuantizer<Metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (uint32_t d = 0; d < this->dim_; d++) {
        if (d & 1) {
            data[d] = ((codes[d / 2] & 0xf0) >> 4) / 15.0 * diff_ + lower_bound_;
            data[d] = data[d];
        } else {
            data[d] = (codes[d / 2] & 0x0f) / 15.0 * diff_ + lower_bound_;
            data[d] = data[d];
        }
    }

    return true;
}

template <MetricType Metric>
bool
SQ4UniformQuantizer<Metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->codeSize_, data + i * this->dim_);
    }
    return true;
}

template <MetricType Metric>
inline float
SQ4UniformQuantizer<Metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    float result = 0;
    if (Metric == MetricType::METRIC_TYPE_L2SQR) {
        result = SQ4UniformComputeCodesIP(codes1, codes2, this->dim_);

        norm_type norm1 = *((norm_type*)(codes1 + offsets_.at(OFFSET_KEY_NORM)));
        norm_type norm2 = *((norm_type*)(codes2 + offsets_.at(OFFSET_KEY_NORM)));

        result = norm1 + norm2 - 2 * result;
    } else if (Metric == MetricType::METRIC_TYPE_IP) {
        result = SQ4UniformComputeCodesIP(codes1, codes2, this->dim_);

        sum_type sum1 = *((sum_type*)(codes1 + offsets_.at(OFFSET_KEY_SUM)));
        sum_type sum2 = *((sum_type*)(codes2 + offsets_.at(OFFSET_KEY_SUM)));

        result = lower_bound_ * (sum1 + sum2) + (diff_ / 15.0) * (diff_ / 15.0) * result -
                 lower_bound_ * lower_bound_;

        //        result = -1 * result;
    } else if (Metric == MetricType::METRIC_TYPE_COSINE) {
        result = SQ4UniformComputeCodesIP(codes1, codes2, this->dim_);

        sum_type sum1 = *((sum_type*)(codes1 + offsets_.at(OFFSET_KEY_SUM)));
        sum_type sum2 = *((sum_type*)(codes2 + offsets_.at(OFFSET_KEY_SUM)));

        result = lower_bound_ * (sum1 + sum2) + (diff_ / 15.0) * (diff_ / 15.0) * result -
                 lower_bound_ * lower_bound_;

        //        norm_type norm1 =
        //            *((norm_type*)(codes1 + offsets_.at(OFFSET_KEY_NORM)));
        //        norm_type norm2 = *((norm_type*)(codes2 + offsets_.at(OFFSET_KEY_NORM)));
        //
        //        result =
        //            lower_bound_ * (sum1 + sum2) + (diff_ / 15.0) * (diff_ / 15.0) * result - lower_bound_ * lower_bound_;
        //
        //        result = 1.0 - result / (sqrt(norm1) * sqrt(norm2));
    } else {
        result = 0;
    }
    return result;
}

template <MetricType Metric>
void
SQ4UniformQuantizer<Metric>::ProcessQueryImpl(const DataType* query,
                                              Computer<SQ4UniformQuantizer>& computer) const {
    computer.buf_ = new uint8_t[this->codeSize_];
    this->EncodeOneImpl(query, computer.buf_);
}

template <MetricType Metric>
void
SQ4UniformQuantizer<Metric>::ComputeDistImpl(Computer<SQ4UniformQuantizer>& computer,
                                             const uint8_t* codes,
                                             float* dists) const {
    dists[0] = this->ComputeImpl(computer.buf_, codes);
}

}  // namespace vsag
