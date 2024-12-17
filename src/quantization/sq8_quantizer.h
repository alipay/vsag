
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
#include <cstring>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>

#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "quantizer.h"
#include "scalar_quantization_trainer.h"
#include "simd/normalize.h"
#include "simd/sq8_simd.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ8Quantizer : public Quantizer<SQ8Quantizer<metric>> {
public:
    explicit SQ8Quantizer(int dim, Allocator* allocator);

    SQ8Quantizer(const JsonType& quantization_param, const IndexCommonParam& common_param);

    ~SQ8Quantizer() = default;

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes);

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    inline float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    inline void
    ProcessQueryImpl(const DataType* query, Computer<SQ8Quantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<SQ8Quantizer>& computer, const uint8_t* codes, float* dists) const;

    inline void
    SerializeImpl(StreamWriter& writer);

    inline void
    DeserializeImpl(StreamReader& reader);

    inline void
    ReleaseComputerImpl(Computer<SQ8Quantizer<metric>>& computer) const;

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_SQ8;
    }

public:
    Vector<DataType> diff_;
    Vector<DataType> lower_bound_;
};

template <MetricType Metric>
SQ8Quantizer<Metric>::SQ8Quantizer(int dim, Allocator* allocator)
    : Quantizer<SQ8Quantizer<Metric>>(dim, allocator), diff_(allocator), lower_bound_(allocator) {
    // align 64 bytes (512 bits) to avoid illegal memory access in SIMD
    this->code_size_ = this->dim_;
    this->diff_.resize(dim, 0);
    this->lower_bound_.resize(dim, std::numeric_limits<DataType>::max());
}

template <MetricType metric>
SQ8Quantizer<metric>::SQ8Quantizer(const JsonType& quantization_param,
                                   const IndexCommonParam& common_param)
    : Quantizer<SQ8Quantizer<metric>>(common_param.dim_, common_param.allocator_),
      diff_(common_param.allocator_),
      lower_bound_(common_param.allocator_) {
    // align 64 bytes (512 bits) to avoid illegal memory access in SIMD
    this->code_size_ = this->dim_;
    this->diff_.resize(this->dim_, 0);
    this->lower_bound_.resize(this->dim_, std::numeric_limits<DataType>::max());
}

template <MetricType metric>
bool
SQ8Quantizer<metric>::TrainImpl(const vsag::DataType* data, uint64_t count) {
    if (this->is_trained_) {
        return true;
    }
    bool need_normalize = false;
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        need_normalize = true;
    }

    ScalarQuantizationTrainer trainer(this->dim_, 8);
    trainer.Train(data, count, this->diff_.data(), this->lower_bound_.data(), need_normalize);

    for (uint64_t i = 0; i < this->dim_; ++i) {
        this->diff_[i] -= this->lower_bound_[i];
    }
    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
SQ8Quantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) {
    const DataType* cur = data;
    Vector<float> tmp(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        tmp.resize(this->dim_);
        Normalize(data, tmp.data(), this->dim_);
        cur = tmp.data();
    }
    for (int i = 0; i < this->dim_; ++i) {
        float xi = 0;
        if (diff_[i] != 0) {
            xi = (cur[i] - lower_bound_[i]) / diff_[i];
            if (xi < 0.0) {
                xi = 0;
            }
            if (xi > 0.999) {
                xi = 1.0;
            }
        }
        codes[i] = int(xi * 255);
    }
    return true;
}

template <MetricType metric>
bool
SQ8Quantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
SQ8Quantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
bool
SQ8Quantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (uint64_t i = 0; i < this->dim_; ++i) {
        data[i] = static_cast<DataType>(static_cast<float>(codes[i]) / 255.0 * diff_[i] +
                                        lower_bound_[i]);
    }
    return true;
}

template <MetricType metric>
inline float
SQ8Quantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        return SQ8ComputeCodesL2Sqr(
            codes1, codes2, this->lower_bound_.data(), this->diff_.data(), this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        return 1 - SQ8ComputeCodesIP(
                       codes1, codes2, this->lower_bound_.data(), this->diff_.data(), this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        return 1 - SQ8ComputeCodesIP(
                       codes1, codes2, this->lower_bound_.data(), this->diff_.data(), this->dim_);
    } else {
        return 0.0f;
    }
}

template <MetricType metric>
void
SQ8Quantizer<metric>::ProcessQueryImpl(const DataType* query,
                                       Computer<SQ8Quantizer>& computer) const {
    try {
        computer.buf_ =
            reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->dim_ * sizeof(float)));
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        logger::error("bad alloc when init computer buf");
        throw std::bad_alloc();
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        Normalize(query, reinterpret_cast<float*>(computer.buf_), this->dim_);
    } else {
        memcpy(computer.buf_, query, this->dim_ * sizeof(float));
    }
}

template <MetricType metric>
void
SQ8Quantizer<metric>::ComputeDistImpl(Computer<SQ8Quantizer>& computer,
                                      const uint8_t* codes,
                                      float* dists) const {
    auto* query = reinterpret_cast<float*>(computer.buf_);

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        *dists = SQ8ComputeL2Sqr(
            query, codes, this->lower_bound_.data(), this->diff_.data(), this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        *dists = 1 - SQ8ComputeIP(
                         query, codes, this->lower_bound_.data(), this->diff_.data(), this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        *dists = 1 - SQ8ComputeIP(
                         query, codes, this->lower_bound_.data(), this->diff_.data(), this->dim_);
    } else {
        *dists = 0.0f;
    }
}

template <MetricType metric>
void
SQ8Quantizer<metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteVector(writer, this->diff_);
    StreamWriter::WriteVector(writer, this->lower_bound_);
}

template <MetricType metric>
void
SQ8Quantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->diff_);
    StreamReader::ReadVector(reader, this->lower_bound_);
}

template <MetricType metric>
void
SQ8Quantizer<metric>::ReleaseComputerImpl(Computer<SQ8Quantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

}  // namespace vsag
