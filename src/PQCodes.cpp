//
// Created by root on 8/16/24.
//

#include "vsag/PQCodes.h"
#include "clustering.h"
#include "simd/simd.h"
#include <cstring>
#include <tmmintrin.h>
#include <immintrin.h>
#include <fstream>

void SelectRandomNumbers(std::vector<int64_t> &result, int64_t N, int64_t X) {
    if (X > N) {
        throw std::invalid_argument("X cannot be greater than N.");
    }

    std::vector<int> numbers(N);
    std::iota(numbers.begin(), numbers.end(), 0); // 生成0到N-1的数列

    result.resize(X);
    std::sample(numbers.begin(), numbers.end(), result.begin(), X, std::mt19937{std::random_device{}()});
}


PQCodes::PQCodes(int subspace, int dim) {
    subSpace_ = subspace;
    dim_ = dim;
    codebook.resize(dim * centerCount_);
    dimPerSpace_ = dim_ / subSpace_;
}

void PQCodes::Train(const float *data, int64_t count) {

    auto newCount = count;
    if (newCount > 65536) {
        newCount = 65536;
    }
    std::vector<int64_t> sample;
    SelectRandomNumbers(sample, count, newCount);
#pragma omp parallel for
    for (int64_t i = 0; i < subSpace_; ++i) {
        std::vector<float> curData(newCount * dimPerSpace_);
        for (int64_t j = 0; j < newCount; ++j) {
            auto id = sample[j];
            memcpy(curData.data() + j * dimPerSpace_, data + this->dim_ * id + i * dimPerSpace_,
                   dimPerSpace_ * sizeof(float));
        }
        vsag::Clustering clustering(dimPerSpace_, centerCount_);
        clustering.verbose = true;
        clustering.train(newCount, curData.data());

        memcpy(this->codebook.data() + i * dimPerSpace_ * centerCount_,
               clustering.centroids.data(),
               dimPerSpace_ * centerCount_ * sizeof(float));
    }
}


void PQCodes::BatchEncode(const float *data, int64_t count, std::vector<uint8_t> &codes) {
    codes.resize(count * subSpace_);
#pragma omp parallel for
    for (int64_t i = 0; i < count; ++i) {
        const float *curData = data + i * this->dim_;
        auto *curCodes = codes.data() + i * subSpace_;
        for (int64_t j = 0; j < subSpace_; ++j) {
            auto *book = this->codebook.data() + j * dimPerSpace_ * centerCount_;
            auto *subquery = curData + j * dimPerSpace_;
            auto id = 0;
            auto distMin = 10000.0;
            for (auto r = 0; r < centerCount_; ++r) {
                auto dist = vsag::L2Sqr(subquery, book + r * dimPerSpace_, &dimPerSpace_);
                if (dist < distMin) {
                    distMin = dist;
                    id = r;
                }
            }
            curCodes[j] = id;
        }
    }
}

void PQCodes::Packaged(std::vector<uint8_t> &codes)
{
    std::vector<int> maps = {
            0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14,
            30, 15, 31
    };
    auto count = codes.size() / subSpace_;
    auto newCount = ((count + 31) / 32) * 32;
    codes.resize(newCount * subSpace_, 0);
    auto blockCount = newCount / 32;
    std::vector<uint8_t> tmp(16 * subSpace_, 0);
    for (int j = 0; j < blockCount; ++j) {
        auto *data = codes.data() + j * subSpace_ * 32;
        tmp.clear();
        tmp.resize(16 * subSpace_, 0);
        for (int i = 0; i < subSpace_; ++i) {
            auto *to = tmp.data() + 16 * i;
            int num = 0;
            while (num < 32) {
                auto x = num / 2;
                auto value = data[maps[num] * subSpace_ + i];
                if (num % 2 == 1) {
                    value <<= 4;
                }
                to[x] |= value;
                ++num;
            }
        }
        memcpy(codes.data() + j * 16 * subSpace_, tmp.data(), 16 * subSpace_);
    }
    codes.resize(blockCount * 16 * subSpace_);
}

void PQCodes::ToFile(std::string filepath)
{
    std::ofstream outf(filepath.c_str(), std::ios::binary);
    auto size = this->codebook.size();
    outf.write((char*)(&size), sizeof(size));
    outf.write((char*)(this->codebook.data()), size * sizeof(float));
    outf.close();
}

void PQCodes::FromFile(std::string filepath)
{
    std::ifstream inf(filepath.c_str(), std::ios::binary);
    size_t size = 0;
    inf.read((char*)&size, sizeof(size));
    this->codebook.resize(size);
    inf.read((char*)(this->codebook.data()), size * sizeof(float));
    inf.close();
}


PQScanner::PQScanner(PQCodes *pq) {
    this->pqCodes_ = pq;
    this->lut_.resize(16 * pq->subSpace_);
    result.resize(32);
}

void PQScanner::SetQuery(float *query) {
    std::vector<float> lutf(lut_.size());
    for (int i = 0; i < pqCodes_->subSpace_; ++i) {
        for (int j = 0; j < 16; ++j) {
            lutf[i * 16 + j] = vsag::L2Sqr(query + i * pqCodes_->dimPerSpace_,
                                           pqCodes_->codebook.data() + (i * 16 + j) * pqCodes_->dimPerSpace_,
                                           &(pqCodes_->dimPerSpace_));
            this->sqMax_ = std::max(lutf[i * 16 + j], this->sqMax_);
            this->sqMin_ = std::min(lutf[i * 16 + j], this->sqMin_);
        }
    }
    for (int i = 0; i < lutf.size(); ++i) {
        lut_[i] = uint8_t(255.0 * ((lutf[i] - sqMin_) / (sqMax_ - sqMin_)));
    }
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "portability-simd-intrinsics"
void PQScanner::ScanCodes(const uint8_t *codes, std::vector<float> &dists) {
    auto *lut = this->lut_.data();
    auto M = this->pqCodes_->subSpace_ / 4;
    __m512i sum1 = _mm512_set1_epi32(0);
    __m512i sum2 = _mm512_set1_epi32(0);
    __m512i mask = _mm512_set1_epi8(0x0F);
    for (int i = 0; i < M; ++i) {
        __m512i curCode = _mm512_loadu_epi8(codes);
        __m512i curLut = _mm512_loadu_epi8(lut);

        __m512i c1 = _mm512_and_si512(curCode, mask);
        __m512i c2 = _mm512_and_si512(_mm512_srli_epi16(curCode, 4), mask);

        __m512i res1 = _mm512_shuffle_epi8(curLut, c1);
        __m512i res2 = _mm512_shuffle_epi8(curLut, c2);
        sum1 = _mm512_add_epi16(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(res1, 0)), sum1);
        sum1 = _mm512_add_epi16(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(res1, 1)), sum1);
        sum1 = _mm512_add_epi16(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(res1, 2)), sum1);
        sum1 = _mm512_add_epi16(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(res1, 3)), sum1);
        sum2 = _mm512_add_epi16(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(res2, 0)), sum2);
        sum2 = _mm512_add_epi16(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(res2, 1)), sum2);
        sum2 = _mm512_add_epi16(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(res2, 2)), sum2);
        sum2 = _mm512_add_epi16(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(res2, 3)), sum2);
        codes += 64;
        lut += 64;
    }
    auto *curp = result.data();
    _mm512_storeu_epi16(curp, sum1);
    _mm512_storeu_epi16(curp + 16, sum2);
    for (auto i = 0; i < 32; ++i) {
        dists[i] = (double(result[i]) / 255.0) * (sqMax_ - sqMin_) + M * 2.0 * sqMin_;
    }
}
#pragma clang diagnostic pop
