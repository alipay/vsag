
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
#include <cpuinfo.h>
#include <x86intrin.h>

#include <vector>

namespace vsag {

extern float (*SQ4ComputeIPAlign64)(const uint8_t* query,
                                    const uint8_t* codes,
                                    const float* lowerBound,
                                    const float* diff,
                                    uint64_t dim);
extern float (*SQ4ComputeIPAlign32)(const uint8_t* query,
                                    const uint8_t* codes,
                                    const float* lowerBound,
                                    const float* diff,
                                    uint64_t dim);
extern float (*SQ4ComputeIPAlign16)(const uint8_t* query,
                                    const uint8_t* codes,
                                    const float* lowerBound,
                                    const float* diff,
                                    uint64_t dim);
extern float (*SQ4ComputeIPAlign1)(const uint8_t* query,
                                   const uint8_t* codes,
                                   const float* lowerBound,
                                   const float* diff,
                                   uint64_t dim);

extern float (*SQ4ComputeL2Align64)(const uint8_t* query,
                                    const uint8_t* codes,
                                    const float* lowerBound,
                                    const float* diff,
                                    uint64_t dim);
extern float (*SQ4ComputeL2Align32)(const uint8_t* query,
                                    const uint8_t* codes,
                                    const float* lowerBound,
                                    const float* diff,
                                    uint64_t dim);
extern float (*SQ4ComputeL2Align16)(const uint8_t* query,
                                    const uint8_t* codes,
                                    const float* lowerBound,
                                    const float* diff,
                                    uint64_t dim);
extern float (*SQ4ComputeL2Align1)(const uint8_t* query,
                                   const uint8_t* codes,
                                   const float* lowerBound,
                                   const float* diff,
                                   uint64_t dim);

float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeL2(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim);
float
SQ4ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim);

namespace Generic {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeL2(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim);
float
SQ4ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim);
}  // namespace Generic

namespace SSE {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeL2(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim);
float
SQ4ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim);
}  // namespace SSE

namespace AVX2 {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeL2(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim);
float
SQ4ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim);
}  // namespace AVX2

namespace AVX512 {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeL2(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim);
float
SQ4ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim);
}  // namespace AVX512

void
SetSIMD();

}  // namespace vsag