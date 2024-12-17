
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

#include "sq4_uniform_simd.h"

#include "simd_status.h"

namespace vsag {

static SQ4UniformComputeCodesType
GetSQ4UniformComputeCodesIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::SQ4UniformComputeCodesIP;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::SQ4UniformComputeCodesIP;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::SQ4UniformComputeCodesIP;
#endif
    }
    return generic::SQ4UniformComputeCodesIP;
}
SQ4UniformComputeCodesType SQ4UniformComputeCodesIP = GetSQ4UniformComputeCodesIP();
}  // namespace vsag
