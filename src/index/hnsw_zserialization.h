
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

#include "vsag/binaryset.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/readerset.h"

namespace vsag {

class HNSW;
class serial_suite;

class HnswSerialization {
public:
    /* kv, metadata in key "_metadata" */
    static tl::expected<BinarySet, Error>
    KvSerialize(const HNSW& hnsw,
                uint64_t version = 1  // should use newest version always, keep for debugging
    );

    static tl::expected<void, Error>
    KvDeserialize(HNSW& hnsw, const BinarySet& binary_set);

    static tl::expected<void, Error>
    KvDeserialize(HNSW& hnsw, const ReaderSet& reader_set);

public:
    /* streaming, metadata in footer */
    static tl::expected<void, Error>
    StreamingSerialize(const HNSW& hnsw,
                       std::ostream& out_stream,
                       uint64_t version = 1  // should use newest version always, keep for debugging
    );

    static tl::expected<void, Error>
    StreamingDeserialize(HNSW& hnsw, std::istream& in_stream);

private:
    friend class serial;

    /* persistent format version 1: add metadata */
    class v1 {
    public:
        static tl::expected<BinarySet, Error>
        KvSerialize(const HNSW& hnsw);

        static tl::expected<void, Error>
        KvDeserialize(HNSW& hnsw, const BinarySet& binary_set);

        static tl::expected<void, Error>
        KvDeserialize(HNSW& hnsw, const ReaderSet& reader_set);

        static tl::expected<void, Error>
        StreamingSerialize(const HNSW& hnsw, std::ostream& out_stream);

        static tl::expected<void, Error>
        StreamingDeserialize(HNSW& hnsw, std::istream& in_stream);
    };

private:
    /* persistent format version 0: original */
    class v0 {
    public:
        static tl::expected<BinarySet, Error>
        KvSerialize(const HNSW& hnsw);

        static tl::expected<void, Error>
        KvDeserialize(HNSW& hnsw, const BinarySet& binary_set);

        static tl::expected<void, Error>
        KvDeserialize(HNSW& hnsw, const ReaderSet& reader_set);

        static tl::expected<void, Error>
        StreamingSerialize(const HNSW& hnsw, std::ostream& out_stream);

        static tl::expected<void, Error>
        StreamingDeserialize(HNSW& hnsw, std::istream& in_stream);

    private:
        static BinarySet
        empty_binaryset();
    };
};

}  // namespace vsag
