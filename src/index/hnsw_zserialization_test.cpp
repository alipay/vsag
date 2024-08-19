
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

#include "hnsw_zserialization.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "../logger.h"
#include "catch2/catch_message.hpp"
#include "catch2_utils.h"
#include "fixtures.h"
#include "hnsw.h"
#include "hnsw_zparameters.h"
#include "vsag/binaryset.h"
#include "vsag/dataset.h"
#include "vsag/readerset.h"

namespace {

struct kv_store {
    void
    put(const std::string& key, const std::string& value) {
        data[key] = value;
    }

    std::string
    get(const std::string& key) const {
        return data.at(key);
    }

    auto
    begin() const {
        return data.begin();
    }

    auto
    end() const {
        return data.end();
    }

    std::unordered_map<std::string, std::string> data;
};

std::shared_ptr<vsag::HNSW>
generate_empty_hnsw(int64_t dim, int64_t max_degree, int64_t ef_construction) {
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    return index;
}

std::shared_ptr<vsag::HNSW>
generate_hnsw(int64_t num_elements, int64_t dim, int64_t max_degree, int64_t ef_construction) {
    auto index = generate_empty_hnsw(dim, max_degree, ef_construction);
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    auto base = vsag::Dataset::Make();
    base->NumElements(num_elements)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    if (not index->Build(base).has_value()) {
        abort();
    }

    return index;
}

};  // namespace

namespace vsag {

template <typename s_impl, typename d_impl>
struct serialization_test_wrapper {
    static tl::expected<BinarySet, Error>
    kv_serialize(const HNSW& hnsw) {
        if constexpr (std::is_same<s_impl, HnswSerialization>::value) {
            return s_impl::KvSerialize(hnsw);
        }
        return s_impl::kv_serialize(hnsw);
    }

    static tl::expected<void, Error>
    kv_deserialize(HNSW& hnsw, const BinarySet& binary_set) {
        if constexpr (std::is_same<d_impl, HnswSerialization>::value) {
            d_impl::KvDeserialize(hnsw, binary_set);
            return {};
        } else {
            d_impl::kv_deserialize(hnsw, binary_set);
            return {};
        }
    }

    static tl::expected<void, Error>
    kv_deserialize(HNSW& hnsw, const ReaderSet& reader_set) {
        if constexpr (std::is_same<d_impl, HnswSerialization>::value) {
            d_impl::KvDeserialize(hnsw, reader_set);
            return {};
        } else {
            d_impl::kv_deserialize(hnsw, reader_set);
            return {};
        }
    }

    static tl::expected<void, Error>
    stream_serialize(const HNSW& hnsw, std::ostream& out_stream) {
        if constexpr (std::is_same<s_impl, HnswSerialization>::value) {
            s_impl::StreamingSerialize(hnsw, out_stream);
            return {};
        } else {
            s_impl::streaming_serialize(hnsw, out_stream);
            return {};
        }
    }

    static tl::expected<void, Error>
    stream_deserialize(HNSW& hnsw, std::istream& in_stream) {
        if constexpr (std::is_same<d_impl, HnswSerialization>::value) {
            d_impl::StreamingDeserialize(hnsw, in_stream);
            return {};
        } else {
            d_impl::streaming_deserialize(hnsw, in_stream);
            return {};
        }
    }
};

struct test_suite {
    using entry = HnswSerialization;
    using v1 = HnswSerialization::v1;
    using v0 = HnswSerialization::v0;

    using v0_v0 = serialization_test_wrapper<v0, v0>;
    using v1_v1 = serialization_test_wrapper<v1, v1>;
    using v0_entry = serialization_test_wrapper<v0, entry>;
    using v1_entry = serialization_test_wrapper<v1, entry>;
};

TEMPLATE_TEST_CASE("hnsw kv-stype serialization",
                   "[ut][hnsw]",
                   test_suite::v0_v0,
                   test_suite::v1_v1,
                   test_suite::v0_entry,
                   test_suite::v1_entry) {
    logger::set_level(logger::level::debug);
    auto index = ::generate_hnsw(/*num_elements=*/1000,
                                 /*dim=*/17,
                                 /*max_degree=*/12,
                                 /*ef_construction=*/100);

    // for testing
    ::kv_store kv;

    // save to a kv store, and delete index
    {
        auto serialize = TestType::kv_serialize(*index);
        index = nullptr;
        REQUIRE(serialize.has_value());
        auto serialize_result = serialize.value();
        for (const auto& key : serialize_result.GetKeys()) {
            auto value = serialize_result.Get(key);
            kv.put(key, std::string((char*)value.data.get(), value.size));
        }
        index = nullptr;
    }

    // load from a kv store via binaryset
    {
        index = ::generate_empty_hnsw(
            /*dim=*/17,
            /*max_degree=*/12,
            /*ef_construction=*/100);
        BinarySet bs;
        for (const auto& pair : kv) {
            Binary b{
                .data = std::shared_ptr<int8_t[]>(new int8_t[pair.second.size()]),
                .size = pair.second.length(),
            };
            memcpy(b.data.get(), pair.second.data(), b.size);
            bs.Set(pair.first, b);
        }

        auto deserialize = TestType::kv_deserialize(*index, bs);
        REQUIRE(deserialize.has_value());
    }

    // load from a kv store via reader
    {
        class BufReader : public Reader {
        public:
            BufReader(const std::string& buf) : buf_(buf) {
            }

        public:
            void
            Read(uint64_t offset, uint64_t len, void* dest) override {
                memcpy(dest, buf_.data() + offset, len);
            }

            void
            AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
                memcpy(dest, buf_.data() + offset, len);
                callback(IOErrorCode::IO_SUCCESS, "");
            }

            uint64_t
            Size() const override {
                return buf_.size();
            }

        private:
            const std::string& buf_;
        };

        index = ::generate_empty_hnsw(
            /*dim=*/17,
            /*max_degree=*/12,
            /*ef_construction=*/100);
        ReaderSet rs;
        for (const auto& pair : kv) {
            auto reader = std::make_shared<BufReader>(pair.second);
            rs.Set(pair.first, reader);
        }

        auto deserialize = TestType::kv_deserialize(*index, rs);
        REQUIRE(deserialize.has_value());
    }
}

TEMPLATE_TEST_CASE("hnsw stream-stype serialization",
                   "[ut][hnsw]",
                   test_suite::v0_v0,
                   test_suite::v1_v1,
                   test_suite::v0_entry,
                   test_suite::v1_entry) {
    logger::set_level(logger::level::debug);
    auto index = ::generate_hnsw(/*num_elements=*/1000,
                                 /*dim=*/17,
                                 /*max_degree=*/12,
                                 /*ef_construction=*/100);

    // for testing
    std::stringstream buf;

    // save to a stream, and delete index
    {
        auto serialize = TestType::stream_serialize(*index, buf);
        index = nullptr;
        CAPTURE(serialize);
        REQUIRE(serialize.has_value());
        index = nullptr;
    }

    // load from a stream
    {
        index = ::generate_empty_hnsw(
            /*dim=*/17,
            /*max_degree=*/12,
            /*ef_construction=*/100);

        auto deserialize = TestType::stream_deserialize(*index, buf);
        CAPTURE(deserialize);
        REQUIRE(deserialize.has_value());
    }
}

}  // namespace vsag
