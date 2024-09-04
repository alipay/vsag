
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

#include "./hnsw_zserialization.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "../exceptions.h"
#include "./hnsw.h"
#include "nlohmann/json_fwd.hpp"
#include "vsag/constants.h"

namespace vsag {

constexpr bool enable_v1 = true;
constexpr bool enable_v0 = true;

/* ---------------- interfaces ---------------- */
tl::expected<BinarySet, Error>
HnswSerialization::KvSerialize(const HNSW& hnsw, uint64_t version) {
    switch (version) {
        case 1:
            if constexpr (enable_v1) {
                return v1::KvSerialize(hnsw);
            }
            // goto next if disable
        case 0:
            if constexpr (enable_v0) {
                return v0::KvSerialize(hnsw);
            }
            // goto default if disable
        default: {
            return tl::unexpected(
                Error(ErrorType::UNKNOWN_ERROR,
                      "failed to serialize index with version " + std::to_string(version)));
        }
    }
}

tl::expected<void, Error>
HnswSerialization::KvDeserialize(HNSW& hnsw, const BinarySet& binary_set) {
    //  check hnsw object
    if (hnsw.alg_hnsw->getCurrentElementCount() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty");
    }

    // try v1
    if constexpr (enable_v1) {  // if stmt for debugging
        try {
            v1::KvDeserialize(hnsw, binary_set);
            return {};
        } catch (const DesearializationException& e) {
            logger::debug("binaryset is not version 1 format");
        } catch (const std::exception& e) {
            LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR,
                                  "failed to (kv-)deserialize index(version 1): ",
                                  e.what());
        }
    }

    // try v0
    if constexpr (enable_v0) {  // if stmt for debugging
        try {
            return v0::KvDeserialize(hnsw, binary_set);
        } catch (...) {
            return tl::unexpected(
                Error(ErrorType::UNKNOWN_ERROR, "failed to (kv-)deserialize index(version 0)"));
        }
    }

    return tl::unexpected(Error(ErrorType::UNKNOWN_ERROR,
                                "failed to (kv-)deserialize index: cannot match any deserializer"));
}

tl::expected<void, Error>
HnswSerialization::KvDeserialize(HNSW& hnsw, const ReaderSet& reader_set) {
    // try v1
    if constexpr (enable_v1) {  // if stmt for debugging
        try {
            v1::KvDeserialize(hnsw, reader_set);
            return {};
        } catch (const DesearializationException& e) {
            logger::debug("binaryset is not version 1 format");
        } catch (const std::exception& e) {
            LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR,
                                  "failed to (kv-)deserialize index(version 1): ",
                                  e.what());
        }
    }

    // try v0
    if constexpr (enable_v0) {  // if stmt for debugging
        try {
            return v0::KvDeserialize(hnsw, reader_set);
        } catch (...) {
            return tl::unexpected(
                Error(ErrorType::UNKNOWN_ERROR, "failed to (kv-)deserialize index(version 0)"));
        }
    }

    return tl::unexpected(Error(ErrorType::UNKNOWN_ERROR,
                                "failed to (kv-)deserialize index: cannot match any deserializer"));
}

tl::expected<void, Error>
HnswSerialization::StreamingSerialize(const HNSW& hnsw,
                                      std::ostream& out_stream,
                                      uint64_t version) {
    switch (version) {
        case 1:
            if constexpr (enable_v1) {
                v1::StreamingSerialize(hnsw, out_stream);
                return {};
            }
            // goto next if disable
        case 0:
            if constexpr (enable_v0) {
                return v0::StreamingSerialize(hnsw, out_stream);
            }
            // goto default if disable
        default: {
            return tl::unexpected(Error(
                ErrorType::UNKNOWN_ERROR,
                "failed to (streaming-)serialize index with version " + std::to_string(version)));
        }
    }
}

tl::expected<void, Error>
HnswSerialization::StreamingDeserialize(HNSW& hnsw, std::istream& in_stream) {
    //  check hnsw object
    if (hnsw.alg_hnsw->getCurrentElementCount() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty");
    }

    // try v1
    if constexpr (enable_v1) {  // if stmt for debugging
        try {
            v1::StreamingDeserialize(hnsw, in_stream);
            return {};
        } catch (const DesearializationException& e) {
            logger::debug("binaryset is not version 1 format");
        } catch (const std::exception& e) {
            LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR,
                                  "failed to (streaming-)deserialize index(version 1): ",
                                  e.what());
        }
    }

    // try v0
    if constexpr (enable_v0) {  // if stmt for debugging
        try {
            return v0::StreamingDeserialize(hnsw, in_stream);
        } catch (...) {
            return tl::unexpected(
                Error(ErrorType::UNKNOWN_ERROR, "failed to (stream-)deserialize index(version 0)"));
        }
    }

    return tl::unexpected(Error(ErrorType::UNKNOWN_ERROR,
                                "failed to (kv-)deserialize index: cannot match any deserializer"));
}

/* ---------------- v1 format ---------------- */
tl::expected<BinarySet, Error>
HnswSerialization::v1::KvSerialize(const HNSW& hnsw) {
    SlowTaskTimer t("hnsw serialize");

    BinarySet bs;
    nlohmann::json metadata = hnsw.Metadata();

    // empty check
    if (hnsw.GetNumElements() == 0) {
        metadata["empty"] = true;
        goto mdata;
    }

    // major graph
    {
        size_t num_bytes = hnsw.alg_hnsw->calcSerializeSize();
        std::shared_ptr<int8_t[]> index_buf(new int8_t[num_bytes]);
        hnsw.alg_hnsw->saveIndex(index_buf.get());
        Binary b_index{
            .data = index_buf,
            .size = num_bytes,
        };
        bs.Set(HNSW_DATA, b_index);
    }

    // conjugate graph
    if (hnsw.use_conjugate_graph_) {
        // TODO(wxyu): process error
        Binary b_cg = *hnsw.conjugate_graph_->Serialize();
        bs.Set(CONJUGATE_GRAPH_DATA, b_cg);
    }

    // add here if new part

mdata:
    // example mdata structure:
    //  - empty:
    //    { "version":1,"empty":true }
    //  - normal:
    //    { "version":1,"num_elements":1000,"dim":32,"raw_parameters":"..." }
    //
    metadata["version"] = 1;
    auto mdata = metadata.dump();
    logger::debug("serialize metadata: " + mdata);
    auto mdata_size = mdata.size();
    auto mdata_buf = std::shared_ptr<int8_t[]>(new int8_t[mdata.size()]);
    memcpy(mdata_buf.get(), mdata.c_str(), mdata.size());
    Binary b_mdata{
        .data = mdata_buf,
        .size = mdata.size(),
    };
    bs.Set("_mdata", b_mdata);

    // binaryset structure:
    //   - "_mdata": {...}
    //   - "hnsw_data": [...]
    //   - "conjugate_graph_data": [...]
    //
    return bs;
}

tl::expected<void, Error>
HnswSerialization::v1::KvDeserialize(HNSW& hnsw, const BinarySet& binary_set) {
    SlowTaskTimer t("hnsw deserialize");

    // check metadata
    if (not binary_set.Contains("_mdata")) {
        // not v1
        throw DesearializationException("older than v1, not contains metadata");
    }

    auto b_mdata = binary_set.Get("_mdata");
    auto metadata = nlohmann::json::parse(std::string((char*)b_mdata.data.get(), b_mdata.size));
    if (not metadata.contains("version")) {
        throw DesearializationException("unexpected: version must in");
    }
    if (metadata["version"] != 1) {
        throw DesearializationException("version not match");
    }

    // check if it's an empty index
    if (metadata.contains("empty") && metadata["empty"]) {
        // deserialize a special index that is empty
        hnsw.empty_index_ = true;
        return {};
    }

    try {
        // deserialize major graph
        Binary b_index = binary_set.Get(HNSW_DATA);
        auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
            std::memcpy(dest, b_index.data.get() + offset, len);
        };

        hnsw.alg_hnsw->loadIndex(func, hnsw.space.get());

        // deserialize conjugate graph
        if (hnsw.use_conjugate_graph_) {
            Binary b_cg = binary_set.Get(CONJUGATE_GRAPH_DATA);
            if (not hnsw.conjugate_graph_->Deserialize(b_cg).has_value()) {
                throw std::runtime_error("error in deserialize conjugate graph");
            }
        }
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    } catch (const std::out_of_range& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

tl::expected<void, Error>
HnswSerialization::v1::KvDeserialize(HNSW& hnsw, const ReaderSet& reader_set) {
    SlowTaskTimer t("hnsw deserialize");

    // check metadata
    if (not reader_set.Contains("_mdata")) {
        // not v1
        throw DesearializationException("older than v1, not contains metadata");
    }

    auto r_mdata = reader_set.Get("_mdata");
    auto mdata_buf = std::shared_ptr<int8_t[]>(new int8_t[r_mdata->Size()]);
    r_mdata->Read(0, r_mdata->Size(), mdata_buf.get());
    auto metadata = nlohmann::json::parse(std::string((char*)mdata_buf.get(), r_mdata->Size()));
    if (not metadata.contains("version")) {
        throw DesearializationException("unexpected: version must in");
    }
    if (metadata["version"] != 1) {
        throw DesearializationException("version not match");
    }

    // check if it's an empty index
    if (metadata.contains("empty") && metadata["empty"]) {
        // deserialize a special index that is empty
        hnsw.empty_index_ = true;
        return {};
    }

    try {
        // deserialize via reader
        auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
            reader_set.Get(HNSW_DATA)->Read(offset, len, dest);
        };

        hnsw.alg_hnsw->loadIndex(func, hnsw.space.get());
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

tl::expected<void, Error>
HnswSerialization::v1::StreamingSerialize(const HNSW& hnsw, std::ostream& out_stream) {
    SlowTaskTimer t("hnsw serialize");

    // TODO(wxyu): unify check
    if (hnsw.GetNumElements() == 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_EMPTY, "failed to serialize: hnsw index is empty");

        // FIXME(wxyu): cannot support serialize empty index by stream
        // auto bs = empty_binaryset();
        // for (const auto& key : bs.GetKeys()) {
        //     auto b = bs.Get(key);
        //     out_stream.write((char*)b.data.get(), b.size);
        // }
        // return {};
    }

    // part 1: major graph
    // no expected exception
    hnsw.alg_hnsw->saveIndex(out_stream);

    // part 2: conjugate graph
    if (hnsw.use_conjugate_graph_) {
        hnsw.conjugate_graph_->Serialize(out_stream);
    }

    // last part: footer, fixed to 4kb, padding with 0
    std::string footer = hnsw.Metadata().dump();
    if ((sizeof("VSAG") + footer.size() > 4096)) {
        return tl::unexpected(
            Error(ErrorType::INTERNAL_ERROR, "failed to serialize: footer length excceed 4kb"));
    }
    char buffer[4096] = {};
    memcpy(buffer, "VSAG", 4);
    memcpy(buffer + 4, footer.c_str(), footer.length());
    out_stream.write(buffer, 4096);

    return {};
}

tl::expected<void, Error>
HnswSerialization::v1::StreamingDeserialize(HNSW& hnsw, std::istream& in_stream) {
    SlowTaskTimer t("hnsw deserialize");

    // footer part: check and parse the metadata
    in_stream.seekg(0, std::ios::end);
    uint64_t length = in_stream.tellg();
    logger::debug("in_stream.length=" + std::to_string(length));
    if (length < 4096) {
        throw DesearializationException("not v1");
    }

    in_stream.seekg(-4096, std::ios::end);
    char buffer[4096] = {};
    in_stream.read(buffer, 4096);
    in_stream.seekg(0, std::ios::beg);

    const char* VSAG = "VSAG";
    if (std::memcmp(buffer, VSAG, 4) != 0) {
        throw DesearializationException("not v1");
    }

    auto metadata = nlohmann::json::parse(std::string(buffer + 4, 4096 - 4));
    if (not metadata.contains("version") or metadata["version"] != 1) {
        throw DesearializationException("not v1");
    }

    try {
        // part 1: major graph
        hnsw.alg_hnsw->loadIndex(in_stream, hnsw.space.get());

        // part 2: conjugate graph
        if (hnsw.use_conjugate_graph_ and
            not hnsw.conjugate_graph_->Deserialize(in_stream).has_value()) {
            throw std::runtime_error("error in deserialize conjugate graph");
        }
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

/* ---------------- v0 format ---------------- */
tl::expected<BinarySet, Error>
HnswSerialization::v0::KvSerialize(const HNSW& hnsw) {
    if (hnsw.GetNumElements() == 0) {
        // return a special binaryset means empty
        return empty_binaryset();
    }

    SlowTaskTimer t("hnsw serialize");
    size_t num_bytes = hnsw.alg_hnsw->calcSerializeSize();
    try {
        std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
        hnsw.alg_hnsw->saveIndex(bin.get());
        Binary b{
            .data = bin,
            .size = num_bytes,
        };
        BinarySet bs;
        bs.Set(HNSW_DATA, b);

        if (hnsw.use_conjugate_graph_) {
            Binary b_cg = *hnsw.conjugate_graph_->Serialize();
            bs.Set(CONJUGATE_GRAPH_DATA, b_cg);
        }

        return bs;
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::NO_ENOUGH_MEMORY, "failed to serialize(bad alloc): ", e.what());
    }
}

tl::expected<void, Error>
HnswSerialization::v0::KvDeserialize(HNSW& hnsw, const BinarySet& binary_set) {
    SlowTaskTimer t("hnsw deserialize");

    // check if binaryset is a empty index
    if (binary_set.Contains(BLANK_INDEX)) {
        hnsw.empty_index_ = true;
        return {};
    }

    try {
        Binary b_index = binary_set.Get(HNSW_DATA);
        auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
            std::memcpy(dest, b_index.data.get() + offset, len);
        };

        hnsw.alg_hnsw->loadIndex(func, hnsw.space.get());
        if (hnsw.use_conjugate_graph_) {
            Binary b_cg = binary_set.Get(CONJUGATE_GRAPH_DATA);
            if (not hnsw.conjugate_graph_->Deserialize(b_cg).has_value()) {
                throw std::runtime_error("error in deserialize conjugate graph");
            }
        }
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    } catch (const std::out_of_range& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

tl::expected<void, Error>
HnswSerialization::v0::KvDeserialize(HNSW& hnsw, const ReaderSet& reader_set) {
    SlowTaskTimer t("hnsw deserialize");

    // check if readerset is a empty index
    if (reader_set.Contains(BLANK_INDEX)) {
        hnsw.empty_index_ = true;
        return {};
    }

    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        reader_set.Get(HNSW_DATA)->Read(offset, len, dest);
    };

    try {
        hnsw.alg_hnsw->loadIndex(func, hnsw.space.get());
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

tl::expected<void, Error>
HnswSerialization::v0::StreamingSerialize(const HNSW& hnsw, std::ostream& out_stream) {
    if (hnsw.GetNumElements() == 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_EMPTY, "failed to serialize: hnsw index is empty");

        // FIXME(wxyu): cannot support serialize empty index by stream
        // auto bs = empty_binaryset();
        // for (const auto& key : bs.GetKeys()) {
        //     auto b = bs.Get(key);
        //     out_stream.write((char*)b.data.get(), b.size);
        // }
        // return {};
    }

    SlowTaskTimer t("hnsw serialize");

    // no expected exception
    hnsw.alg_hnsw->saveIndex(out_stream);

    if (hnsw.use_conjugate_graph_) {
        hnsw.conjugate_graph_->Serialize(out_stream);
    }

    return {};
}

tl::expected<void, Error>
HnswSerialization::v0::StreamingDeserialize(HNSW& hnsw, std::istream& in_stream) {
    SlowTaskTimer t("hnsw deserialize");

    try {
        hnsw.alg_hnsw->loadIndex(in_stream, hnsw.space.get());
        if (hnsw.use_conjugate_graph_ and
            not hnsw.conjugate_graph_->Deserialize(in_stream).has_value()) {
            throw std::runtime_error("error in deserialize conjugate graph");
        }
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

BinarySet
HnswSerialization::v0::empty_binaryset() {
    // version 0 pairs:
    // - hnsw_blank: b"EMPTY_HNSW"
    const std::string empty_str = "EMPTY_HNSW";
    size_t num_bytes = empty_str.length();
    std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
    memcpy(bin.get(), empty_str.c_str(), empty_str.length());
    Binary b{
        .data = bin,
        .size = num_bytes,
    };
    BinarySet bs;
    bs.Set(BLANK_INDEX, b);

    return bs;
}

}  // namespace vsag
