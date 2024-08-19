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

#include "./diskann_zserialization.h"

#include <cstring>

#include "./diskann.h"
#include "vsag/constants.h"

namespace vsag {

tl::expected<BinarySet, Error>
DiskannSerialization::Serialize(const DiskANN& diskann) {
    if (diskann.status_ == IndexStatus::EMPTY) {
        // return a special binaryset means empty
        return EmptyBinaryset();
    }

    SlowTaskTimer t("diskann serialize");
    try {
        BinarySet bs;

        bs.Set(DISKANN_PQ, ConvertStreamToBinary(diskann.pq_pivots_stream_));
        bs.Set(DISKANN_COMPRESSED_VECTOR,
               ConvertStreamToBinary(diskann.disk_pq_compressed_vectors_));
        bs.Set(DISKANN_LAYOUT_FILE, ConvertStreamToBinary(diskann.disk_layout_stream_));
        bs.Set(DISKANN_TAG_FILE, ConvertStreamToBinary(diskann.tag_stream_));
        if (diskann.preload_) {
            bs.Set(DISKANN_GRAPH, ConvertStreamToBinary(diskann.graph_stream_));
        }
        return bs;
    } catch (const std::bad_alloc& e) {
        return tl::unexpected(Error(ErrorType::NO_ENOUGH_MEMORY, ""));
    }
}

tl::expected<void, Error>
DiskannSerialization::Deserialize(DiskANN& diskann, const BinarySet& binary_set) {
    SlowTaskTimer t("diskann deserialize");
    if (diskann.index_) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty")
    }

    // check if binaryset is a empty index
    if (binary_set.Contains(BLANK_INDEX)) {
        diskann.empty_index_ = true;
        return {};
    }

    ConvertBinaryToStream(binary_set.Get(DISKANN_LAYOUT_FILE), diskann.disk_layout_stream_);
    auto graph = binary_set.Get(DISKANN_GRAPH);
    if (diskann.preload_) {
        if (graph.data) {
            ConvertBinaryToStream(graph, diskann.graph_stream_);
        } else {
            LOG_ERROR_AND_RETURNS(
                ErrorType::MISSING_FILE,
                fmt::format("missing file: {} when deserialize diskann index", DISKANN_GRAPH));
        }
    } else {
        if (graph.data) {
            logger::warn("serialize without using file: {} ", DISKANN_GRAPH);
        }
    }
    diskann.load_disk_index(binary_set);
    diskann.status_ = IndexStatus::MEMORY;

    return {};
}

tl::expected<void, Error>
DiskannSerialization::Deserialize(DiskANN& diskann, const ReaderSet& reader_set) {
    SlowTaskTimer t("diskann deserialize");

    if (diskann.index_) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              fmt::format("failed to deserialize: {} is not empty", INDEX_DISKANN));
    }

    // check if readerset is a empty index
    if (reader_set.Contains(BLANK_INDEX)) {
        diskann.empty_index_ = true;
        return {};
    }

    std::stringstream pq_pivots_stream, disk_pq_compressed_vectors, graph, tag_stream;

    {
        auto pq_reader = reader_set.Get(DISKANN_PQ);
        auto pq_pivots_data = std::make_unique<char[]>(pq_reader->Size());
        pq_reader->Read(0, pq_reader->Size(), pq_pivots_data.get());
        pq_pivots_stream.write(pq_pivots_data.get(), pq_reader->Size());
        pq_pivots_stream.seekg(0);
    }

    {
        auto compressed_vector_reader = reader_set.Get(DISKANN_COMPRESSED_VECTOR);
        auto compressed_vector_data = std::make_unique<char[]>(compressed_vector_reader->Size());
        compressed_vector_reader->Read(
            0, compressed_vector_reader->Size(), compressed_vector_data.get());
        disk_pq_compressed_vectors.write(compressed_vector_data.get(),
                                         compressed_vector_reader->Size());
        disk_pq_compressed_vectors.seekg(0);
    }

    {
        auto tag_reader = reader_set.Get(DISKANN_TAG_FILE);
        auto tag_data = std::make_unique<char[]>(tag_reader->Size());
        tag_reader->Read(0, tag_reader->Size(), tag_data.get());
        tag_stream.write(tag_data.get(), tag_reader->Size());
        tag_stream.seekg(0);
    }

    diskann.disk_layout_reader_ = reader_set.Get(DISKANN_LAYOUT_FILE);
    diskann.reader_.reset(new LocalFileReader(diskann.batch_read_));
    diskann.index_.reset(new diskann::PQFlashIndex<float, int64_t>(
        diskann.reader_, diskann.metric_, diskann.sector_len_, diskann.use_bsa_));
    diskann.index_->set_sector_size(Option::Instance().sector_size());
    diskann.index_->load_from_separate_paths(
        omp_get_num_procs(), pq_pivots_stream, disk_pq_compressed_vectors, tag_stream);

    auto graph_reader = reader_set.Get(DISKANN_GRAPH);
    if (diskann.preload_) {
        if (graph_reader) {
            auto graph_data = std::make_unique<char[]>(graph_reader->Size());
            graph_reader->Read(0, graph_reader->Size(), graph_data.get());
            graph.write(graph_data.get(), graph_reader->Size());
            graph.seekg(0);
            diskann.index_->load_graph(graph);
        } else {
            LOG_ERROR_AND_RETURNS(
                ErrorType::MISSING_FILE,
                fmt::format("miss file: {} when deserialize diskann index", DISKANN_GRAPH));
        }
    } else {
        if (graph_reader) {
            logger::warn("serialize without using file: {} ", DISKANN_GRAPH);
        }
    }
    diskann.status_ = IndexStatus::HYBRID;

    return {};
}

BinarySet
DiskannSerialization::EmptyBinaryset() {
    // version 0 pairs:
    // - hnsw_blank: b"EMPTY_DISKANN"
    const std::string empty_str = "EMPTY_DISKANN";
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

Binary
DiskannSerialization::ConvertStreamToBinary(const std::stringstream& stream) {
    std::streambuf* buf = stream.rdbuf();
    std::streamsize size = buf->pubseekoff(0, stream.end, stream.in);  // get the stream buffer size
    buf->pubseekpos(0, stream.in);                                     // reset pointer pos
    std::shared_ptr<int8_t[]> binary_data(new int8_t[size]);
    buf->sgetn((char*)binary_data.get(), size);
    Binary binary{
        .data = binary_data,
        .size = (size_t)size,
    };
    return std::move(binary);
}

void
DiskannSerialization::ConvertBinaryToStream(const Binary& binary, std::stringstream& stream) {
    stream.str("");
    if (binary.data && binary.size > 0) {
        stream.write((const char*)binary.data.get(), binary.size);
    }
}

}  // namespace vsag
