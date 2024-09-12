
#pragma once
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "io/basic_io.h"
#include "quantization/quantizer.h"

namespace vsag {

/**
 * built by nn-descent or incremental insertion
 * add neighbors and pruning
 * retrieve neighbors
 */

template <typename IOTmpl>
class GraphDataCell {
public:
    GraphDataCell(uint64_t maximum_degree = 32) : maximum_degree_(maximum_degree){};

    explicit GraphDataCell(const std::string& initializeJson);  // todo

    uint64_t
    InsertNode(const std::vector<uint64_t> neighbor_ids);

    void
    InsertNeighbors(uint64_t id, const std::vector<uint64_t> neighbor_ids) {  // todo
        return;
    };

    void
    Prune(uint64_t id) {
        return;
    }

    uint32_t
    GetNeighborSize(uint64_t id);

    void
    GetNeighbors(uint64_t id, std::vector<uint64_t>& neighbor_ids);

    inline void
    SetMaxCapacity(uint64_t capacity) {
        this->maxCapacity_ = std::max(capacity, this->totalCount_);  // TODO add warning
    }

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>>& io) {
        this->io_.swap(io);
    }

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>>&& io) {
        this->io_.swap(io);
    }

    inline uint64_t
    TotalCount() {
        return this->totalCount_;
    }

private:
    inline uint64_t
    GetSingleOffset() {
        return maximum_degree_ * sizeof(uint64_t) + sizeof(uint32_t);
    }

private:
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    uint64_t totalCount_{0};

    uint64_t maxCapacity_{1000000};

    uint64_t maximum_degree_{0};
};

template <typename IOTmpl>
uint64_t
GraphDataCell<IOTmpl>::InsertNode(const std::vector<uint64_t> neighbor_ids) {
    auto cur_offset = totalCount_ * this->GetSingleOffset();
    uint32_t neighbor_size = neighbor_ids.size();
    if (neighbor_size > this->maximum_degree_) {
        neighbor_size = maximum_degree_;
    }
    this->io_->Write(&neighbor_size, sizeof(neighbor_size), cur_offset);
    cur_offset += sizeof(neighbor_size);

    this->io_->Write(reinterpret_cast<const uint8_t*>(neighbor_ids.data()),
                     neighbor_size * sizeof(uint64_t),
                     cur_offset);
    cur_offset += neighbor_size * sizeof(uint64_t);

    totalCount_++;
    return totalCount_;
}

template <typename IOTmpl>
uint32_t
GraphDataCell<IOTmpl>::GetNeighborSize(uint64_t id) {
    uint32_t size = 0;
    if (id >= totalCount_) {
        return 0;
    }

    io_->Read(reinterpret_cast<uint8_t*>(&size), sizeof(size), id * this->GetSingleOffset());

    return size;
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::GetNeighbors(uint64_t id, std::vector<uint64_t>& neighbor_ids) {
    uint32_t size = GetNeighborSize(id);
    uint64_t cur_offset = id * this->GetSingleOffset() + sizeof(size);
    if (size == 0) {
        return;
    }
    neighbor_ids.resize(size, 0);
    for (int i = 0; i < size; i++) {
        uint64_t neighbor_id = 0;
        io_->Read(reinterpret_cast<uint8_t*>(&neighbor_id), sizeof(uint64_t), cur_offset);
        cur_offset += sizeof(uint64_t);
    }
}

}  // namespace vsag