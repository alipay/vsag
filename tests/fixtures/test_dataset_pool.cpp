//
// Created by root on 12/16/24.
//

#include "test_dataset_pool.h"

namespace fixtures {
TestDatasetPtr
TestDatasetPool::GetDatasetAndCreate(uint64_t dim,
                                     uint64_t count,
                                     const std::string& metric_str,
                                     bool with_path) {
    auto key = key_gen(dim, count, metric_str, with_path);
    if (this->pool_.find(key) == this->pool_.end()) {
        this->dim_counts_.emplace_back(dim, count);
        this->pool_[key] = std::make_shared<TestDataset>(dim, count, metric_str, with_path);
    }
    return this->pool_.at(key);
}
std::string
TestDatasetPool::key_gen(int64_t dim,
                         uint64_t count,
                         const std::string& metric_str,
                         bool with_path) {
    return std::to_string(dim) + "_" + std::to_string(count) + "_" + metric_str + "_" +
           std::to_string(with_path);
}
}  // namespace fixtures
