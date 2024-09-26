
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

#include <cstdint>
#include <vector>

namespace vsag {
/**
 * @class GraphInterface
 * @brief This class provides an interface for graph operations
 */
class GraphInterface {
public:
    /**
     * @brief Insert a node into the graph with given neighbor ids
     * @param neighbor_ids The ids of the neighbors of the node being inserted
     * @return The id of the inserted node
     */
    virtual uint64_t
    InsertNode(const std::vector<uint64_t>& neighbor_ids) {
        return 0;
    }

    /**
     * @brief Insert neighbors for a node with the given id
     * @param id The id of the node
     * @param neighbor_ids The ids of the neighbors to be inserted
     */
    virtual void
    InsertNeighbors(uint64_t id, const std::vector<uint64_t>& neighbor_ids){};

    /**
     * @brief Get the size of the neighbors for a node with the given id
     * @param id The id of the node
     * @return The size of the neighbors
     */
    virtual uint32_t
    GetNeighborSize(uint64_t id) {
        return 0;
    }

    /**
     * @brief Get the neighbors of a node with the given id
     * @param id The id of the node
     * @param neighbor_ids The vector to store the neighbor ids
     */
    virtual void
    GetNeighbors(uint64_t id, std::vector<uint64_t>& neighbor_ids){};

    /**
     * @brief Prefetch neighbor data for a node with the given id
     * @param id The id of the node
     * @param neighbor_i The index of the neighbor to prefetch
     */
    virtual void
    Prefetch(uint64_t id, uint64_t neighbor_i){};

    /**
     * @brief Set the maximum capacity of the graph
     * @param capacity The maximum capacity to be set
     */
    virtual void
    SetMaxCapacity(uint64_t capacity){};

    /**
     * @brief Get the total count of nodes in the graph
     * @return The total count of nodes
     */
    virtual uint64_t
    TotalCount() {
        return 0;
    }

    /**
     * @brief Get the maximum degree of any node in the graph
     * @return The maximum degree
     */
    virtual uint32_t
    GetMaximumDegree() {
        return 0;
    }

    /**
     * @brief Set the maximum degree of any node in the graph
     * @param maximum_degree The maximum degree to be set
     */
    virtual void
    SetMaximumDegree(uint32_t maximum_degree){};

    /**
     * @brief Set the total count of nodes in the graph
     * @param total_count The total count to be set
     */
    virtual void
    SetTotalCount(uint64_t total_count){};
};

}  // namespace vsag