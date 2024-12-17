
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

#include "vsag/engine.h"

#include <fmt/format-inl.h>

#include <string>

#include "common.h"
#include "index/diskann.h"
#include "index/diskann_zparameters.h"
#include "index/hgraph_index.h"
#include "index/hgraph_zparameters.h"
#include "index/hnsw.h"
#include "index/hnsw_zparameters.h"
#include "index/index_common_param.h"
#include "resource_owner_wrapper.h"
#include "typing.h"

namespace vsag {

Engine::Engine(Resource* resource) {
    if (resource == nullptr) {
        this->resource_ = std::make_shared<ResourceOwnerWrapper>(new Resource(), /*owned*/ true);
    } else {
        this->resource_ = std::make_shared<ResourceOwnerWrapper>(resource, /*owned*/ false);
    }
}

void
Engine::Shutdown() {
    this->resource_.reset();
}

tl::expected<std::shared_ptr<Index>, Error>
Engine::CreateIndex(const std::string& origin_name, const std::string& parameters) {
    try {
        auto* allocator = this->resource_->allocator.get();
        std::string name = origin_name;
        transform(name.begin(), name.end(), name.begin(), ::tolower);
        JsonType parsed_params = JsonType::parse(parameters);
        auto index_common_params = IndexCommonParam::CheckAndCreate(parsed_params, allocator);
        if (name == INDEX_HNSW) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_HNSW),
                           fmt::format("parameters must contains {}", INDEX_HNSW));
            auto& hnsw_param_obj = parsed_params[INDEX_HNSW];
            auto hnsw_params = HnswParameters::FromJson(hnsw_param_obj, index_common_params);
            logger::debug("created a hnsw index");
            return std::make_shared<HNSW>(hnsw_params, index_common_params);
        } else if (name == INDEX_FRESH_HNSW) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_HNSW),
                           fmt::format("parameters must contains {}", INDEX_HNSW));
            auto& hnsw_param_obj = parsed_params[INDEX_HNSW];
            auto hnsw_params = FreshHnswParameters::FromJson(hnsw_param_obj, index_common_params);
            logger::debug("created a fresh-hnsw index");
            return std::make_shared<HNSW>(hnsw_params, index_common_params);
        } else if (name == INDEX_DISKANN) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_DISKANN),
                           fmt::format("parameters must contains {}", INDEX_DISKANN));
            auto& diskann_param_obj = parsed_params[INDEX_DISKANN];
            auto diskann_params =
                DiskannParameters::FromJson(diskann_param_obj, index_common_params);
            logger::debug("created a diskann index");
            return std::make_shared<DiskANN>(diskann_params, index_common_params);
        } else if (name == INDEX_HGRAPH) {
            if (allocator == nullptr) {
                index_common_params.allocator_ = DefaultAllocator::Instance().get();
            }
            logger::debug("created a hgraph index");
            JsonType hgraph_params;
            if (parsed_params.contains(INDEX_PARAM)) {
                hgraph_params = std::move(parsed_params[INDEX_PARAM]);
            }
            HGraphParameters hgraph_param(hgraph_params, index_common_params);
            auto hgraph_index =
                std::make_shared<HGraphIndex>(hgraph_param.GetJson(), index_common_params);
            hgraph_index->Init();
            return hgraph_index;
        } else {
            LOG_ERROR_AND_RETURNS(
                ErrorType::UNSUPPORTED_INDEX, "failed to create index(unsupported): ", name);
        }
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to create index(invalid argument): ", e.what());
    } catch (const std::exception& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::UNSUPPORTED_INDEX, "failed to create index(unknown error): ", e.what());
    }
}
}  // namespace vsag
