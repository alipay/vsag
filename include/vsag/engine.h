
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

#include <memory>

#include "index.h"
#include "resource.h"

namespace vsag {
class Engine {
public:
    explicit Engine(Resource* resource = nullptr);

    void
    Shutdown();

    tl::expected<std::shared_ptr<Index>, Error>
    CreateIndex(const std::string& name, const std::string& parameters);

private:
    std::shared_ptr<Resource> resource_;
};
}  // namespace vsag
