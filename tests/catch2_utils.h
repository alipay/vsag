
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

#include <vsag/errors.h>

#include <catch2/catch_test_macros.hpp>
#include <vsag/expected.hpp>

namespace Catch {
template <typename T>
struct StringMaker<tl::expected<T, vsag::Error>> {
    static std::string
    convert(tl::expected<T, vsag::Error> const& value) {
        if (value.has_value()) {
            return "";
        } else {
            return "null, error message: " + value.error().message;
        }
    }
};
}  // namespace Catch
