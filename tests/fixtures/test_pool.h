
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
#include <shared_mutex>
#include <unordered_map>

namespace fixtures {

template <typename T>
class TestPool {
public:
    static bool
    Set(const std::string& key, std::shared_ptr<T> value) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        if (data.find(key) != data.end()) {
            return false;
        }
        data[key] = value;
        return true;
    }

    static std::shared_ptr<T>
    Get(const std::string& key) {
        std::shared_lock<std::shared_mutex> lock(mutex);
        auto iter = data.find(key);
        if (iter == data.end()) {
            return nullptr;
        }
        return iter->second;
    }

    static bool
    CheckInit() {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return inited;
    }

    static void
    SetInit() {
        std::unique_lock<std::shared_mutex> lock(mutex);
        inited = true;
    }

private:
    static std::unordered_map<std::string, std::shared_ptr<T>> data;

    static std::shared_mutex mutex;

    static bool inited;
};

template <typename T>
bool TestPool<T>::inited = false;

template <typename T>
std::unordered_map<std::string, std::shared_ptr<T>> TestPool<T>::data = {};

template <typename T>
std::shared_mutex TestPool<T>::mutex{};
}  // namespace fixtures