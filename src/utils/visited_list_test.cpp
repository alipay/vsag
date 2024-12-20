
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

#include "visited_list.h"

#include <thread>

#include "catch2/catch_test_macros.hpp"
#include "default_allocator.h"
using namespace vsag;

TEST_CASE("test visited_list basic", "[ut][visited_list]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    auto size = 10000;
    auto vl_ptr = std::make_shared<VisitedList>(size, allocator.get());

    SECTION("test set & get normal") {
        int count = 500;
        std::unordered_set<InnerIdType> ids;
        for (int i = 0; i < count; ++i) {
            auto id = random() % size;
            ids.insert(id);
            vl_ptr->Set(id);
        }
        for (auto& id : ids) {
            REQUIRE(vl_ptr->Get(id));
        }

        for (int i = 0; i < size; ++i) {
            if (ids.count(i) == 0) {
                REQUIRE(not vl_ptr->Get(i));
            }
        }
    }

    SECTION("test reset") {
        int count = 500;
        std::unordered_set<InnerIdType> ids;
        for (int i = 0; i < count; ++i) {
            auto id = random() % size;
            ids.insert(id);
            vl_ptr->Set(id);
        }
        vl_ptr->Reset();
        for (auto& id : ids) {
            REQUIRE(not vl_ptr->Get(id));
        }
    }
}

TEST_CASE("test visited_list_pool basic", "[ut][visited_list_pool]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    auto init_size = 10;
    auto vl_size = 1000;
    auto pool = std::make_shared<VisitedListPool>(init_size, vl_size, allocator.get());

    auto TestVL = [&](std::shared_ptr<VisitedList>& vl_ptr) {
        int count = 500;
        std::unordered_set<InnerIdType> ids;
        for (int i = 0; i < count; ++i) {
            auto id = random() % vl_size;
            ids.insert(id);
            vl_ptr->Set(id);
        }
        for (auto& id : ids) {
            REQUIRE(vl_ptr->Get(id) == true);
        }

        for (InnerIdType i = 0; i < vl_size; ++i) {
            if (ids.count(i) == 0) {
                REQUIRE(vl_ptr->Get(i) == false);
            }
        }
    };

    SECTION("test basic") {
        std::vector<std::shared_ptr<VisitedList>> lists;
        REQUIRE(pool->GetSize() == init_size);
        lists.reserve(init_size * 2);
        for (auto i = 0; i < init_size * 2; ++i) {
            lists.emplace_back(pool->TakeOne());
        }
        REQUIRE(pool->GetSize() == 0);
        for (auto& ptr : lists) {
            pool->ReturnOne(ptr);
        }
        REQUIRE(pool->GetSize() == init_size * 2);

        auto ptr = pool->TakeOne();
        REQUIRE(pool->GetSize() == init_size * 2 - 1);
        TestVL(ptr);
    }

    SECTION("test concurrency") {
        auto func = [&]() {
            int count = 10;
            int max_operators = 20;
            std::vector<std::shared_ptr<VisitedList>> results;
            for (int i = 0; i < count; ++i) {
                auto opt = random() % max_operators + 1;
                for (auto j = 0; j < opt; ++j) {
                    results.emplace_back(pool->TakeOne());
                }
                auto test = results[random() % opt];
                TestVL(test);
                for (auto& result : results) {
                    pool->ReturnOne(result);
                }
                results.clear();
            }
        };
        std::vector<std::shared_ptr<std::thread>> ths;
        auto thread_count = 2;
        for (auto i = 0; i < thread_count; ++i) {
            ths.emplace_back((std::make_shared<std::thread>(func)));
        }
        for (auto& thread : ths) {
            thread->join();
        }
    }
}