
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

#include "stream_reader.h"
ReadFuncStreamReader::ReadFuncStreamReader(
    const std::function<void(uint64_t, uint64_t, void*)>& read_func,
    uint64_t cursor,
    size_t max_size)
    : readFunc_(read_func), cursor_(cursor), max_size_(max_size), StreamReader() {
}

void
ReadFuncStreamReader::Read(char* data, uint64_t size) {
    readFunc_(cursor_, size, data);
    cursor_ += size;
}

size_t
ReadFuncStreamReader::Size() const {
    return max_size_;
}

IOStreamReader::IOStreamReader(std::istream& istream) : istream_(istream), StreamReader() {
    istream_.seekg(0, std::ios::end);
    std::streamsize length = istream_.tellg();
    istream_.seekg(0, std::ios::beg);
}

void
IOStreamReader::Read(char* data, uint64_t size) {
    this->istream_.read(data, static_cast<int64_t>(size));
}

size_t
IOStreamReader::Size() const {
    return max_size_;
}
