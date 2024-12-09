
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

#include <fmt/format-inl.h>

#include "vsag/options.h"

ReadFuncStreamReader::ReadFuncStreamReader(
    const std::function<void(uint64_t, uint64_t, void*)> read_func, uint64_t cursor)
    : readFunc_(read_func), cursor_(cursor), StreamReader() {
}

void
ReadFuncStreamReader::Read(char* data, uint64_t size) {
    readFunc_(cursor_, size, data);
    cursor_ += size;
}

void
ReadFuncStreamReader::Seek(uint64_t cursor) {
    cursor_ = cursor;
}

uint64_t
ReadFuncStreamReader::GetCursor() const {
    return cursor_;
}

IOStreamReader::IOStreamReader(std::istream& istream) : istream_(istream), StreamReader() {
}

void
IOStreamReader::Read(char* data, uint64_t size) {
    this->istream_.read(data, static_cast<int64_t>(size));
    if (istream_.fail()) {
        auto remaining = std::streamsize(this->istream_.gcount());
        throw std::runtime_error(fmt::format(
            "Attempted to read: {} bytes. Remaining content size: {} bytes.", size, remaining));
    }
}

void
IOStreamReader::Seek(uint64_t cursor) {
    istream_.seekg(cursor, std::ios::beg);
}

uint64_t
IOStreamReader::GetCursor() const {
    uint64_t cursor = istream_.tellg();
    return cursor;
}

BufferStreamReader::BufferStreamReader(StreamReader* reader,
                                       size_t max_size,
                                       vsag::SafeAllocator* allocator)
    : reader_impl_(reader), max_size_(max_size), buffer_(allocator), StreamReader() {
    buffer_size_ = std::min(max_size_, vsag::Options::Instance().block_size_limit());
    buffer_.resize(buffer_size_);
    buffer_cursor_ = buffer_size_;
    valid_size_ = buffer_size_;
}

void
BufferStreamReader::Read(char* data, uint64_t size) {
    // Total bytes copied to dest
    size_t total_copied = 0;

    // Loop to read until read_size is satisfied
    while (total_copied < size) {
        // Calculate the available data in buffer_
        size_t available_in_src = valid_size_ - buffer_cursor_;

        // If there is available data in buffer_, copy it to dest
        if (available_in_src > 0) {
            size_t bytes_to_copy = std::min(size - total_copied, available_in_src);
            memcpy(data + total_copied, buffer_.data() + buffer_cursor_, bytes_to_copy);
            total_copied += bytes_to_copy;
            buffer_cursor_ += bytes_to_copy;
        }
        // If we have copied enough data, we can exit
        if (total_copied >= size) {
            break;
        }

        // If buffer_ is full, reset cursor and read new data from reader
        buffer_cursor_ = 0;  // Reset cursor to overwrite buffer_'s content
        valid_size_ = std::min(max_size_ - cursor_, buffer_size_);
        if (valid_size_ == 0) {
            throw std::runtime_error(
                "BufferStreamReader: The file size is smaller than the memory you want to read.");
        }
        reader_impl_->Read(buffer_.data(), valid_size_);
        cursor_ += valid_size_;
    }
}

void
BufferStreamReader::Seek(uint64_t cursor) {
    reader_impl_->Seek(cursor);
    buffer_cursor_ = valid_size_;  // record the invalidation of the buffer
    cursor_ = cursor;
}

uint64_t
BufferStreamReader::GetCursor() const {
    return reader_impl_->GetCursor() - (valid_size_ - buffer_cursor_);
}