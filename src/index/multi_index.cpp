
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

#include "multi_index.h"

namespace vsag {

// Function to convert BinarySet to a Binary
Binary
binaryset_to_binary(const BinarySet binarySet) {
    // 计算总大小
    size_t totalSize = 0;
    auto keys = binarySet.GetKeys();

    for (const auto& key : keys) {
        totalSize += sizeof(size_t) + key.size();  // key 的大小
        totalSize += sizeof(size_t);               // Binary.size 的大小
        totalSize += binarySet.Get(key).size;      // Binary.data 的大小
    }

    // 创建一个足够大的 Binary
    Binary result;
    result.data = std::shared_ptr<int8_t[]>(new int8_t[totalSize]);
    result.size = totalSize;

    size_t offset = 0;

    // 编码 keys 和对应的 Binaries
    for (const auto& key : keys) {
        // 复制 key 大小和内容
        size_t keySize = key.size();
        memcpy(result.data.get() + offset, &keySize, sizeof(size_t));
        offset += sizeof(size_t);
        memcpy(result.data.get() + offset, key.data(), keySize);
        offset += keySize;

        // 获取 Binary 对象
        Binary binary = binarySet.Get(key);
        // 复制 Binary 大小和内容
        memcpy(result.data.get() + offset, &binary.size, sizeof(size_t));
        offset += sizeof(size_t);
        memcpy(result.data.get() + offset, binary.data.get(), binary.size);
        offset += binary.size;
    }

    return result;
}

// 从 Binary 解码恢复 BinarySet
BinarySet
binary_to_binaryset(const Binary binary) {
    BinarySet binarySet;
    size_t offset = 0;

    while (offset < binary.size) {
        // 读取 key 的大小
        size_t keySize;
        memcpy(&keySize, binary.data.get() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        // 读取 key 的内容
        std::string key(reinterpret_cast<const char*>(binary.data.get() + offset), keySize);
        offset += keySize;

        // 读取 Binary 大小
        size_t binarySize;
        memcpy(&binarySize, binary.data.get() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        // 读取 Binary 数据
        Binary newBinary;
        newBinary.size = binarySize;
        newBinary.data = std::shared_ptr<int8_t[]>(new int8_t[binarySize]);
        memcpy(newBinary.data.get(), binary.data.get() + offset, binarySize);
        offset += binarySize;

        // 将新 Binary 放入 BinarySet
        binarySet.Set(key, newBinary);
    }

    return binarySet;
}


ReaderSet
reader_to_readerset(std::shared_ptr<Reader> reader) {
    ReaderSet readerSet;
    size_t offset = 0;

    while (offset < reader->Size()) {
        // 读取 key 的大小
        size_t keySize;
        reader->Read(offset, sizeof(size_t), &keySize);
        offset += sizeof(size_t);
        // 读取 key 的内容
        std::shared_ptr<char[]> key_chars = std::shared_ptr<char[]>(new char[keySize]);
        reader->Read(offset, keySize, key_chars.get());
        std::string key(key_chars.get(), keySize);
        offset += keySize;

        // 读取 Binary 大小
        size_t binarySize;
        reader->Read(offset, sizeof(size_t), &binarySize);
        offset += sizeof(size_t);

        // 读取 Binary 数据
        auto newReader = std::shared_ptr<SubReader>(new SubReader(reader, offset, binarySize));
        offset += binarySize;

        // 将新 Binary 放入 BinarySet
        readerSet.Set(key, newReader);
    }

    return readerSet;
}


}  // namespace vsag