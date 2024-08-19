//
// Created by root on 8/16/24.
//

#ifndef GITHUB_VSAG_PQCODES_H
#define GITHUB_VSAG_PQCODES_H

#include <vector>
#include <cstdint>
#include <string>

class PQCodes {
public:

    PQCodes(int subspace, int dim);

    int subSpace_;

    int nbit_ = 4;

    int centerCount_ = 16;

    int dim_;

    int dimPerSpace_;

    std::vector<float> codebook;

    void Train(const float* data, int64_t count);

    void BatchEncode(const float* data, int64_t count, std::vector<uint8_t>& codes);

    void Packaged(std::vector<uint8_t>& codes);

    void ToFile(std::string filepath);

    void FromFile(std::string filepath);
};

class PQScanner {
public:

    explicit PQScanner(PQCodes* pq);

    PQCodes* pqCodes_ = nullptr;

    std::vector<uint8_t> lut_;

    float sqMax_{-10000};

    float sqMin_{10000};

    void SetQuery(float* query);

    void ScanCodes(const uint8_t* codes, std::vector<float>& dists);

};


#endif //GITHUB_VSAG_PQCODES_H
