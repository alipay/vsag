
// -*- c++ -*-

#include "clustering.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "algorithm/hnswlib/hnswlib.h"

namespace vsag {

struct RandomGenerator {
    std::mt19937 mt;

    /// random positive integer

    int
    rand_int(int max) {
        return mt() % max;
    }

    float
    rand_float() {
        return mt() / float(mt.max());
    }

    RandomGenerator(int64_t seed = 1234) {
        mt.seed(seed);
    }
};

void
rand_perm(int* perm, size_t n, int64_t seed) {
    for (size_t i = 0; i < n; i++) perm[i] = i;

    RandomGenerator rng(seed);

    for (size_t i = 0; i + 1 < n; i++) {
        int i2 = i + rng.rand_int(n - i);
        std::swap(perm[i], perm[i2]);
    }
}
float
fvec_norm_L2sqr(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}
void
fvec_renorm_L2(size_t d, size_t nx, float* __restrict x) {
    for (int64_t i = 0; i < nx; i++) {
        float* __restrict xi = x + i * d;

        float nr = fvec_norm_L2sqr(xi, d);

        if (nr > 0) {
            size_t j;
            const float inv_nr = 1.0 / sqrtf(nr);
            for (j = 0; j < d; j++) xi[j] *= inv_nr;
        }
    }
}
ClusteringParameters::ClusteringParameters()
    : niter(25),
      nredo(1),
      verbose(false),
      spherical(false),
      int_centroids(false),
      update_index(false),
      frozen_centroids(false),
      min_points_per_centroid(39),
      max_points_per_centroid(256),
      seed(1234),
      decode_block_size(32768) {
}
// 39 corresponds to 10000 / 256 -> to avoid warnings on PQ tests with randu10k

Clustering::Clustering(int d, int k) : d(d), k(k) {
}

Clustering::Clustering(int d, int k, const ClusteringParameters& cp)
    : ClusteringParameters(cp), d(d), k(k) {
}

static double
imbalance_factor(int n, int k, int64_t* assign) {
    std::vector<int> hist(k, 0);
    for (int i = 0; i < n; i++) hist[assign[i]]++;

    double tot = 0, uf = 0;

    for (int i = 0; i < k; i++) {
        tot += hist[i];
        uf += hist[i] * (double)hist[i];
    }
    uf = uf * k / (tot * tot);

    return uf;
}

void
Clustering::post_process_centroids() {
    if (spherical) {
        fvec_renorm_L2(d, k, centroids.data());
    }

    if (int_centroids) {
        for (size_t i = 0; i < centroids.size(); i++) centroids[i] = roundf(centroids[i]);
    }
}

void
Clustering::train(idx_t nx, const float* x_in, const float* weights) {
    train_encoded(nx, reinterpret_cast<const uint8_t*>(x_in), weights);
}

idx_t
subsample_training_set(const Clustering& clus,
                       idx_t nx,
                       const uint8_t* x,
                       size_t line_size,
                       const float* weights,
                       uint8_t** x_out,
                       float** weights_out) {
    if (clus.verbose) {
        printf("Sampling a subset of %zd / %" PRId64 " for training\n",
               clus.k * clus.max_points_per_centroid,
               nx);
    }
    std::vector<int> perm(nx);
    rand_perm(perm.data(), nx, clus.seed);
    nx = clus.k * clus.max_points_per_centroid;
    uint8_t* x_new = new uint8_t[nx * line_size];
    *x_out = x_new;
    for (idx_t i = 0; i < nx; i++) {
        memcpy(x_new + i * line_size, x + perm[i] * line_size, line_size);
    }
    if (weights) {
        float* weights_new = new float[nx];
        for (idx_t i = 0; i < nx; i++) {
            weights_new[i] = weights[perm[i]];
        }
        *weights_out = weights_new;
    } else {
        *weights_out = nullptr;
    }
    return nx;
}

/** compute centroids as (weighted) sum of training points
*
* @param x            training vectors, size n * code_size (from codec)
* @param codec        how to decode the vectors (if NULL then cast to float*)
* @param weights      per-training vector weight, size n (or NULL)
* @param assign       nearest centroid for each training vector, size n
* @param k_frozen     do not update the k_frozen first centroids
* @param centroids    centroid vectors (output only), size k * d
* @param hassign      histogram of assignments per centroid (size k),
*                     should be 0 on input
*
*/

void
compute_centroids(size_t d,
                  size_t k,
                  size_t n,
                  size_t k_frozen,
                  const uint8_t* x,
                  const int64_t* assign,
                  const float* weights,
                  float* hassign,
                  float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    memset(centroids, 0, sizeof(*centroids) * d * k);

    size_t line_size = d * sizeof(float);
    int nt = 1;
    for (int rank = 0; rank < nt; rank++) {
        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;
        std::vector<float> decode_buffer(d);

        for (size_t i = 0; i < n; i++) {
            int64_t ci = assign[i];
            assert(ci >= 0 && ci < k + k_frozen);
            ci -= k_frozen;
            if (ci >= c0 && ci < c1) {
                float* c = centroids + ci * d;
                const float* xi;
                xi = reinterpret_cast<const float*>(x + i * line_size);
                if (weights) {
                    float w = weights[i];
                    hassign[ci] += w;
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j] * w;
                    }
                } else {
                    hassign[ci] += 1.0;
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j];
                    }
                }
            }
        }
    }

    //#pragma omp parallel for
    for (idx_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        float norm = 1 / hassign[ci];
        float* c = centroids + ci * d;
        for (size_t j = 0; j < d; j++) {
            c[j] *= norm;
        }
    }
}

// a bit above machine epsilon for float16
#define EPS (1 / 1024.)

/** Handle empty clusters by splitting larger ones.
*
* It works by slightly changing the centroids to make 2 clusters from
* a single one. Takes the same arguments as compute_centroids.
*
* @return           nb of spliting operations (larger is worse)
*/
int
split_clusters(size_t d, size_t k, size_t n, size_t k_frozen, float* hassign, float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    /* Take care of void clusters */
    size_t nsplit = 0;
    RandomGenerator rng(1234);
    for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) { /* need to redefine a centroid */
            size_t cj;
            for (cj = 0; 1; cj = (cj + 1) % k) {
                /* probability to pick this cluster for split */
                float p = (hassign[cj] - 1.0) / (float)(n - k);
                float r = rng.rand_float();
                if (r < p) {
                    break; /* found our cluster to be split */
                }
            }

            memcpy(centroids + ci * d, centroids + cj * d, sizeof(*centroids) * d);

            /* small symmetric pertubation */
            for (size_t j = 0; j < d; j++) {
                if (j % 2 == 0) {
                    centroids[ci * d + j] *= 1 + EPS;
                    centroids[cj * d + j] *= 1 - EPS;
                } else {
                    centroids[ci * d + j] *= 1 - EPS;
                    centroids[cj * d + j] *= 1 + EPS;
                }
            }

            /* assume even split of the cluster */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
        }
    }

    return nsplit;
}

void
Clustering::train_encoded(idx_t nx, const uint8_t* x_in, const float* weights) {
    const uint8_t* x = x_in;
    std::unique_ptr<uint8_t[]> del1;
    std::unique_ptr<float[]> del3;
    size_t line_size = sizeof(float) * d;

    if (nx == k) {
        // this is a corner case, just copy training set to clusters
        if (verbose) {
            printf("Number of training points (%" PRId64
                   ") same as number of "
                   "clusters, just copying\n",
                   nx);
        }
        centroids.resize(d * k);
        memcpy(centroids.data(), x_in, sizeof(float) * d * k);
        // one fake iteration...
        ClusteringIterationStats stats = {0.0, 0.0, 0.0, 1.0, 0};
        iteration_stats.push_back(stats);
        return;
    }

    std::unique_ptr<idx_t[]> assign(new idx_t[nx]);
    std::unique_ptr<float[]> dis(new float[nx]);

    // remember best iteration for redo
    bool lower_is_better = true;
    float best_obj = lower_is_better ? HUGE_VALF : -HUGE_VALF;
    std::vector<ClusteringIterationStats> best_iteration_stats;
    std::vector<float> best_centroids;

    // support input centroids

    size_t n_input_centroids = centroids.size() / d;

    if (verbose && n_input_centroids > 0) {
        printf("  Using %zd centroids provided as input (%sfrozen)\n",
               n_input_centroids,
               frozen_centroids ? "" : "not ");
    }

    double t_search_tot = 0;

    // temporary buffer to decode vectors during the optimization
    std::vector<float> decode_buffer(0);
    hnswlib::DISTFUNC distfunc;
    if (spherical) {
        hnswlib::InnerProductSpace space(d);
        distfunc = space.get_dist_func();
    } else {
        hnswlib::L2Space space(d);
        distfunc = space.get_dist_func();
    }
    for (int redo = 0; redo < nredo; redo++) {
        if (verbose && nredo > 1) {
            printf("Outer iteration %d / %d\n", redo, nredo);
        }

        // initialize (remaining) centroids with random points from the dataset
        centroids.resize(d * k);
        std::vector<int> perm(nx);

        rand_perm(perm.data(), nx, seed + 1 + redo * 15486557L);

        for (int i = n_input_centroids; i < k; i++) {
            memcpy(&centroids[i * d], x + perm[i] * line_size, line_size);
        }

        post_process_centroids();

        // k-means iterations

        float obj = 0;
        for (int i = 0; i < niter; i++) {
            for (int j = 0; j < nx; ++j) {
                float distance = 10000;
                int id = 0;
                for (int l = 0; l < k; ++l) {
                    float tmp_distance = distfunc(centroids.data() + l * d, x + j * line_size, &d);
                    if (tmp_distance < distance) {
                        id = l;
                        distance = tmp_distance;
                    }
                }
                dis[j] = distance;
                assign[j] = id;
            }
            // accumulate objective
            obj = 0;
            for (int j = 0; j < nx; j++) {
                obj += dis[j];
            }

            // update the centroids
            std::vector<float> hassign(k);

            size_t k_frozen = frozen_centroids ? n_input_centroids : 0;
            compute_centroids(
                d, k, nx, k_frozen, x_in, assign.get(), weights, hassign.data(), centroids.data());

            int nsplit = split_clusters(d, k, nx, k_frozen, hassign.data(), centroids.data());
            // collect statistics
            ClusteringIterationStats stats = {
                obj, 0, t_search_tot / 1000, imbalance_factor(nx, k, assign.get()), nsplit};
            iteration_stats.push_back(stats);

            if (verbose) {
                printf(
                    "  Iteration %d (%.2f s, search %.2f s): "
                    "objective=%g imbalance=%.3f nsplit=%d       \r",
                    i,
                    stats.time,
                    stats.time_search,
                    stats.obj,
                    stats.imbalance_factor,
                    nsplit);
                fflush(stdout);
            }

            post_process_centroids();
        }
        if (nredo > 1) {
            if ((lower_is_better && obj < best_obj) || (!lower_is_better && obj > best_obj)) {
                if (verbose) {
                    printf("Objective improved: keep new clusters\n");
                }
                best_centroids = centroids;
                best_iteration_stats = iteration_stats;
                best_obj = obj;
            }
        }
    }
    if (nredo > 1) {
        centroids = best_centroids;
        iteration_stats = best_iteration_stats;
    }
}

float
kmeans_clustering(size_t d,
                  size_t n,
                  size_t k,
                  const float* x,
                  float* centroids,
                  const std::string& dis_type = "l2") {
    ClusteringParameters cp;
    if (dis_type == "ip") {
        cp.spherical = true;
    }
    Clustering clus(d, k, cp);
    clus.verbose = d * n * k > (1L << 30);
    // display logs if > 1Gflop per iteration
    clus.train(n, x);
    memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * k);
    return clus.iteration_stats.back().obj;
}

}  // namespace vsag

