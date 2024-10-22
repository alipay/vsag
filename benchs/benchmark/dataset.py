#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import oss2
import os
import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import datetime
from oss2.credentials import EnvironmentVariableCredentialsProvider
import ast
from tqdm import tqdm

OSS_ACCESS_KEY_ID = os.environ.get('OSS_ACCESS_KEY_ID')
OSS_ACCESS_KEY_SECRET = os.environ.get('OSS_ACCESS_KEY_SECRET')
OSS_ENDPOINT = os.environ.get('OSS_ENDPOINT')
OSS_BUCKET = os.environ.get('OSS_BUCKET')
OSS_SOURCE_DIR = os.environ.get('OSS_SOURCE_DIR')

_auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
_bucket = oss2.Bucket(_auth, OSS_ENDPOINT, OSS_BUCKET)
target_dir = '/tmp/dataset'
def download_and_open_dataset(dataset_name, logging=None):
    if None in [OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_ENDPOINT, OSS_BUCKET, OSS_SOURCE_DIR]:
        if logging is not None:
            logging.error("missing oss env")
        exit(-1)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    source_file = os.path.join(OSS_SOURCE_DIR, dataset_name)
    target_file = os.path.join(target_dir, dataset_name)

    if not os.path.exists(target_file):
        _bucket.get_object_to_file(source_file, target_file)
    return h5py.File(target_file, 'r')


def read_dataset(dataset_name, logging=None):
    with download_and_open_dataset(dataset_name, logging) as file:
        train = np.array(file["train"])
        test = np.array(file["test"])
        neighbors = np.array(file["neighbors"])
        distances = np.array(file["distances"])
    return train, test, neighbors, distances


def create_dataset(ids, base, query, topk, dataset_name, distance):
    if distance == "angular":
        metric = "cosine"
    elif distance == "euclidean":
        metric = "euclidean"
    print("data size:", len(ids), len(base))
    print("query size:", len(query))
    nbrs = NearestNeighbors(n_neighbors=topk, metric=metric, algorithm='brute').fit(base)
    batch_size = 50
    n_query = len(query)
    distances = []
    indices = []

    for i in tqdm(range(0, n_query, batch_size)):
        end = min(i + batch_size, n_query)
        batch_query = query[i:end]
        D_batch, I_batch = nbrs.kneighbors(batch_query)
        distances.append(D_batch)
        indices.append(I_batch)

    D = np.vstack(distances)
    I = np.vstack(indices)

    with h5py.File(os.path.join(target_dir, dataset_name), "w") as f:
        f.create_dataset("ids", data=ids)
        f.create_dataset("train", data=base)
        f.create_dataset("test", data=query)
        f.create_dataset("neighbors", data=I)
        f.create_dataset("distances", data=D)
        f.attrs["type"] = "dense"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = len(base[0])
        f.attrs["point_type"] = "float"

def generate_random_data():
    n_labels = 100
    n_bases = 100000
    n_querys = 10000
    dims = 256
    per_bases = n_bases // n_labels
    per_querys = n_querys // n_labels
    bases = np.random.rand(n_bases, dims)
    querys = np.random.rand(n_querys, dims)
    query_labels = np.random.rand(n_bases, dims)
    base_labels = np.random.randint(0, n_labels, size=n_bases)
    query_labels = np.random.randint(0, n_labels, size=n_querys)

    train = []
    test = []
    neighbors = []
    final_distances = []
    train_labels = []
    test_labels = []
    for t in range(n_labels):
        bases_mask = bases[base_labels == t]
        querys_mask = querys[query_labels == t]
        nbrs = NearestNeighbors(n_neighbors=10, metric="euclidean", algorithm='brute').fit(bases_mask)
        batch_size = 50
        n_query = len(querys_mask)
        distances = []
        indices = []

        for i in tqdm(range(0, n_query, batch_size)):
            end = min(i + batch_size, n_query)
            batch_query = querys_mask[i:end]
            D_batch, I_batch = nbrs.kneighbors(batch_query)
            distances.append(D_batch)
            indices.append(I_batch)

        D = np.vstack(distances)
        I = np.vstack(indices)
        train.extend(bases_mask)
        test.extend(querys_mask)
        final_distances.extend(D)
        neighbors.extend(I)
        train_labels.extend(len(bases_mask) * [t])
        test_labels.extend(len(querys_mask) * [t])
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    with h5py.File("random.hdf5", "w") as f:
        f.create_dataset("train", data=np.array(train))
        f.create_dataset("test", data=np.array(test))
        f.create_dataset("train_labels", data=np.array(train_labels, dtype=np.int64))
        f.create_dataset("test_labels", data=np.array(test_labels, dtype=np.int64))
        f.create_dataset("neighbors", data=np.array(neighbors))
        f.create_dataset("distances", data=np.array(final_distances))
        f.attrs["type"] = "dense"
        f.attrs["distance"] = "euclidean"
        f.attrs["dimension"] = dims
        f.attrs["point_type"] = "float"







def csv_to_data(filename, id_column, base_column, dim):
    parser = vector_parse_wrapper(dim)
    df = pd.read_csv(filename=10000)
    df[base_column] = df[base_column].apply(parser)
    df_cleaned = df.dropna(subset=[base_column])
    base = np.array(df_cleaned[base_column].tolist())
    ids = np.array(df_cleaned[id_column].tolist())
    ids_dtype = h5py.string_dtype(encoding='utf-8', length=max(len(s) for s in ids))
    ids_array = np.array(ids, dtype=ids_dtype)
    return ids_array, base



def csv_to_dataset(base_filename, base_size, query_size, id_column, base_column, dim, distance, dataset_name):
    data_ids, data = csv_to_data(base_filename, id_column, base_column, dim)
    unique_ids, index = np.unique(data_ids, return_index=True)
    unique_data = data[index]
    base_ids, base = unique_ids[:base_size], unique_data[:base_size]
    qeury_ids, query = unique_ids[-query_size:], unique_data[-query_size:]
    create_dataset(base_ids, base, query, 100, dataset_name, distance)


# You can customize the parsing function for the vector column and then run the main function.
fail_count = 0
def vector_parse_wrapper(dim):
    def split_vector(x):
        global fail_count
        try:
            data = np.array([float(i) for i in x.split(",")], dtype=np.float32)
            if data.shape[0] != dim:
                print(fail_count, x, dim)
                fail_count += 1
                return None
            return data
        except:
            print(fail_count, x)
            fail_count += 1
            return None
    return split_vector


"""
python script.py data/base.csv 1000000 --id_column=id --vector_column=base_vector --vector_size=2048 --output_file=test.hdf5
"""

if __name__ == '__main__':
    generate_random_data()





