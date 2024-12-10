
# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from setuptools import setup, find_packages, Extension
import os
import shutil
import platform


long_description="""VSAG is a vector indexing library used for similarity search. The indexing algorithm allows users to search through various sizes of vector sets, especially those that cannot fit in memory. The library also provides methods for generating parameters based on vector dimensions and data scale, allowing developers to use it without understanding the algorithm’s principles. VSAG is written in C++ and provides a Python wrapper package called pyvsag. Developed by the Vector Database Team at Ant Group."""


# DON'T REMOVE: to make the wheel's name contains python version
example_module = Extension('example', sources=['example.c'])

setup(
    name='pyvsag',
    version='0.0.10',
    description='vsag is a vector indexing library used for similarity search',
    long_description=long_description,
    url='https://github.com/antgroup/vsag',
    author='the vsag project',
    author_email='the.vsag.project@gmail.com',
    license='Apache-2.0',
    keywords='search nearest neighbors',
    install_requires=['packaging'],
    packages=find_packages(),
    pacakge_dir={'':'.'},
    package_data={
        '': ['*.so', '*.so.*'],
    },
    include_package_data=True,
    zip_safe=False,
    ext_modules=[example_module],
)
