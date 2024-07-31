
include(FetchContent)

FetchContent_Declare(
        thread_pool
        URL https://github.com/log4cplus/ThreadPool/archive/refs/heads/master.tar.gz
            https://vsagcache.oss-rg-china-mainland.aliyuncs.com/thread_pool/master.tar.gz
        URL_HASH MD5=99f810ce40388f6e142e62d99c9b076a)

FetchContent_MakeAvailable(thread_pool)
include_directories(${thread_pool_SOURCE_DIR}/)
