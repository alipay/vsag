#!/bin/bash

pids=()
exit_codes=()
logger_files=()
parallel_tags="[diskann] [hnsw] [hgraph]"
othertag=""

mkdir ./log

./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests --shard-count 1 --shard-index 0 > ./log/unittest.log &
pids+=($!)
logger_files+=("./log/unittest.log")

for tag in ${parallel_tags}
do
  othertag="~"${tag}${othertag}
  ./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${tag} > ./log/${tag}.log &
  pids+=($!)
  logger_files+=("./log/${tag}.log")
done

./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${othertag} > ./log/other.log &
pids+=($!)
logger_files+=("./log/other.log")

for pid in "${pids[@]}"
do
  wait $pid
  exit_codes+=($?)
done

index=0
all_successful=true
for code in "${exit_codes[@]}"
do
  if [ $code -ne 0 ]; then
    all_successful=false
    cat ${logger_files[${index}]}
  fi
  ((index+=1))
done

rm -rf ./log

if [ $all_successful = true ]; then
  exit 0
else
  exit 1
fi
