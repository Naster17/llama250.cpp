#!/bin/bash
# round 3: mimalloc A/B, RADV_PERFTEST=sam A/B, QKV/gate-up fusion GEMM microbench
OUT=$HOME/bench-20261115
M8=$HOME/models/LFM3-1.7B-Q8_0.gguf
NEW=$HOME/llama250.cpp/build/bin/llama-bench
COMMON="-ngl 99 -t 6 -r 3 -fa 1 -ub 512 -lm mmap -o csv"

run() { # name cmd...
    local name=$1; shift
    { echo "### $name"; "$@"; echo; } >> "$OUT/$name.log" 2>&1
}

echo "== mimalloc build"
MI=""
if [ ! -d /tmp/mimalloc-2.2.4 ]; then
    curl -sL -m 120 https://github.com/microsoft/mimalloc/archive/refs/tags/v2.2.4.tar.gz -o /tmp/mi.tgz && tar xzf /tmp/mi.tgz -C /tmp
fi
if [ -d /tmp/mimalloc-2.2.4 ]; then
    cmake -S /tmp/mimalloc-2.2.4 -B /tmp/mimalloc-2.2.4/build -DCMAKE_BUILD_TYPE=Release -DMI_BUILD_TESTS=OFF -DMI_BUILD_OBJECT=OFF -DMI_BUILD_STATIC=OFF > /tmp/mi-build.log 2>&1
    cmake --build /tmp/mimalloc-2.2.4/build -j6 >> /tmp/mi-build.log 2>&1
    MI=$(ls /tmp/mimalloc-2.2.4/build/libmimalloc.so* 2>/dev/null | head -1)
fi
echo "mimalloc lib: ${MI:-MISSING}"

if [ -n "$MI" ]; then
    echo "== mimalloc A/B pp512n128 (2 rounds)"
    for i in 1 2; do
        run "mi-base-$i" "$NEW" -m "$M8" $COMMON -p 512 -n 128
        run "mi-mimalloc-$i" env LD_PRELOAD="$MI" "$NEW" -m "$M8" $COMMON -p 512 -n 128
    done
    echo "== mimalloc A/B pp2048"
    run mi-base-p2048 "$NEW" -m "$M8" $COMMON -p 2048 -n 0
    run mi-mimalloc-p2048 env LD_PRELOAD="$MI" "$NEW" -m "$M8" $COMMON -p 2048 -n 0
fi

echo "== RADV_PERFTEST=sam A/B pp2048"
run sam-base "$NEW" -m "$M8" $COMMON -p 2048 -n 0
run sam-on env RADV_PERFTEST=sam "$NEW" -m "$M8" $COMMON -p 2048 -n 0

echo "== fusion GEMM microbench"
cd "$HOME/llama250.cpp" || exit 1
LIBDIR=$(dirname "$(find build -name 'libggml-base.so*' | head -1)")
echo "libdir: $LIBDIR"
g++ -O2 -std=c++17 -I ggml/include -I include /tmp/fusion-bench.cpp -o build/bin/fusion-bench -L "$LIBDIR" -lggml -lggml-base -Wl,-rpath,"$PWD/$LIBDIR" 2> "$OUT/fusion-compile.log"
if [ -x build/bin/fusion-bench ]; then
    {
        echo "# QKV: q=2048 k=512 v=512 (m), n=512 tokens, k=2048"
        ./build/bin/fusion-bench 2048 512 2048 30
        ./build/bin/fusion-bench 512  512 2048 30
        ./build/bin/fusion-bench 3072 512 2048 30
        echo "# gate+up: 2x m=7680 vs fused m=15360"
        ./build/bin/fusion-bench 7680  512 2048 30
        ./build/bin/fusion-bench 15360 512 2048 30
        echo "# ffn_down reference"
        ./build/bin/fusion-bench 2048 512 7680 30
    } >> "$OUT/fusion.log" 2>&1
else
    echo "fusion-bench compile failed" >> "$OUT/fusion.log"
fi
cat "$OUT/fusion.log"
echo ROUND3-DONE