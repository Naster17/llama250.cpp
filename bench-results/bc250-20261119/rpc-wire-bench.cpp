// RPC cross-server copy latency/bandwidth microbench (row-split feasibility).
// Measures the exact primitive a cross-server allreduce would pay per layer:
// ggml_backend_tensor_copy between two RPC endpoints (host-bridged GET+SET).
// build: g++ -O2 -std=c++17 -I ggml/include rpc-wire-bench.cpp -o rpc-wire-bench \
//        -L build/bin -lggml-rpc -lggml-base -lggml -Wl,-rpath,$PWD/build/bin
// run:   ./rpc-wire-bench <endpoint1> <endpoint2>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s endpoint1 endpoint2\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    fprintf(stderr, "stage: init b1\n");
    ggml_backend_t b1 = ggml_backend_rpc_init(argv[1], 0);
    fprintf(stderr, "stage: init b2\n");
    ggml_backend_t b2 = ggml_backend_rpc_init(argv[2], 0);
    if (!b1 || !b2) { fprintf(stderr, "rpc init failed\n"); return 1; }
    fprintf(stderr, "stage: backends up\n");
    ggml_backend_buffer_type_t buft1 = ggml_backend_get_default_buffer_type(b1);
    ggml_backend_buffer_type_t buft2 = ggml_backend_get_default_buffer_type(b2);

    printf("%10s %12s %12s %12s\n", "bytes", "us/copy", "MB/s", "iters");
    for (size_t nbytes : {4096, 8192, 16384, 65536, 262144, 1048576, 4194304}) {
        size_t n = nbytes / 4;
        ggml_backend_buffer_t buf1 = ggml_backend_buft_alloc_buffer(buft1, nbytes);
        ggml_backend_buffer_t buf2 = ggml_backend_buft_alloc_buffer(buft2, nbytes);
        struct ggml_init_params ip = { .mem_size = ggml_tensor_overhead() * 2, .mem_buffer = nullptr, .no_alloc = true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_tensor * t2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        t1->buffer = buf1; t1->data = ggml_backend_buffer_get_base(buf1);
        t2->buffer = buf2; t2->data = ggml_backend_buffer_get_base(buf2);
        std::vector<float> host(n, 1.0f);
        ggml_backend_tensor_set(t1, host.data(), 0, nbytes);
        ggml_backend_synchronize(b1);
        fprintf(stderr, "stage: size %zu allocated, first copy\n", nbytes);

        int iters = nbytes <= 65536 ? 300 : (nbytes <= 1048576 ? 50 : 10);
        fprintf(stderr, "stage: size %zu warmup done, timing %d iters\n", nbytes, iters);
        for (int i = 0; i < 5; i++) { ggml_backend_tensor_copy(t1, t2); }
        ggml_backend_synchronize(b2);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) { ggml_backend_tensor_copy(t1, t2); }
        ggml_backend_synchronize(b2);
        auto t1me = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1me - t0).count() / iters;
        printf("%10zu %12.1f %12.1f %12d\n", nbytes, us, nbytes / us, iters);

        ggml_free(ctx);
        ggml_backend_buffer_free(buf1);
        ggml_backend_buffer_free(buf2);
    }
    ggml_backend_free(b1);
    ggml_backend_free(b2);
    return 0;
}
