// fusion-bench: time q8_0 mul_mat GEMM shapes on Vulkan0 (separate vs fused QKV / gate+up)
// usage: fusion-bench m n k [reps]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s m n k [reps]\n", argv[0]); return 1; }
    const int64_t m = atoll(argv[1]);
    const int64_t n = atoll(argv[2]);
    const int64_t k = atoll(argv[3]);
    const int reps = argc > 4 ? atoi(argv[4]) : 30;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Vulkan0");
    if (!dev) { fprintf(stderr, "Vulkan0 not found\n"); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    struct ggml_init_params ip = {
        ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        nullptr,
        true
    };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }

    // zero-filled inputs: shader timing is data-independent
    size_t wbytes = ggml_nbytes(w), xbytes = ggml_nbytes(x);
    std::vector<char> zeros(wbytes > xbytes ? wbytes : xbytes, 0);
    ggml_backend_tensor_set(w, zeros.data(), 0, wbytes);
    ggml_backend_tensor_set(x, zeros.data(), 0, xbytes);

    for (int i = 0; i < 3; i++) { ggml_backend_graph_compute(backend, gf); }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) { ggml_backend_graph_compute(backend, gf); }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    double gflops = 2.0 * m * n * k / (us * 1e-6) / 1e9;
    printf("m=%-6lld n=%-4lld k=%-6lld  %9.1f us  %8.1f GFLOPS\n",
           (long long)m, (long long)n, (long long)k, us, gflops);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}