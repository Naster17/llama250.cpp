// WC read speed microbench: memcpy vs movntdqa from ggml_vk_host_malloc memory.
#include "ggml-vulkan.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <immintrin.h>
__attribute__((target("sse4.1")))
static void nt_copy(void * dst, const void * src, size_t n) {
    const uint8_t * s = (const uint8_t *) src;
    uint8_t * d = (uint8_t *) dst;
    const size_t head = (16 - ((uintptr_t) s & 15)) & 15;
    size_t i = head < n ? head : n;
    memcpy(d, s, i);
    for (; i + 64 <= n; i += 64) {
        __m128i a = _mm_stream_load_si128((__m128i *) (s + i));
        __m128i b = _mm_stream_load_si128((__m128i *) (s + i + 16));
        __m128i c = _mm_stream_load_si128((__m128i *) (s + i + 32));
        __m128i e = _mm_stream_load_si128((__m128i *) (s + i + 48));
        _mm_storeu_si128((__m128i *) (d + i), a);
        _mm_storeu_si128((__m128i *) (d + i + 16), b);
        _mm_storeu_si128((__m128i *) (d + i + 32), c);
        _mm_storeu_si128((__m128i *) (d + i + 48), e);
    }
    for (; i + 16 <= n; i += 16) {
        _mm_storeu_si128((__m128i *) (d + i), _mm_stream_load_si128((__m128i *) (s + i)));
    }
    if (i < n) memcpy(d + i, s + i, n - i);
}

int main() {
    ggml_backend_load_all();
    const size_t N = 4 << 20;
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_vk_host_buffer_type(), N);
    if (!buf) { fprintf(stderr, "host buffer alloc failed\n"); return 1; }
    void * wc = ggml_backend_buffer_get_base(buf);
    memset(wc, 1, N);
    std::vector<char> cached(N);
    for (int rep = 0; rep < 3; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        memcpy(cached.data(), wc, N);
        auto t1 = std::chrono::steady_clock::now();
        nt_copy(cached.data(), wc, N);
        auto t2 = std::chrono::steady_clock::now();
        double ms_mc = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_nt = std::chrono::duration<double, std::milli>(t2 - t1).count();
        printf("memcpy %.2f ms (%.0f MB/s)   nt_copy %.2f ms (%.0f MB/s)\n",
               ms_mc, N / ms_mc / 1000.0, ms_nt, N / ms_nt / 1000.0);
    }
    ggml_backend_buffer_free(buf);
    return 0;
}
