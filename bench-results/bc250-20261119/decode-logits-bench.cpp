// decode-logits-bench: per-token timing split of llama_decode vs logits extraction vs sampling.
// build: g++ -O2 -std=c++17 -I include -I ggml/include decode-logits-bench.cpp -o dlb \
//        -L build/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build/bin
#include "llama.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [n_tokens]\n", argv[0]); return 1; }
    int n = argc > 2 ? atoi(argv[2]) : 256;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 256; cp.n_ubatch = 256;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0;
    cp.n_threads = 10; cp.n_threads_batch = 10;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    const char * prompt = "Explain step by step how a flash attention kernel computes softmax online.";
    std::vector<llama_token> toks(512);
    int nt = llama_tokenize(vocab, prompt, strlen(prompt), toks.data(), toks.size(), true, false);
    toks.resize(nt);
    llama_batch b = llama_batch_get_one(toks.data(), nt);
    llama_decode(ctx, b);
    llama_token cur = 0;
    double t_dec = 0, t_log = 0, t_cpy = 0, t_smp = 0;
    for (int i = 0; i < n; i++) {
        auto a0 = std::chrono::steady_clock::now();
        llama_batch b1 = llama_batch_get_one(&cur, 1);
        llama_decode(ctx, b1);
        auto a1 = std::chrono::steady_clock::now();
        float * logits = llama_get_logits_ith(ctx, 0);
        auto a2 = std::chrono::steady_clock::now();
        static std::vector<float> local(n_vocab);
        memcpy(local.data(), logits, n_vocab * sizeof(float));
        auto a2b = std::chrono::steady_clock::now();
        cur = (llama_token)(std::max_element(local.begin(), local.end()) - local.begin());
        auto a3 = std::chrono::steady_clock::now();
        t_dec += std::chrono::duration<double, std::milli>(a1 - a0).count();
        t_log += std::chrono::duration<double, std::milli>(a2 - a1).count();
        t_cpy += std::chrono::duration<double, std::milli>(a2b - a2).count();
        t_smp += std::chrono::duration<double, std::milli>(a3 - a2b).count();
    }
    printf("tokens=%d  decode=%.2f ms  logits=%.2f ms  memcpy=%.2f ms  argmax=%.2f ms  total=%.2f ms  tg=%.1f t/s\n",
           n, t_dec/n, t_log/n, t_cpy/n, t_smp/n, (t_dec+t_log+t_cpy+t_smp)/n, n*1000.0/(t_dec+t_log+t_cpy+t_smp));
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
