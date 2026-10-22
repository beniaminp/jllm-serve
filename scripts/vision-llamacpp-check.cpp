// vision-llamacpp-check: llama.cpp (libmtmd) with a picture, to compare with `jllm run --image` (docs/20): the prompt's token
// ids per chunk, the first position's top-5 log-probabilities and a greedy answer of n tokens.
//
//   build:  clang++ -std=c++17 -O1 -I/opt/homebrew/include scripts/vision-llamacpp-check.cpp -o vision-llamacpp-check \
//             -L/opt/homebrew/opt/llama.cpp/lib -L/opt/homebrew/opt/ggml/lib -lllama -lmtmd -lggml -lggml-base \
//             -Wl,-rpath,/opt/homebrew/opt/llama.cpp/lib -Wl,-rpath,/opt/homebrew/opt/ggml/lib      (brew install llama.cpp)
//   usage:  GPU=1 vision-llamacpp-check text.gguf mmproj.gguf "<prompt with <__media__>>" picture|- n_tokens [n_gpu_layers]
//           GPU=1: the projector on the GPU; "-" instead of a picture: a text-only prompt (no <__media__> in it).
//   the jllm side:  JAVA_OPTS=-Djllm.vision.debug=true ./jllm run --model text.gguf --mmproj mmproj.gguf --image picture.png \
//                     --prompt "..." --temp 0 --max-tokens 32 --backend metal   (prints the prompt ids, the top-5, the token ids;
//                     --image - for a text-only prompt)
#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::string piece(const llama_vocab * v, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(v, t, buf, sizeof buf, 0, true);
    return n < 0 ? "?" : std::string(buf, n);
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s text.gguf mmproj.gguf prompt picture|- n_tokens [n_gpu_layers]\n", argv[0]);
        return 1;
    }
    const int nPredict = atoi(argv[5]);
    const int ngl = argc > 6 ? atoi(argv[6]) : 0;
    llama_backend_init();
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 512;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    llama_context * lctx = llama_init_from_model(model, cparams);
    if (!lctx) { fprintf(stderr, "cannot create context\n"); return 1; }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = getenv("GPU") && atoi(getenv("GPU"));
    cp.n_threads = 8;
    cp.print_timings = false;
    cp.warmup = false;
    mtmd_context * ctx = mtmd_init_from_file(argv[2], model, cp);
    if (!ctx) { fprintf(stderr, "cannot load mmproj\n"); return 1; }
    const bool noImage = std::string(argv[4]) == "-";
    mtmd_bitmap * b = noImage ? nullptr : mtmd_helper_bitmap_init_from_file(ctx, argv[4], false, mtmd_helper_init_opt_default()).bitmap;
    if (!b && !noImage) { fprintf(stderr, "cannot load image\n"); return 1; }
    const mtmd_bitmap * cb[1] = {b};
    mtmd_input_text text;
    std::string prompt = argv[3];
    text.text = prompt.c_str();
    text.text_len = prompt.size();
    text.add_special = true;
    text.parse_special = true;
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    int rc = mtmd_tokenize(ctx, chunks, &text, cb, noImage ? 0 : 1);
    if (rc != 0) { fprintf(stderr, "tokenize failed %d\n", rc); return 1; }
    size_t n = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n; i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t nt;
            const llama_token * t = mtmd_input_chunk_get_tokens_text(c, &nt);
            printf("chunk %zu text %zu:", i, nt);
            for (size_t j = 0; j < nt; j++) printf(" %d", t[j]);
            printf("\n");
        } else {
            printf("chunk %zu image n_tokens=%zu n_pos=%d\n", i, mtmd_image_tokens_get_n_tokens(mtmd_input_chunk_get_tokens_image(c)),
                   (int) mtmd_input_chunk_get_n_pos(c));
        }
    }
    llama_pos n_past = 0;
    auto t0 = std::chrono::steady_clock::now();
    rc = mtmd_helper_eval_chunks(ctx, lctx, chunks, 0, 0, 512, true, &n_past);
    auto t1 = std::chrono::steady_clock::now();
    if (rc != 0) { fprintf(stderr, "eval failed %d\n", rc); return 1; }
    printf("prompt evaluated in %.0f ms, n_past=%d\n", std::chrono::duration<double, std::milli>(t1 - t0).count(), (int) n_past);
    const int nv = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> out;
    for (int step = 0; step < nPredict; step++) {
        const float * lg = llama_get_logits_ith(lctx, -1);
        std::vector<int> idx(nv);
        for (int i = 0; i < nv; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int bb) { return lg[a] > lg[bb]; });
        double mx = lg[idx[0]], sum = 0;
        for (int i = 0; i < nv; i++) sum += std::exp((double) lg[i] - mx);
        double lse = mx + std::log(sum);
        if (step == 0) {
            printf("top5 at the first position:\n");
            for (int k = 0; k < 5; k++) printf("  %d %.5f '%s'\n", idx[k], lg[idx[k]] - lse, piece(vocab, idx[k]).c_str());
        }
        llama_token tok = idx[0];
        if (llama_vocab_is_eog(vocab, tok)) { printf("[eog %d]\n", tok); break; }
        out.push_back(tok);
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = tok;
        batch.pos[0] = n_past + step; // (explicit: after a picture the next position is not the KV cache's maximum + 1)
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        int drc = llama_decode(lctx, batch);
        llama_batch_free(batch);
        if (drc != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    printf("tokens:");
    for (llama_token t : out) printf(" %d", t);
    printf("\ntext:");
    for (llama_token t : out) printf("%s", piece(vocab, t).c_str());
    printf("\n");
    return 0;
}
