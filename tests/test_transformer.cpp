#include "tts_transformer.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>

static bool load_binary_file(const std::string & path, std::vector<uint8_t> & data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return false;
    }
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    data.resize(size);
    f.read(reinterpret_cast<char *>(data.data()), size);
    return f.good();
}

template<typename T>
static bool load_binary_array(const std::string & path, std::vector<T> & arr) {
    std::vector<uint8_t> data;
    if (!load_binary_file(path, data)) {
        return false;
    }
    arr.resize(data.size() / sizeof(T));
    memcpy(arr.data(), data.data(), data.size());
    return true;
}

static void print_usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model <path>   Path to TTS GGUF model (default: models/qwen3-tts-0.6b-f16.gguf)\n");
    printf("  --tokens <path>  Path to text tokens binary (default: reference/text_tokens.bin)\n");
    printf("  --speaker <path> Path to speaker embedding binary (default: reference/ref_audio_embedding.bin)\n");
    printf("  --ref-codes <path> Path to reference speech codes (default: reference/speech_codes.bin)\n");
    printf("  --max-len <n>    Maximum generation length (default: 100)\n");
    printf("  --help           Show this help\n");
}

int main(int argc, char ** argv) {
    std::string model_path = "models/qwen3-tts-0.6b-f16.gguf";
    std::string tokens_path = "reference/text_tokens.bin";
    std::string speaker_path = "reference/ref_audio_embedding.bin";
    std::string ref_codes_path = "reference/speech_codes.bin";
    int max_len = 100;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            tokens_path = argv[++i];
        } else if (strcmp(argv[i], "--speaker") == 0 && i + 1 < argc) {
            speaker_path = argv[++i];
        } else if (strcmp(argv[i], "--ref-codes") == 0 && i + 1 < argc) {
            ref_codes_path = argv[++i];
        } else if (strcmp(argv[i], "--max-len") == 0 && i + 1 < argc) {
            max_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    printf("=== TTS Transformer Test ===\n\n");
    
    printf("Test 1: Load model\n");
    qwen3_tts::TTSTransformer transformer;
    
    if (!transformer.load_model(model_path)) {
        printf("  FAIL: %s\n", transformer.get_error().c_str());
        return 1;
    }
    printf("  PASS: Model loaded successfully\n");
    
    auto config = transformer.get_config();
    printf("  Config: hidden_size=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d\n",
           config.hidden_size, config.n_layers, config.n_attention_heads, config.n_key_value_heads);
    printf("  Codec: vocab_size=%d, n_codebooks=%d\n", config.codec_vocab_size, config.n_codebooks);
    printf("  Code predictor: layers=%d, vocab_size=%d\n\n", config.code_pred_layers, config.code_pred_vocab_size);
    
    printf("Test 2: Initialize KV cache\n");
    if (!transformer.init_kv_cache(4096)) {
        printf("  FAIL: %s\n", transformer.get_error().c_str());
        return 1;
    }
    printf("  PASS: KV cache initialized\n\n");
    
    printf("Test 3: Load input data\n");
    std::vector<int64_t> text_tokens_i64;
    if (!load_binary_array(tokens_path, text_tokens_i64)) {
        printf("  FAIL: Could not load text tokens from %s\n", tokens_path.c_str());
        return 1;
    }
    
    std::vector<int32_t> text_tokens(text_tokens_i64.begin(), text_tokens_i64.end());
    printf("  Loaded %zu text tokens\n", text_tokens.size());
    printf("  Tokens: ");
    for (size_t i = 0; i < std::min(text_tokens.size(), (size_t)10); ++i) {
        printf("%d ", text_tokens[i]);
    }
    printf("\n");
    
    std::vector<float> speaker_embd;
    if (!load_binary_array(speaker_path, speaker_embd)) {
        printf("  WARNING: Could not load speaker embedding from %s, using nullptr\n", speaker_path.c_str());
    } else {
        printf("  Loaded speaker embedding: %zu floats\n", speaker_embd.size());
    }
    
    std::vector<int64_t> ref_codes_i64;
    if (!load_binary_array(ref_codes_path, ref_codes_i64)) {
        printf("  WARNING: Could not load reference codes from %s\n", ref_codes_path.c_str());
    } else {
        printf("  Loaded reference codes: %zu values\n", ref_codes_i64.size());
    }
    printf("  PASS: Input data loaded\n\n");
    
    printf("Test 4: Forward pass (text prefill)\n");
    std::vector<float> hidden_out;
    const float * spk_ptr = speaker_embd.empty() ? nullptr : speaker_embd.data();
    
    if (!transformer.forward_text(text_tokens.data(), (int32_t)text_tokens.size(), spk_ptr, 0, hidden_out)) {
        printf("  FAIL: %s\n", transformer.get_error().c_str());
        return 1;
    }
    printf("  PASS: Forward pass completed\n");
    printf("  Hidden states shape: [%zu, %d]\n", text_tokens.size(), config.hidden_size);
    
    float hidden_sum = 0.0f;
    for (float v : hidden_out) {
        hidden_sum += v;
    }
    printf("  Hidden states sum: %.6f\n\n", hidden_sum);
    
    printf("Test 5: Generate speech codes\n");
    transformer.clear_kv_cache();
    
    std::vector<int32_t> generated_codes;
    if (!transformer.generate(text_tokens.data(), (int32_t)text_tokens.size(), spk_ptr, max_len, generated_codes)) {
        printf("  FAIL: %s\n", transformer.get_error().c_str());
        return 1;
    }
    
    int n_frames = (int)generated_codes.size() / config.n_codebooks;
    printf("  PASS: Generated %d frames (%zu codes total)\n", n_frames, generated_codes.size());
    
    printf("  First 3 frames:\n");
    for (int f = 0; f < std::min(3, n_frames); ++f) {
        printf("    Frame %d: ", f);
        for (int cb = 0; cb < config.n_codebooks; ++cb) {
            printf("%d ", generated_codes[f * config.n_codebooks + cb]);
        }
        printf("\n");
    }
    
    if (!ref_codes_i64.empty()) {
        printf("\n  Reference first 3 frames:\n");
        int ref_frames = (int)ref_codes_i64.size() / config.n_codebooks;
        for (int f = 0; f < std::min(3, ref_frames); ++f) {
            printf("    Frame %d: ", f);
            for (int cb = 0; cb < config.n_codebooks; ++cb) {
                printf("%lld ", (long long)ref_codes_i64[f * config.n_codebooks + cb]);
            }
            printf("\n");
        }
    }
    printf("\n");
    
    printf("Test 6: Verify output\n");
    if (n_frames > 0) {
        printf("  PASS: Speech codes generated successfully\n");
        printf("  Output shape: [%d, %d]\n", n_frames, config.n_codebooks);
    } else {
        printf("  FAIL: No speech codes generated\n");
        return 1;
    }
    
    printf("\n=== All tests passed! ===\n");
    return 0;
}
