#include "tts_transformer.h"
#include "gguf_loader.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <algorithm>

namespace qwen3_tts {

TTSTransformer::TTSTransformer() = default;

TTSTransformer::~TTSTransformer() {
    free_tts_kv_cache(state_.cache);
    free_transformer_model(model_);
    
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        ggml_backend_free(state_.backend);
        state_.backend = nullptr;
    }
}

bool TTSTransformer::load_model(const std::string & model_path) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    
    struct gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }
    
    if (!parse_config(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!create_tensors(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!load_tensor_data(model_path, ctx)) {
        free_transformer_model(model_);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);
    
    state_.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state_.backend) {
        error_msg_ = "Failed to initialize CPU backend";
        return false;
    }
    
    std::vector<ggml_backend_t> backends = { state_.backend };
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, 1, QWEN3_TTS_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_MAX_NODES + ggml_graph_overhead());
    
    return true;
}

bool TTSTransformer::parse_config(struct gguf_context * ctx) {
    auto get_u32 = [&](const char * key, int32_t default_val) -> int32_t {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) return default_val;
        return (int32_t)gguf_get_val_u32(ctx, idx);
    };
    
    auto get_f32 = [&](const char * key, float default_val) -> float {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) return default_val;
        return gguf_get_val_f32(ctx, idx);
    };
    
    auto & cfg = model_.config;
    cfg.text_vocab_size = get_u32("qwen3-tts.text.vocab_size", 151936);
    cfg.text_embd_dim = get_u32("qwen3-tts.text.embedding_dim", 2048);
    cfg.hidden_size = get_u32("qwen3-tts.talker.embedding_length", 1024);
    cfg.n_layers = get_u32("qwen3-tts.talker.block_count", 28);
    cfg.n_attention_heads = get_u32("qwen3-tts.talker.attention.head_count", 16);
    cfg.n_key_value_heads = get_u32("qwen3-tts.talker.attention.head_count_kv", 8);
    cfg.intermediate_size = get_u32("qwen3-tts.talker.feed_forward_length", 3072);
    cfg.head_dim = get_u32("qwen3-tts.talker.attention.key_length", 128);
    cfg.rms_norm_eps = get_f32("qwen3-tts.talker.attention.layer_norm_rms_epsilon", 1e-6f);
    cfg.rope_theta = get_f32("qwen3-tts.talker.rope.freq_base", 1000000.0f);
    
    cfg.codec_vocab_size = get_u32("qwen3-tts.talker.codec_vocab_size", 3072);
    cfg.n_codebooks = get_u32("qwen3-tts.talker.num_codebooks", 16);
    
    cfg.code_pred_layers = get_u32("qwen3-tts.code_pred.layer_count", 5);
    cfg.code_pred_vocab_size = get_u32("qwen3-tts.code_pred.vocab_size", 2048);
    
    cfg.codec_pad_id = get_u32("qwen3-tts.codec.pad_id", 2148);
    cfg.codec_bos_id = get_u32("qwen3-tts.codec.bos_id", 2149);
    cfg.codec_eos_id = get_u32("qwen3-tts.codec.eos_id", 2150);
    
    return true;
}

bool TTSTransformer::create_tensors(struct gguf_context * ctx) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = model_.config;
    
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to create GGML context";
        return false;
    }
    
    model_.layers.resize(cfg.n_layers);
    model_.code_pred_layers.resize(cfg.code_pred_layers);
    model_.code_pred_embd.resize(cfg.n_codebooks - 1);
    model_.code_pred_head.resize(cfg.n_codebooks - 1);
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;
        
        if (strstr(name, "spk_enc.") || strstr(name, "tok_")) {
            continue;
        }
        
        if (strstr(name, "talker.text_embd.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_embd_dim;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            ne[0] = cfg.text_embd_dim;
            n_dims = 1;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.codec_head.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.output_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "talker.blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < cfg.n_layers) {
                
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_attention_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.n_attention_heads * cfg.head_dim;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.intermediate_size;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "code_pred.blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_attention_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.n_attention_heads * cfg.head_dim;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.intermediate_size;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            if (sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.lm_head.")) {
            int cb_idx = -1;
            if (sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
        } else {
            continue;
        }
        
        struct ggml_tensor * tensor = ggml_new_tensor(model_.ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg_ = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        model_.tensors[name] = tensor;
        
        if (strstr(name, "talker.text_embd.weight")) {
            model_.text_embd = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            model_.text_proj_fc1 = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            model_.text_proj_fc1_bias = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            model_.text_proj_fc2 = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            model_.text_proj_fc2_bias = tensor;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            model_.codec_embd = tensor;
        } else if (strstr(name, "talker.codec_head.weight")) {
            model_.codec_head = tensor;
        } else if (strstr(name, "talker.output_norm.weight")) {
            model_.output_norm = tensor;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            sscanf(name, "talker.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.n_layers) {
                auto & layer = model_.layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.blk.")) {
            int layer_idx = -1;
            sscanf(name, "code_pred.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                auto & layer = model_.code_pred_layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                model_.code_pred_embd[cb_idx] = tensor;
            }
        } else if (strstr(name, "code_pred.lm_head.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                model_.code_pred_head[cb_idx] = tensor;
            }
        }
    }
    
    return true;
}

bool TTSTransformer::load_tensor_data(const std::string & path, struct gguf_context * ctx) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        error_msg_ = "Failed to initialize CPU backend for loading";
        return false;
    }
    
    model_.buffer = ggml_backend_alloc_ctx_tensors(model_.ctx, backend);
    if (!model_.buffer) {
        error_msg_ = "Failed to allocate tensor buffer";
        ggml_backend_free(backend);
        return false;
    }
    
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg_ = "Failed to open file for reading: " + path;
        ggml_backend_free(backend);
        return false;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        
        auto it = model_.tensors.find(name);
        if (it == model_.tensors.end()) {
            continue;
        }
        
        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);
        
        read_buf.resize(nbytes);
        
        if (fseek(f, data_offset + offset, SEEK_SET) != 0) {
            error_msg_ = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        
        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg_ = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        
        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }
    
    fclose(f);
    ggml_backend_free(backend);
    
    return true;
}

bool TTSTransformer::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;
    
    free_tts_kv_cache(state_.cache);
    
    state_.cache.n_ctx = n_ctx;
    state_.cache.n_used = 0;
    state_.cache.head_dim = cfg.head_dim;
    state_.cache.n_kv_heads = cfg.n_key_value_heads;
    state_.cache.n_layers = cfg.n_layers;
    
    const size_t n_tensors = cfg.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    state_.cache.ctx = ggml_init(params);
    if (!state_.cache.ctx) {
        error_msg_ = "Failed to create KV cache context";
        return false;
    }
    
    state_.cache.k_cache.resize(cfg.n_layers);
    state_.cache.v_cache.resize(cfg.n_layers);
    
    for (int il = 0; il < cfg.n_layers; ++il) {
        state_.cache.k_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);
        
        state_.cache.v_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
    }
    
    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, state_.backend);
    if (!state_.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }
    
    return true;
}

void TTSTransformer::clear_kv_cache() {
    state_.cache.n_used = 0;
}

struct ggml_cgraph * TTSTransformer::build_text_graph(int32_t n_tokens, int32_t n_past,
                                                       bool has_speaker_embd) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_layers;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor * inp_speaker = nullptr;
    if (has_speaker_embd) {
        inp_speaker = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
        ggml_set_name(inp_speaker, "inp_speaker");
        ggml_set_input(inp_speaker);
    }
    
    struct ggml_tensor * cur = ggml_get_rows(ctx0, model_.text_embd, inp_tokens);
    
    cur = ggml_mul_mat(ctx0, model_.text_proj_fc1, cur);
    cur = ggml_add(ctx0, cur, model_.text_proj_fc1_bias);
    cur = ggml_gelu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, model_.text_proj_fc2, cur);
    cur = ggml_add(ctx0, cur, model_.text_proj_fc2_bias);
    
    if (has_speaker_embd) {
        struct ggml_tensor * speaker_expanded = ggml_repeat(ctx0, 
            ggml_reshape_2d(ctx0, inp_speaker, hidden_size, 1),
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_tokens));
        cur = ggml_add(ctx0, cur, speaker_expanded);
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        int n_kv = n_past + n_tokens;
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max(ctx0, KQ);
        
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "hidden_states");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * TTSTransformer::build_codec_graph(int32_t n_past) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_layers;
    const int n_tokens = 1;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    struct ggml_tensor * inp_token = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_token, "inp_codec_token");
    ggml_set_input(inp_token);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor * cur = ggml_get_rows(ctx0, model_.codec_embd, inp_token);
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        int n_kv = n_past + n_tokens;
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max(ctx0, KQ);
        
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "hidden_states");
    ggml_set_output(cur);
    
    struct ggml_tensor * logits = ggml_mul_mat(ctx0, model_.codec_head, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    
    ggml_build_forward_expand(gf, logits);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * TTSTransformer::build_code_pred_graph(int32_t n_prev_codes) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const int n_layer = cfg.code_pred_layers;
    const int n_codebooks = cfg.n_codebooks;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);
    
    struct ggml_tensor * inp_prev_codes = nullptr;
    if (n_prev_codes > 0) {
        inp_prev_codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_prev_codes);
        ggml_set_name(inp_prev_codes, "inp_prev_codes");
        ggml_set_input(inp_prev_codes);
    }
    
    struct ggml_tensor * cur = ggml_reshape_2d(ctx0, inp_hidden, hidden_size, 1);
    
    if (n_prev_codes > 0 && inp_prev_codes) {
        for (int cb = 0; cb < n_prev_codes && cb < n_codebooks - 1; ++cb) {
            struct ggml_tensor * code_idx = ggml_view_1d(ctx0, inp_prev_codes, 1, cb * sizeof(int32_t));
            struct ggml_tensor * code_embd = ggml_get_rows(ctx0, model_.code_pred_embd[cb], code_idx);
            cur = ggml_add(ctx0, cur, code_embd);
        }
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, 1);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, 1);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, 1);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        struct ggml_tensor * V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);
        
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_soft_max(ctx0, KQ);
        
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, 1);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    std::vector<struct ggml_tensor *> all_logits;
    for (int cb = 0; cb < n_codebooks - 1; ++cb) {
        struct ggml_tensor * cb_logits = ggml_mul_mat(ctx0, model_.code_pred_head[cb], cur);
        ggml_format_name(cb_logits, "logits_cb%d", cb + 1);
        ggml_set_output(cb_logits);
        all_logits.push_back(cb_logits);
    }
    
    for (auto * logits : all_logits) {
        ggml_build_forward_expand(gf, logits);
    }
    
    ggml_free(ctx0);
    
    return gf;
}

bool TTSTransformer::forward_text(const int32_t * text_tokens, int32_t n_tokens,
                                   const float * speaker_embd, int32_t n_past,
                                   std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        if (!init_kv_cache(4096)) {
            return false;
        }
    }
    
    if (n_past + n_tokens > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }
    
    bool has_speaker = (speaker_embd != nullptr);
    struct ggml_cgraph * gf = build_text_graph(n_tokens, n_past, has_speaker);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens) {
        ggml_backend_tensor_set(inp_tokens, text_tokens, 0, n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));
    }
    
    if (has_speaker) {
        struct ggml_tensor * inp_speaker = ggml_graph_get_tensor(gf, "inp_speaker");
        if (inp_speaker) {
            ggml_backend_tensor_set(inp_speaker, speaker_embd, 0, 
                                    model_.config.hidden_size * sizeof(float));
        }
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    output.resize(n_tokens * model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, output.data(), 0, output.size() * sizeof(float));
    
    last_hidden_.resize(model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, last_hidden_.data(), 
                           (n_tokens - 1) * model_.config.hidden_size * sizeof(float),
                           model_.config.hidden_size * sizeof(float));
    
    state_.cache.n_used = n_past + n_tokens;
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

bool TTSTransformer::forward_codec(int32_t codec_token, int32_t n_past,
                                    std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        if (!init_kv_cache(4096)) {
            return false;
        }
    }
    
    if (n_past + 1 > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }
    
    struct ggml_cgraph * gf = build_codec_graph(n_past);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    struct ggml_tensor * inp_token = ggml_graph_get_tensor(gf, "inp_codec_token");
    if (inp_token) {
        ggml_backend_tensor_set(inp_token, &codec_token, 0, sizeof(int32_t));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        int32_t pos = n_past;
        ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (hidden) {
        last_hidden_.resize(model_.config.hidden_size);
        ggml_backend_tensor_get(hidden, last_hidden_.data(), 0, 
                               model_.config.hidden_size * sizeof(float));
    }
    
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    output.resize(model_.config.codec_vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    
    state_.cache.n_used = n_past + 1;
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

bool TTSTransformer::get_hidden_states(std::vector<float> & hidden) const {
    if (last_hidden_.empty()) {
        return false;
    }
    hidden = last_hidden_;
    return true;
}

bool TTSTransformer::predict_codes(const float * hidden, const int32_t * prev_codes,
                                    std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;
    int n_prev = (prev_codes != nullptr) ? cfg.n_codebooks - 1 : 0;
    
    struct ggml_cgraph * gf = build_code_pred_graph(n_prev);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate code predictor graph";
        return false;
    }
    
    struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
    if (inp_hidden) {
        ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
    }
    
    if (n_prev > 0) {
        struct ggml_tensor * inp_prev = ggml_graph_get_tensor(gf, "inp_prev_codes");
        if (inp_prev) {
            ggml_backend_tensor_set(inp_prev, prev_codes, 0, n_prev * sizeof(int32_t));
        }
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute code predictor graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    output.resize((cfg.n_codebooks - 1) * cfg.code_pred_vocab_size);
    
    for (int cb = 0; cb < cfg.n_codebooks - 1; ++cb) {
        char name[32];
        snprintf(name, sizeof(name), "logits_cb%d", cb + 1);
        struct ggml_tensor * cb_logits = ggml_graph_get_tensor(gf, name);
        if (cb_logits) {
            ggml_backend_tensor_get(cb_logits, output.data() + cb * cfg.code_pred_vocab_size,
                                   0, cfg.code_pred_vocab_size * sizeof(float));
        }
    }
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

static int32_t argmax(const float * data, int32_t n) {
    int32_t max_idx = 0;
    float max_val = data[0];
    for (int32_t i = 1; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

bool TTSTransformer::generate(const int32_t * text_tokens, int32_t n_tokens,
                               const float * speaker_embd, int32_t max_len,
                               std::vector<int32_t> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;
    
    clear_kv_cache();
    
    std::vector<float> hidden_out;
    if (!forward_text(text_tokens, n_tokens, speaker_embd, 0, hidden_out)) {
        return false;
    }
    
    output.clear();
    output.reserve(max_len * cfg.n_codebooks);
    
    int32_t n_past = n_tokens;
    int32_t current_token = cfg.codec_bos_id;
    
    std::vector<float> logits;
    std::vector<float> code_pred_logits;
    std::vector<int32_t> frame_codes(cfg.n_codebooks);
    std::vector<int32_t> prev_codes(cfg.n_codebooks - 1);
    
    for (int frame = 0; frame < max_len; ++frame) {
        if (!forward_codec(current_token, n_past, logits)) {
            return false;
        }
        
        int32_t next_token = argmax(logits.data(), cfg.codec_vocab_size);
        
        if (next_token == cfg.codec_eos_id) {
            break;
        }
        
        if (next_token >= cfg.code_pred_vocab_size) {
            next_token = next_token % cfg.code_pred_vocab_size;
        }
        
        frame_codes[0] = next_token;
        
        std::vector<float> hidden;
        if (!get_hidden_states(hidden)) {
            error_msg_ = "Failed to get hidden states";
            return false;
        }
        
        if (frame > 0) {
            if (!predict_codes(hidden.data(), prev_codes.data(), code_pred_logits)) {
                return false;
            }
        } else {
            if (!predict_codes(hidden.data(), nullptr, code_pred_logits)) {
                return false;
            }
        }
        
        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            frame_codes[cb] = argmax(
                code_pred_logits.data() + (cb - 1) * cfg.code_pred_vocab_size,
                cfg.code_pred_vocab_size);
        }
        
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            output.push_back(frame_codes[cb]);
        }
        
        for (int cb = 0; cb < cfg.n_codebooks - 1; ++cb) {
            prev_codes[cb] = frame_codes[cb + 1];
        }
        
        current_token = next_token;
        n_past++;
    }
    
    return true;
}

bool TTSTransformer::forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                              std::vector<float> & output) {
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

bool TTSTransformer::forward_with_audio(const int32_t * tokens, int32_t n_tokens,
                                         const float * audio_embd, int32_t n_audio,
                                         int32_t audio_start_pos, int32_t n_past,
                                         std::vector<float> & output) {
    (void)audio_embd;
    (void)n_audio;
    (void)audio_start_pos;
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

void free_transformer_model(tts_transformer_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
    model.layers.clear();
    model.code_pred_layers.clear();
    model.code_pred_embd.clear();
    model.code_pred_head.clear();
}

void free_tts_kv_cache(tts_kv_cache & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.n_ctx = 0;
    cache.n_used = 0;
}

} // namespace qwen3_tts
