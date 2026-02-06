#include "audio_tokenizer_encoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#define QWEN3_TTS_MAX_NODES 16384

namespace qwen3_tts {

// Mel filterbank computation using librosa slaney normalization
// This matches librosa.filters.mel with norm='slaney'
static void compute_mel_filterbank_slaney(float * filterbank, int n_mels, int n_fft, 
                                           int sample_rate, float f_min, float f_max) {
    // Slaney-style mel scale (used by librosa default)
    auto hz_to_mel_slaney = [](float hz) -> float {
        // Linear below 1000 Hz, logarithmic above
        const float f_sp = 200.0f / 3.0f;  // 66.67 Hz
        const float min_log_hz = 1000.0f;
        const float min_log_mel = (min_log_hz - 0.0f) / f_sp;  // 15
        const float logstep = logf(6.4f) / 27.0f;  // log(6400/1000) / 27
        
        if (hz < min_log_hz) {
            return (hz - 0.0f) / f_sp;
        } else {
            return min_log_mel + logf(hz / min_log_hz) / logstep;
        }
    };
    
    auto mel_to_hz_slaney = [](float mel) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = (min_log_hz - 0.0f) / f_sp;
        const float logstep = logf(6.4f) / 27.0f;
        
        if (mel < min_log_mel) {
            return 0.0f + f_sp * mel;
        } else {
            return min_log_hz * expf(logstep * (mel - min_log_mel));
        }
    };
    
    float mel_min = hz_to_mel_slaney(f_min);
    float mel_max = hz_to_mel_slaney(f_max);
    
    int n_fft_bins = n_fft / 2 + 1;
    
    // Compute mel center frequencies
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }
    
    // Convert to Hz and then to FFT bin indices
    std::vector<float> hz_points(n_mels + 2);
    std::vector<float> fft_freqs(n_fft_bins);
    
    for (int i = 0; i < n_mels + 2; ++i) {
        hz_points[i] = mel_to_hz_slaney(mel_points[i]);
    }
    
    for (int i = 0; i < n_fft_bins; ++i) {
        fft_freqs[i] = (float)i * sample_rate / n_fft;
    }
    
    memset(filterbank, 0, n_mels * n_fft_bins * sizeof(float));
    
    // Create triangular filters with slaney normalization
    for (int m = 0; m < n_mels; ++m) {
        float f_left = hz_points[m];
        float f_center = hz_points[m + 1];
        float f_right = hz_points[m + 2];
        
        // Slaney normalization: divide by bandwidth (area normalization)
        float enorm = 2.0f / (f_right - f_left);
        
        for (int k = 0; k < n_fft_bins; ++k) {
            float freq = fft_freqs[k];
            
            if (freq >= f_left && freq <= f_center) {
                if (f_center > f_left) {
                    filterbank[m * n_fft_bins + k] = enorm * (freq - f_left) / (f_center - f_left);
                }
            } else if (freq > f_center && freq <= f_right) {
                if (f_right > f_center) {
                    filterbank[m * n_fft_bins + k] = enorm * (f_right - freq) / (f_right - f_center);
                }
            }
        }
    }
}

static void compute_dft(const float * input, float * real, float * imag, int n) {
    for (int k = 0; k < n; ++k) {
        real[k] = 0.0f;
        imag[k] = 0.0f;
        for (int t = 0; t < n; ++t) {
            float angle = -2.0f * M_PI * k * t / n;
            real[k] += input[t] * cosf(angle);
            imag[k] += input[t] * sinf(angle);
        }
    }
}

// Periodic Hann window (matches torch.hann_window with periodic=True, which is default)
static void compute_hann_window(float * window, int n) {
    for (int i = 0; i < n; ++i) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / n));
    }
}

// Compute centered window for STFT (PyTorch centers win_length window in n_fft frame)
static void compute_centered_window(float * window, int n_fft, int win_length) {
    // Zero-initialize
    memset(window, 0, n_fft * sizeof(float));
    
    // Compute Hann window of win_length
    int offset = (n_fft - win_length) / 2;
    for (int i = 0; i < win_length; ++i) {
        window[offset + i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / win_length));
    }
}

AudioTokenizerEncoder::AudioTokenizerEncoder() = default;

AudioTokenizerEncoder::~AudioTokenizerEncoder() {
    free_speaker_encoder_model(model_);
    
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        ggml_backend_free(state_.backend);
        state_.backend = nullptr;
    }
}

bool AudioTokenizerEncoder::load_model(const std::string & model_path) {
    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }
    
    model_.config.sample_rate = loader.get_u32("qwen3-tts.speaker_encoder.sample_rate", 24000);
    model_.config.embedding_dim = loader.get_u32("qwen3-tts.speaker_encoder.embedding_length", 1024);
    
    int64_t n_tensors = loader.get_n_tensors();
    int spk_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "spk_enc.", 8) == 0) {
            spk_tensor_count++;
        }
    }
    
    if (spk_tensor_count == 0) {
        error_msg_ = "No speaker encoder tensors found in model";
        return false;
    }
    
    size_t ctx_size = ggml_tensor_overhead() * spk_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to initialize GGML context";
        return false;
    }
    
    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "spk_enc.", 8) != 0) {
            continue;
        }
        
        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }
        
        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta_tensor);
        ggml_set_name(tensor, name);
        
        model_.tensors[name] = tensor;
        
        std::string sname(name);
        
        if (sname == "spk_enc.conv0.weight") model_.conv0_w = tensor;
        else if (sname == "spk_enc.conv0.bias") model_.conv0_b = tensor;
        else if (sname == "spk_enc.mfa.weight") model_.mfa_w = tensor;
        else if (sname == "spk_enc.mfa.bias") model_.mfa_b = tensor;
        else if (sname == "spk_enc.asp.conv.weight") model_.asp_conv_w = tensor;
        else if (sname == "spk_enc.asp.conv.bias") model_.asp_conv_b = tensor;
        else if (sname == "spk_enc.asp.tdnn.weight") model_.asp_tdnn_w = tensor;
        else if (sname == "spk_enc.asp.tdnn.bias") model_.asp_tdnn_b = tensor;
        else if (sname == "spk_enc.fc.weight") model_.fc_w = tensor;
        else if (sname == "spk_enc.fc.bias") model_.fc_b = tensor;
        else {
            int blk_idx, res_idx;
            char suffix[64];
            
            if (sscanf(name, "spk_enc.blk.%d.tdnn1.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].tdnn1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].tdnn1_b = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.tdnn2.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].tdnn2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].tdnn2_b = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.res2net.%d.%s", &blk_idx, &res_idx, suffix) == 3) {
                if (blk_idx >= 1 && blk_idx <= 3 && res_idx >= 0 && res_idx < 7) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].res2net_w[res_idx] = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].res2net_b[res_idx] = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.se.conv1.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].se_conv1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].se_conv1_b = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.se.conv2.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].se_conv2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].se_conv2_b = tensor;
                }
            }
        }
    }
    
    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx, 
                                     model_.tensors, model_.buffer, error_msg_)) {
        return false;
    }
    
    // Debug: print conv0 weight info
    if (model_.conv0_w) {
        fprintf(stderr, "DEBUG: conv0_w shape=[%lld, %lld, %lld], type=%d\n",
                (long long)model_.conv0_w->ne[0], (long long)model_.conv0_w->ne[1], 
                (long long)model_.conv0_w->ne[2], model_.conv0_w->type);
        if (model_.conv0_w->type == GGML_TYPE_F16) {
            ggml_fp16_t * data = (ggml_fp16_t *)model_.conv0_w->data;
            fprintf(stderr, "DEBUG: conv0_w first 10 values: ");
            for (int i = 0; i < 10; i++) {
                fprintf(stderr, "%.6f ", ggml_fp16_to_fp32(data[i]));
            }
            fprintf(stderr, "\n");
        }
    }
    
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

bool AudioTokenizerEncoder::compute_mel_spectrogram(const float * samples, int32_t n_samples,
                                                     std::vector<float> & mel, int32_t & n_frames) {
    const auto & cfg = model_.config;
    
    // Match PyTorch STFT padding: (n_fft - hop_size) // 2 on each side with reflect
    int padding = (cfg.n_fft - cfg.hop_length) / 2;
    int padded_length = n_samples + 2 * padding;
    
    // Create padded signal with reflect padding
    std::vector<float> padded(padded_length);
    for (int i = 0; i < padded_length; ++i) {
        int src_idx;
        if (i < padding) {
            // Reflect left: padding-1, padding-2, ..., 0 -> samples[padding-i], samples[padding-1-i], ...
            src_idx = padding - i;
        } else if (i >= padding + n_samples) {
            // Reflect right
            src_idx = 2 * n_samples - (i - padding) - 2;
        } else {
            src_idx = i - padding;
        }
        // Clamp to valid range
        src_idx = std::max(0, std::min(n_samples - 1, src_idx));
        padded[i] = samples[src_idx];
    }
    
    // With center=False, frames start at 0 and step by hop_length
    n_frames = (padded_length - cfg.n_fft) / cfg.hop_length + 1;
    if (n_frames <= 0) {
        error_msg_ = "Audio too short for mel spectrogram";
        return false;
    }
    
    int n_fft_bins = cfg.n_fft / 2 + 1;
    
    std::vector<float> filterbank(cfg.n_mels * n_fft_bins);
    compute_mel_filterbank_slaney(filterbank.data(), cfg.n_mels, cfg.n_fft, 
                                   cfg.sample_rate, cfg.f_min, cfg.f_max);
    
    // PyTorch STFT with win_length < n_fft centers the window in the n_fft frame
    // This is critical for matching PyTorch's output
    std::vector<float> window(cfg.n_fft);
    compute_centered_window(window.data(), cfg.n_fft, cfg.win_length);
    
    // Output: [batch, n_mels, n_frames] but we store as [n_mels, n_frames] row-major
    // which means mel[m * n_frames + f] = value at mel bin m, frame f
    mel.resize(cfg.n_mels * n_frames);
    
    std::vector<float> frame(cfg.n_fft, 0.0f);
    std::vector<float> fft_real(cfg.n_fft);
    std::vector<float> fft_imag(cfg.n_fft);
    std::vector<float> magnitude(n_fft_bins);
    
    for (int32_t f = 0; f < n_frames; ++f) {
        int start = f * cfg.hop_length;
        
        // Apply centered window to n_fft samples
        for (int i = 0; i < cfg.n_fft; ++i) {
            frame[i] = padded[start + i] * window[i];
        }
        
        compute_dft(frame.data(), fft_real.data(), fft_imag.data(), cfg.n_fft);
        
        // Compute magnitude (not power) - matches torch.stft with return_complex=True then abs()
        // spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        for (int k = 0; k < n_fft_bins; ++k) {
            magnitude[k] = sqrtf(fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k] + 1e-9f);
        }
        
        // Apply mel filterbank and log compression
        // mel_spec = torch.matmul(mel_basis, spec)
        // mel_spec = dynamic_range_compression_torch(mel_spec)  # log(clamp(x, min=1e-5) * 1)
        for (int m = 0; m < cfg.n_mels; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < n_fft_bins; ++k) {
                sum += filterbank[m * n_fft_bins + k] * magnitude[k];
            }
            // dynamic_range_compression: log(clamp(x, min=1e-5))
            mel[m * n_frames + f] = logf(std::max(sum, 1e-5f));
        }
    }
    
    return true;
}

static struct ggml_tensor * apply_reflect_pad_1d(struct ggml_context * ctx,
                                                  struct ggml_tensor * x,
                                                  int pad) {
    if (pad == 0) {
        return x;
    }
    
    int64_t T = x->ne[0];
    int64_t C = x->ne[1];
    int64_t B = x->ne[2];
    
    struct ggml_tensor * left_slices[16];
    struct ggml_tensor * right_slices[16];
    
    for (int i = 0; i < pad && i < 16; ++i) {
        int left_src_idx = pad - i;
        left_slices[i] = ggml_view_3d(ctx, x, 1, C, B,
                                       x->nb[1], x->nb[2],
                                       left_src_idx * x->nb[0]);
        left_slices[i] = ggml_cont(ctx, left_slices[i]);
        
        int right_src_idx = T - 2 - i;
        right_slices[i] = ggml_view_3d(ctx, x, 1, C, B,
                                        x->nb[1], x->nb[2],
                                        right_src_idx * x->nb[0]);
        right_slices[i] = ggml_cont(ctx, right_slices[i]);
    }
    
    struct ggml_tensor * left_pad = left_slices[0];
    for (int i = 1; i < pad && i < 16; ++i) {
        left_pad = ggml_concat(ctx, left_pad, left_slices[i], 0);
    }
    
    struct ggml_tensor * right_pad = right_slices[0];
    for (int i = 1; i < pad && i < 16; ++i) {
        right_pad = ggml_concat(ctx, right_pad, right_slices[i], 0);
    }
    
    struct ggml_tensor * padded = ggml_concat(ctx, left_pad, x, 0);
    padded = ggml_concat(ctx, padded, right_pad, 0);
    
    return padded;
}

static struct ggml_tensor * apply_conv1d(struct ggml_context * ctx,
                                          struct ggml_tensor * w,
                                          struct ggml_tensor * b,
                                          struct ggml_tensor * x,
                                          int stride, int pad, int dilation,
                                          const char * debug_name = nullptr,
                                          bool use_reflect_pad = true) {
    struct ggml_tensor * input = x;
    int actual_pad = pad;
    
    if (use_reflect_pad && pad > 0) {
        input = apply_reflect_pad_1d(ctx, x, pad);
        actual_pad = 0;
    }
    
    struct ggml_tensor * y = ggml_conv_1d(ctx, w, input, stride, actual_pad, dilation);
    if (debug_name) {
        char name[64];
        snprintf(name, sizeof(name), "%s_conv", debug_name);
        ggml_set_name(y, name);
    }
    if (b) {
        int64_t oc = y->ne[1];
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, oc, 1));
    }
    return y;
}

struct ggml_cgraph * AudioTokenizerEncoder::build_graph(int32_t n_frames) {
    const auto & cfg = model_.config;
    const int hidden_dim = cfg.hidden_dim;  // 512
    const int scale = cfg.res2net_scale;    // 8
    const int branch_dim = hidden_dim / scale;  // 64
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    // Input: mel spectrogram [n_mels, n_frames] - stored as [n_mels, n_frames] row-major
    // GGML uses column-major, so this is [n_frames, n_mels] in GGML notation
    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, cfg.n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);
    ggml_set_output(mel);
    
    // PyTorch: hidden_states = hidden_states.transpose(1, 2)  # [B, T, C] -> [B, C, T]
    // Our mel is [n_frames, n_mels] in GGML = [n_mels, n_frames] row-major
    // For conv1d, we need [T, C, B] in GGML = [B, C, T] row-major
    // So reshape to [n_frames, n_mels, 1]
    struct ggml_tensor * cur = ggml_reshape_3d(ctx0, mel, n_frames, cfg.n_mels, 1);
    ggml_set_name(cur, "mel_3d");
    
    struct ggml_tensor * mel_padded = apply_reflect_pad_1d(ctx0, cur, 2);
    ggml_set_name(mel_padded, "mel_padded");
    ggml_set_output(mel_padded);
    
    fprintf(stderr, "DEBUG: conv0_w shape=[%lld, %lld, %lld]\n", 
            (long long)model_.conv0_w->ne[0], (long long)model_.conv0_w->ne[1], (long long)model_.conv0_w->ne[2]);
    fprintf(stderr, "DEBUG: mel_padded shape=[%lld, %lld, %lld]\n",
            (long long)mel_padded->ne[0], (long long)mel_padded->ne[1], (long long)mel_padded->ne[2]);
    
    cur = ggml_conv_1d(ctx0, model_.conv0_w, mel_padded, 1, 0, 1);
    ggml_set_name(cur, "conv0_conv");
    
    fprintf(stderr, "DEBUG: conv0_conv output shape=[%lld, %lld, %lld]\n",
            (long long)cur->ne[0], (long long)cur->ne[1], (long long)cur->ne[2]);
    
    if (model_.conv0_b) {
        int64_t oc = cur->ne[1];
        fprintf(stderr, "DEBUG: conv0_b n_elements=%lld, trying to reshape to [1, %lld, 1]\n",
                (long long)ggml_nelements(model_.conv0_b), (long long)oc);
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.conv0_b, 1, oc, 1));
    }
    ggml_set_name(cur, "conv0_pre_relu");
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "conv0_out");
    ggml_set_output(cur);
    
    int64_t seq_len = cur->ne[0];
    
    // Store block outputs for MFA (including block 0)
    struct ggml_tensor * block_outputs[4];
    block_outputs[0] = cur;  // Initial TDNN output
    
    // Blocks 1-3: SE-Res2Net blocks
    // Dilations: block1=2, block2=3, block3=4
    int dilations[3] = {2, 3, 4};
    
    for (int blk = 0; blk < 3; ++blk) {
        const auto & block = model_.blocks[blk];
        int dilation = dilations[blk];
        
        struct ggml_tensor * residual = cur;
        
        cur = apply_conv1d(ctx0, block.tdnn1_w, block.tdnn1_b, cur, 1, 0, 1);
        cur = ggml_relu(ctx0, cur);
        if (blk == 0) {
            ggml_set_name(cur, "blk1_tdnn1");
            ggml_set_output(cur);
        }
        
        // Res2Net: Split into 8 branches of 64 channels each
        // cur shape: [seq_len, 512, 1]
        // Branch 0: identity (no conv)
        // Branch i (1-7): conv(hidden_part + previous_output) for i >= 2, conv(hidden_part) for i == 1
        
        // Split channels: view as [seq_len, 64, 8] then split
        struct ggml_tensor * branches[8];
        
        // Extract each branch using view operations
        // cur is [seq_len, 512, 1], we want to split dim 1 into 8 parts of 64
        for (int b = 0; b < scale; ++b) {
            // View into the b-th chunk of 64 channels
            // cur shape: [seq_len, 512, 1], we want [seq_len, 64, 1] starting at channel b*64
            // nb1 = stride for dim 1 = cur->nb[1] (bytes to move from one channel to next)
            // nb2 = stride for dim 2 = cur->nb[2] (bytes to move from one batch to next)
            // offset = b * 64 * cur->nb[1] (skip b*64 channels)
            branches[b] = ggml_view_3d(ctx0, cur, 
                                        seq_len, branch_dim, 1,
                                        cur->nb[1], cur->nb[2], 
                                        b * branch_dim * cur->nb[1]);
            branches[b] = ggml_cont(ctx0, branches[b]);
        }
        
        // Process branches according to Res2Net logic
        struct ggml_tensor * outputs[8];
        outputs[0] = branches[0];  // Branch 0: identity
        
        for (int b = 1; b < scale; ++b) {
            struct ggml_tensor * input;
            if (b == 1) {
                input = branches[b];
            } else {
                // Add previous output to current branch
                input = ggml_add(ctx0, branches[b], outputs[b - 1]);
            }
            
            // Apply conv with dilation (kernel=3)
            // Padding for kernel=3, dilation=d: pad = d * (3-1) / 2 = d
            if (block.res2net_w[b - 1]) {
                outputs[b] = apply_conv1d(ctx0, block.res2net_w[b - 1], block.res2net_b[b - 1], 
                                          input, 1, dilation, dilation);
                outputs[b] = ggml_relu(ctx0, outputs[b]);
            } else {
                outputs[b] = input;  // Fallback if weight missing
            }
        }
        
        cur = outputs[0];
        for (int b = 1; b < scale; ++b) {
            cur = ggml_concat(ctx0, cur, outputs[b], 1);
        }
        if (blk == 0) {
            ggml_set_name(cur, "blk1_res2net");
            ggml_set_output(cur);
            for (int b = 0; b < scale; ++b) {
                char name[32];
                snprintf(name, sizeof(name), "blk1_branch%d", b);
                ggml_set_name(outputs[b], name);
                ggml_set_output(outputs[b]);
            }
        }
        
        cur = apply_conv1d(ctx0, block.tdnn2_w, block.tdnn2_b, cur, 1, 0, 1);
        cur = ggml_relu(ctx0, cur);
        if (blk == 0) {
            ggml_set_name(cur, "blk1_tdnn2");
            ggml_set_output(cur);
        }
        
        // SE (Squeeze-Excitation)
        // Global average pooling over time: mean(dim=2, keepdim=True)
        struct ggml_tensor * se = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
        se = ggml_reshape_3d(ctx0, se, 1, hidden_dim, 1);
        
        // SE conv1: 512 -> 128 with ReLU
        se = apply_conv1d(ctx0, block.se_conv1_w, block.se_conv1_b, se, 1, 0, 1);
        se = ggml_relu(ctx0, se);
        
        // SE conv2: 128 -> 512 with Sigmoid
        se = apply_conv1d(ctx0, block.se_conv2_w, block.se_conv2_b, se, 1, 0, 1);
        se = ggml_sigmoid(ctx0, se);
        
        cur = ggml_mul(ctx0, cur, se);
        if (blk == 0) {
            ggml_set_name(cur, "blk1_se");
            ggml_set_output(cur);
        }
        
        cur = ggml_add(ctx0, cur, residual);
        ggml_set_output(cur);
        
        char block_name[32];
        snprintf(block_name, sizeof(block_name), "block_%d", blk + 1);
        ggml_set_name(cur, block_name);
        
        block_outputs[blk + 1] = cur;
    }
    
    // MFA: Concatenate block outputs [1:] (blocks 1, 2, 3 = indices 1, 2, 3)
    // hidden_states = torch.cat(hidden_states_list[1:], dim=1)
    // Each block output is [seq_len, 512, 1]
    // Concatenated: [seq_len, 1536, 1]
    struct ggml_tensor * mfa_input = ggml_concat(ctx0, block_outputs[1], block_outputs[2], 1);
    mfa_input = ggml_concat(ctx0, mfa_input, block_outputs[3], 1);
    ggml_set_name(mfa_input, "mfa_input");
    ggml_set_output(mfa_input);
    
    // MFA conv: 1536 -> 1536 with ReLU
    cur = apply_conv1d(ctx0, model_.mfa_w, model_.mfa_b, mfa_input, 1, 0, 1);
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "mfa_out");
    ggml_set_output(cur);
    
    // ASP (Attentive Statistics Pooling)
    // cur shape: [seq_len, 1536, 1]
    
    // Step 1: Compute global mean and std over time
    // mean = hidden_states.mean(dim=2, keepdim=True)  # [1, 1536, 1]
    struct ggml_tensor * global_mean = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    global_mean = ggml_reshape_3d(ctx0, global_mean, 1, 1536, 1);
    
    // std = sqrt(E[x^2] - E[x]^2)
    struct ggml_tensor * sq = ggml_sqr(ctx0, cur);
    struct ggml_tensor * mean_sq = ggml_pool_1d(ctx0, sq, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    mean_sq = ggml_reshape_3d(ctx0, mean_sq, 1, 1536, 1);
    struct ggml_tensor * var = ggml_sub(ctx0, mean_sq, ggml_sqr(ctx0, global_mean));
    var = ggml_clamp(ctx0, var, 1e-12f, 1e10f);
    struct ggml_tensor * global_std = ggml_sqrt(ctx0, var);
    
    // Step 2: Expand mean and std to full sequence length and concatenate with hidden_states
    // mean = mean.repeat(1, 1, seq_length)  # [1, 1536, seq_len]
    // std = std.repeat(1, 1, seq_length)    # [1, 1536, seq_len]
    // attention = torch.cat([hidden_states, mean, std], dim=1)  # [1, 4608, seq_len]
    struct ggml_tensor * mean_expanded = ggml_repeat(ctx0, global_mean, 
                                                      ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    struct ggml_tensor * std_expanded = ggml_repeat(ctx0, global_std,
                                                     ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    
    struct ggml_tensor * attention = ggml_concat(ctx0, cur, mean_expanded, 1);
    attention = ggml_concat(ctx0, attention, std_expanded, 1);
    // attention shape: [seq_len, 4608, 1]
    
    // Step 3: TDNN (4608 -> 128) with ReLU, then Tanh
    // self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)  # has ReLU
    // attention = self.conv(self.tanh(self.tdnn(attention)))
    attention = apply_conv1d(ctx0, model_.asp_tdnn_w, model_.asp_tdnn_b, attention, 1, 0, 1);
    attention = ggml_relu(ctx0, attention);  // TDNN has ReLU
    ggml_set_name(attention, "asp_tdnn");
    ggml_set_output(attention);
    attention = ggml_tanh(ctx0, attention);  // Then tanh is applied
    
    // Step 4: Conv (128 -> 1536) for attention weights
    // self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1)
    attention = apply_conv1d(ctx0, model_.asp_conv_w, model_.asp_conv_b, attention, 1, 0, 1);
    ggml_set_name(attention, "asp_conv");
    ggml_set_output(attention);
    // attention shape: [seq_len, 1536, 1]
    
    // Step 5: Softmax over time dimension
    attention = ggml_soft_max(ctx0, attention);
    ggml_set_name(attention, "asp_softmax");
    ggml_set_output(attention);
    
    // Step 6: Compute weighted mean and std
    // mean, std = self._compute_statistics(hidden_states, attention)
    // mean = (attention * hidden_states).sum(dim=2)
    struct ggml_tensor * weighted = ggml_mul(ctx0, attention, cur);
    struct ggml_tensor * weighted_mean = ggml_pool_1d(ctx0, weighted, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    weighted_mean = ggml_scale(ctx0, weighted_mean, (float)seq_len);  // Convert avg to sum
    weighted_mean = ggml_reshape_3d(ctx0, weighted_mean, 1, 1536, 1);
    
    // std = sqrt((attention * (hidden_states - mean)^2).sum(dim=2).clamp(eps))
    struct ggml_tensor * mean_for_std = ggml_repeat(ctx0, weighted_mean,
                                                     ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    struct ggml_tensor * diff = ggml_sub(ctx0, cur, mean_for_std);
    struct ggml_tensor * diff_sq = ggml_sqr(ctx0, diff);
    struct ggml_tensor * weighted_var = ggml_mul(ctx0, attention, diff_sq);
    struct ggml_tensor * var_sum = ggml_pool_1d(ctx0, weighted_var, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    var_sum = ggml_scale(ctx0, var_sum, (float)seq_len);  // Convert avg to sum
    var_sum = ggml_reshape_3d(ctx0, var_sum, 1, 1536, 1);
    var_sum = ggml_clamp(ctx0, var_sum, 1e-12f, 1e10f);
    struct ggml_tensor * weighted_std = ggml_sqrt(ctx0, var_sum);
    
    // Step 7: Concatenate mean and std: [1, 3072, 1]
    struct ggml_tensor * pooled = ggml_concat(ctx0, weighted_mean, weighted_std, 1);
    ggml_set_name(pooled, "asp_pooled");
    ggml_set_output(pooled);
    
    // FC: 3072 -> 1024
    cur = apply_conv1d(ctx0, model_.fc_w, model_.fc_b, pooled, 1, 0, 1);
    ggml_set_name(cur, "fc_out");
    ggml_set_output(cur);
    
    // Squeeze to 1D
    cur = ggml_reshape_1d(ctx0, cur, cfg.embedding_dim);
    
    ggml_set_name(cur, "embedding");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool AudioTokenizerEncoder::encode(const float * samples, int32_t n_samples,
                                    std::vector<float> & embedding) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    ggml_backend_sched_free(state_.sched);
    std::vector<ggml_backend_t> backends = { state_.backend };
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, 1, QWEN3_TTS_MAX_NODES, false, true);
    
    std::vector<float> mel;
    int32_t n_frames;
    if (!compute_mel_spectrogram(samples, n_samples, mel, n_frames)) {
        return false;
    }
    
    struct weight_data {
        struct ggml_tensor * tensor;
        std::vector<uint8_t> data;
    };
    std::vector<weight_data> weights_to_copy;
    
    auto save_weight = [&weights_to_copy](struct ggml_tensor * t) {
        if (t && t->data) {
            size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> data(nbytes);
            ggml_backend_tensor_get(t, data.data(), 0, nbytes);
            weights_to_copy.push_back({t, std::move(data)});
        }
    };
    
    save_weight(model_.conv0_w);
    save_weight(model_.conv0_b);
    for (int blk = 0; blk < 3; ++blk) {
        save_weight(model_.blocks[blk].tdnn1_w);
        save_weight(model_.blocks[blk].tdnn1_b);
        for (int i = 0; i < 7; ++i) {
            save_weight(model_.blocks[blk].res2net_w[i]);
            save_weight(model_.blocks[blk].res2net_b[i]);
        }
        save_weight(model_.blocks[blk].tdnn2_w);
        save_weight(model_.blocks[blk].tdnn2_b);
        save_weight(model_.blocks[blk].se_conv1_w);
        save_weight(model_.blocks[blk].se_conv1_b);
        save_weight(model_.blocks[blk].se_conv2_w);
        save_weight(model_.blocks[blk].se_conv2_b);
    }
    save_weight(model_.mfa_w);
    save_weight(model_.mfa_b);
    save_weight(model_.asp_tdnn_w);
    save_weight(model_.asp_tdnn_b);
    save_weight(model_.asp_conv_w);
    save_weight(model_.asp_conv_b);
    save_weight(model_.fc_w);
    save_weight(model_.fc_b);
    
    fprintf(stderr, "DEBUG: Before build_graph - model_.conv0_w data=%p, buffer=%p\n",
            (void*)model_.conv0_w->data, (void*)model_.conv0_w->buffer);
    
    struct ggml_cgraph * gf = build_graph(n_frames);
    
    fprintf(stderr, "DEBUG: After build_graph - model_.conv0_w data=%p, buffer=%p\n",
            (void*)model_.conv0_w->data, (void*)model_.conv0_w->buffer);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    fprintf(stderr, "DEBUG: Graph has %d nodes\n", ggml_graph_n_nodes(gf));
    for (int i = 0; i < std::min(20, ggml_graph_n_nodes(gf)); i++) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        fprintf(stderr, "DEBUG: Node %d: %s, op=%d, shape=[%lld,%lld,%lld], data=%p\n",
                i, node->name, node->op, 
                (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2],
                (void*)node->data);
    }
    
    fprintf(stderr, "DEBUG: After alloc_graph - model_.conv0_w data=%p, buffer=%p\n",
            (void*)model_.conv0_w->data, (void*)model_.conv0_w->buffer);
    
    for (const auto & w : weights_to_copy) {
        ggml_backend_tensor_set(w.tensor, w.data.data(), 0, w.data.size());
    }
    
    int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW) {
            struct ggml_tensor * src = node->src[0];
            struct ggml_tensor * orig_src = src;
            while (src && (src->op == GGML_OP_RESHAPE || src->op == GGML_OP_VIEW)) {
                src = src->src[0];
            }
            for (const auto & w : weights_to_copy) {
                if (src == w.tensor) {
                    fprintf(stderr, "DEBUG: Found reshape/view of weight tensor %s, node shape=[%lld,%lld], copying %zu bytes\n",
                            w.tensor->name, (long long)node->ne[0], (long long)node->ne[1], w.data.size());
                    ggml_backend_tensor_set(node, w.data.data(), 0, w.data.size());
                    
                    if (strcmp(w.tensor->name, "spk_enc.conv0.weight") == 0) {
                        std::vector<uint8_t> verify_data(w.data.size());
                        ggml_backend_tensor_get(node, verify_data.data(), 0, w.data.size());
                        ggml_fp16_t * vd = (ggml_fp16_t *)verify_data.data();
                        fprintf(stderr, "DEBUG: After copy, reshape node first 5 values: ");
                        for (int j = 0; j < 5; ++j) {
                            fprintf(stderr, "%.6f ", ggml_fp16_to_fp32(vd[j]));
                        }
                        fprintf(stderr, "\n");
                        fprintf(stderr, "DEBUG: Original weight first 5 values: ");
                        const ggml_fp16_t * wd = (const ggml_fp16_t *)w.data.data();
                        for (int j = 0; j < 5; ++j) {
                            fprintf(stderr, "%.6f ", ggml_fp16_to_fp32(wd[j]));
                        }
                        fprintf(stderr, "\n");
                    }
                    break;
                }
            }
        }
    }
    
    fprintf(stderr, "DEBUG: After copy - model_.conv0_w data=%p\n", (void*)model_.conv0_w->data);
    
    if (model_.conv0_w->type == GGML_TYPE_F16) {
        std::vector<uint8_t> check_data(ggml_nbytes(model_.conv0_w));
        ggml_backend_tensor_get(model_.conv0_w, check_data.data(), 0, check_data.size());
        ggml_fp16_t * fp16_data = (ggml_fp16_t *)check_data.data();
        fprintf(stderr, "DEBUG: After copy conv0_w[0:5] = ");
        for (int i = 0; i < 5; ++i) {
            fprintf(stderr, "%.6f ", ggml_fp16_to_fp32(fp16_data[i]));
        }
        fprintf(stderr, "\n");
    }
    
    struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf, "mel");
    if (!mel_tensor) {
        error_msg_ = "Failed to find mel tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    // Try loading Python reference mel for debugging
    FILE * ref_mel_file = fopen("reference/debug/encoder_input.bin", "rb");
    if (ref_mel_file) {
        fseek(ref_mel_file, 0, SEEK_END);
        size_t file_size = ftell(ref_mel_file);
        fseek(ref_mel_file, 0, SEEK_SET);
        size_t ref_n_elements = file_size / sizeof(float);
        int n_mels = model_.config.n_mels;
        int ref_n_frames = ref_n_elements / n_mels;
        
        if (ref_n_frames == n_frames) {
            std::vector<float> ref_mel(ref_n_elements);
            fread(ref_mel.data(), sizeof(float), ref_n_elements, ref_mel_file);
            fclose(ref_mel_file);
            
            for (int c = 0; c < n_mels; c++) {
                for (int t = 0; t < n_frames; t++) {
                    mel[c * n_frames + t] = ref_mel[t * n_mels + c];
                }
            }
            fprintf(stderr, "DEBUG: Loaded Python reference mel (%d frames)\n", n_frames);
        } else {
            fclose(ref_mel_file);
            fprintf(stderr, "DEBUG: Reference mel has %d frames, expected %d - using computed mel\n", 
                    ref_n_frames, n_frames);
        }
    }
    
    // mel is stored as [n_mels, n_frames] row-major: mel[m * n_frames + f] = mel bin m at frame f
    // GGML tensor is [n_frames, n_mels] column-major: element (f, m) at memory[f + m * n_frames]
    // For GGML conv1d, we want input(t, c) = mel bin c at time t
    // So GGML memory[t + c * n_frames] should equal mel[c * n_frames + t]
    // Since the memory layout matches (both are contiguous in frame order for each mel bin),
    // we can copy directly!
    ggml_backend_tensor_set(mel_tensor, mel.data(), 0, mel.size() * sizeof(float));
    
    // Debug: verify the data layout
    fprintf(stderr, "DEBUG: mel_tensor shape=[%lld, %lld], checking values...\n",
            (long long)mel_tensor->ne[0], (long long)mel_tensor->ne[1]);
    // After setting, GGML element (t=0, c=0) should be mel[0 * n_frames + 0] = mel[0]
    // GGML element (t=0, c=1) should be mel[1 * n_frames + 0] = mel[n_frames]
    fprintf(stderr, "DEBUG: Expected GGML (t=0, c=0:5) = ");
    for (int c = 0; c < 5; ++c) {
        fprintf(stderr, "%.4f ", mel[c * n_frames + 0]);
    }
    fprintf(stderr, "\n");
    
    // Verify by reading back from tensor
    std::vector<float> mel_check(mel.size());
    ggml_backend_tensor_get(mel_tensor, mel_check.data(), 0, mel_check.size() * sizeof(float));
    fprintf(stderr, "DEBUG: Actual GGML (t=0, c=0:5) = ");
    for (int c = 0; c < 5; ++c) {
        // GGML element (t, c) is at memory[t + c * ne[0]]
        fprintf(stderr, "%.4f ", mel_check[0 + c * n_frames]);
    }
    fprintf(stderr, "\n");
    
    fprintf(stderr, "DEBUG: mel (t=0:3, c=0) = ");
    for (int t = 0; t < 3; ++t) {
        fprintf(stderr, "%.4f ", mel_check[t + 0 * n_frames]);
    }
    fprintf(stderr, "\n");
    
    fprintf(stderr, "DEBUG: mel (t=0, c=0:5) from memory = ");
    for (int c = 0; c < 5; ++c) {
        fprintf(stderr, "%.4f ", mel_check[0 + c * n_frames]);
    }
    fprintf(stderr, "\n");
    
    fprintf(stderr, "DEBUG: mel_tensor ne=[%lld, %lld], nb=[%zu, %zu]\n",
            (long long)mel_tensor->ne[0], (long long)mel_tensor->ne[1],
            mel_tensor->nb[0], mel_tensor->nb[1]);
    fprintf(stderr, "DEBUG: mel_tensor buffer=%p, data=%p\n",
            (void*)mel_tensor->buffer, (void*)mel_tensor->data);
    
    struct ggml_tensor * mel_3d_tensor = ggml_graph_get_tensor(gf, "mel_3d");
    if (mel_3d_tensor) {
        fprintf(stderr, "DEBUG: mel_3d ne=[%lld, %lld, %lld], nb=[%zu, %zu, %zu]\n",
                (long long)mel_3d_tensor->ne[0], (long long)mel_3d_tensor->ne[1], (long long)mel_3d_tensor->ne[2],
                mel_3d_tensor->nb[0], mel_3d_tensor->nb[1], mel_3d_tensor->nb[2]);
        fprintf(stderr, "DEBUG: mel_3d data=%p (same as mel? %s)\n",
                (void*)mel_3d_tensor->data, 
                (mel_3d_tensor->data == mel_tensor->data) ? "yes" : "no");
    }
    
    fprintf(stderr, "DEBUG: conv0_w shape=[%lld, %lld, %lld], type=%d\n",
            (long long)model_.conv0_w->ne[0], (long long)model_.conv0_w->ne[1], 
            (long long)model_.conv0_w->ne[2], model_.conv0_w->type);
    fprintf(stderr, "DEBUG: conv0_w nb=[%zu, %zu, %zu]\n",
            model_.conv0_w->nb[0], model_.conv0_w->nb[1], model_.conv0_w->nb[2]);
    if (model_.conv0_w->type == GGML_TYPE_F16) {
        ggml_fp16_t * w_data = (ggml_fp16_t *)model_.conv0_w->data;
        fprintf(stderr, "DEBUG: conv0_w[0,0,0:5] = ");
        for (int oc = 0; oc < 5; ++oc) {
            fprintf(stderr, "%.6f ", ggml_fp16_to_fp32(w_data[0 + 0 * 5 + oc * 5 * 128]));
        }
        fprintf(stderr, "\n");
    }
    
    struct ggml_tensor * conv0_conv_tensor = ggml_graph_get_tensor(gf, "conv0_conv");
    if (conv0_conv_tensor && conv0_conv_tensor->src[0]) {
        struct ggml_tensor * mul_mat_result = conv0_conv_tensor->src[0];
        fprintf(stderr, "DEBUG: conv0_conv->src[0] (mul_mat result) op=%d, shape=[%lld,%lld,%lld]\n",
                mul_mat_result->op, (long long)mul_mat_result->ne[0], 
                (long long)mul_mat_result->ne[1], (long long)mul_mat_result->ne[2]);
        
        if (mul_mat_result->src[1]) {
            struct ggml_tensor * weight_reshaped = mul_mat_result->src[1];
            fprintf(stderr, "DEBUG: mul_mat->src[1] (weight_reshaped) op=%d, shape=[%lld,%lld], type=%d\n",
                    weight_reshaped->op, (long long)weight_reshaped->ne[0], 
                    (long long)weight_reshaped->ne[1], weight_reshaped->type);
            fprintf(stderr, "DEBUG: weight_reshaped data=%p, model_.conv0_w data=%p\n",
                    (void*)weight_reshaped->data, (void*)model_.conv0_w->data);
            
            if (weight_reshaped->type == GGML_TYPE_F16) {
                std::vector<uint8_t> wr_data(ggml_nbytes(weight_reshaped));
                ggml_backend_tensor_get(weight_reshaped, wr_data.data(), 0, wr_data.size());
                ggml_fp16_t * wrd = (ggml_fp16_t *)wr_data.data();
                fprintf(stderr, "DEBUG: weight_reshaped[0:5] = ");
                for (int i = 0; i < 5; ++i) {
                    fprintf(stderr, "%.6f ", ggml_fp16_to_fp32(wrd[i]));
                }
                fprintf(stderr, "\n");
            }
            
            if (weight_reshaped->src[0]) {
                struct ggml_tensor * weight_orig = weight_reshaped->src[0];
                fprintf(stderr, "DEBUG: weight_reshaped->src[0] op=%d, shape=[%lld,%lld,%lld], same_as_conv0_w=%s\n",
                        weight_orig->op, (long long)weight_orig->ne[0], 
                        (long long)weight_orig->ne[1], (long long)weight_orig->ne[2],
                        (weight_orig == model_.conv0_w) ? "yes" : "no");
            }
        }
        
        if (mul_mat_result->src[0]) {
            struct ggml_tensor * im2col_reshaped = mul_mat_result->src[0];
            fprintf(stderr, "DEBUG: mul_mat->src[0] (im2col_reshaped) op=%d, shape=[%lld,%lld], type=%d\n",
                    im2col_reshaped->op, (long long)im2col_reshaped->ne[0], 
                    (long long)im2col_reshaped->ne[1], im2col_reshaped->type);
            
            if (im2col_reshaped->src[0]) {
                struct ggml_tensor * im2col = im2col_reshaped->src[0];
                fprintf(stderr, "DEBUG: im2col op=%d, shape=[%lld,%lld,%lld], type=%d, data=%p, buffer=%p\n",
                        im2col->op, (long long)im2col->ne[0], 
                        (long long)im2col->ne[1], (long long)im2col->ne[2], im2col->type,
                        (void*)im2col->data, (void*)im2col->buffer);
                fprintf(stderr, "DEBUG: im2col_reshaped data=%p, buffer=%p\n",
                        (void*)im2col_reshaped->data, (void*)im2col_reshaped->buffer);
                fprintf(stderr, "DEBUG: mul_mat_result data=%p, buffer=%p\n",
                        (void*)mul_mat_result->data, (void*)mul_mat_result->buffer);
                
                struct ggml_tensor * im2col_input = im2col->src[1];
                if (im2col_input) {
                    fprintf(stderr, "DEBUG: im2col->src[1] ne=[%lld,%lld,%lld], nb=[%zu,%zu,%zu]\n",
                            (long long)im2col_input->ne[0], (long long)im2col_input->ne[1], (long long)im2col_input->ne[2],
                            im2col_input->nb[0], im2col_input->nb[1], im2col_input->nb[2]);
                    
                    const int32_t * op_params = (const int32_t *)im2col->op_params;
                    fprintf(stderr, "DEBUG: im2col op_params: s0=%d, s1=%d, p0=%d, p1=%d, d0=%d, d1=%d, is_2D=%d\n",
                            op_params[0], op_params[1], op_params[2], op_params[3], 
                            op_params[4], op_params[5], op_params[6]);
                }
                
                std::vector<uint8_t> im2col_data(ggml_nbytes(im2col));
                ggml_backend_tensor_get(im2col, im2col_data.data(), 0, im2col_data.size());
                ggml_fp16_t * imd = (ggml_fp16_t *)im2col_data.data();
                fprintf(stderr, "DEBUG: im2col[0:10, 0, 0] = ");
                for (int j = 0; j < 10; ++j) {
                    fprintf(stderr, "%.4f ", ggml_fp16_to_fp32(imd[j]));
                }
                fprintf(stderr, "\n");
            }
        }
    }
    
    // Debug: check mel tensor values right before compute
    {
        std::vector<float> mel_before(mel.size());
        ggml_backend_tensor_get(mel_tensor, mel_before.data(), 0, mel_before.size() * sizeof(float));
        fprintf(stderr, "DEBUG: Right before compute, mel (t=0, c=0:5) = ");
        for (int c = 0; c < 5; ++c) {
            fprintf(stderr, "%.4f ", mel_before[c * n_frames + 0]);
        }
        fprintf(stderr, "\n");
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * conv0_conv_after = ggml_graph_get_tensor(gf, "conv0_conv");
    if (conv0_conv_after && conv0_conv_after->src[0] && conv0_conv_after->src[0]->src[0]) {
        struct ggml_tensor * im2col_reshaped = conv0_conv_after->src[0]->src[0];
        if (im2col_reshaped->src[0]) {
            struct ggml_tensor * im2col = im2col_reshaped->src[0];
            std::vector<uint8_t> im2col_data(ggml_nbytes(im2col));
            ggml_backend_tensor_get(im2col, im2col_data.data(), 0, im2col_data.size());
            ggml_fp16_t * imd = (ggml_fp16_t *)im2col_data.data();
            fprintf(stderr, "DEBUG: After compute, im2col[0:10, 0, 0] = ");
            for (int j = 0; j < 10; ++j) {
                fprintf(stderr, "%.4f ", ggml_fp16_to_fp32(imd[j]));
            }
            fprintf(stderr, "\n");
            
            fprintf(stderr, "DEBUG: im2col->src[0] (kernel) name=%s, data=%p\n", 
                    im2col->src[0]->name, (void*)im2col->src[0]->data);
            fprintf(stderr, "DEBUG: im2col->src[1] (input) name=%s, data=%p\n", 
                    im2col->src[1]->name, (void*)im2col->src[1]->data);
            fprintf(stderr, "DEBUG: mel_tensor data=%p\n", (void*)mel_tensor->data);
            
            struct ggml_tensor * im2col_src1 = im2col->src[1];
            std::vector<float> src1_data(im2col_src1->ne[0] * im2col_src1->ne[1]);
            ggml_backend_tensor_get(im2col_src1, src1_data.data(), 0, src1_data.size() * sizeof(float));
            fprintf(stderr, "DEBUG: im2col->src[1] actual data (t=0, c=0:5) = ");
            for (int c = 0; c < 5; ++c) {
                fprintf(stderr, "%.4f ", src1_data[0 + c * im2col_src1->ne[0]]);
            }
            fprintf(stderr, "\n");
            
            struct ggml_tensor * im2col_src0 = im2col->src[0];
            if (im2col_src0->type == GGML_TYPE_F16) {
                std::vector<uint8_t> src0_data(ggml_nbytes(im2col_src0));
                ggml_backend_tensor_get(im2col_src0, src0_data.data(), 0, src0_data.size());
                ggml_fp16_t * src0_fp16 = (ggml_fp16_t *)src0_data.data();
                fprintf(stderr, "DEBUG: im2col->src[0] (kernel) first 10 values = ");
                for (int i = 0; i < 10; ++i) {
                    fprintf(stderr, "%.6f ", ggml_fp16_to_fp32(src0_fp16[i]));
                }
                fprintf(stderr, "\n");
            }
        }
    }
    
    struct ggml_tensor * mel_3d_after = ggml_graph_get_tensor(gf, "mel_3d");
    if (mel_3d_after) {
        std::vector<float> mel_3d_data(mel_3d_after->ne[0] * mel_3d_after->ne[1]);
        ggml_backend_tensor_get(mel_3d_after, mel_3d_data.data(), 0, mel_3d_data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: After compute, mel_3d (t=0, c=0:5) = ");
        for (int c = 0; c < 5; ++c) {
            fprintf(stderr, "%.4f ", mel_3d_data[0 + c * mel_3d_after->ne[0]]);
        }
        fprintf(stderr, "\n");
    }
    
    if (conv0_conv_tensor) {
        std::vector<float> conv0_conv_data(conv0_conv_tensor->ne[0] * conv0_conv_tensor->ne[1]);
        ggml_backend_tensor_get(conv0_conv_tensor, conv0_conv_data.data(), 0, conv0_conv_data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: conv0_conv (before bias)[:10, 0] = ");
        for (int i = 0; i < 10 && i < (int)conv0_conv_tensor->ne[1]; ++i) {
            fprintf(stderr, "%.4f ", conv0_conv_data[0 + i * conv0_conv_tensor->ne[0]]);
        }
        fprintf(stderr, "\n");
    }
    
    // Debug: print conv0 output before ReLU
    struct ggml_tensor * conv0_pre_tensor = ggml_graph_get_tensor(gf, "conv0_pre_relu");
    if (conv0_pre_tensor) {
        std::vector<float> conv0_pre_data(conv0_pre_tensor->ne[0] * conv0_pre_tensor->ne[1]);
        ggml_backend_tensor_get(conv0_pre_tensor, conv0_pre_data.data(), 0, conv0_pre_data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: conv0_pre_relu (after bias)[:10, 0] = ");
        for (int i = 0; i < 10 && i < (int)conv0_pre_tensor->ne[1]; ++i) {
            fprintf(stderr, "%.4f ", conv0_pre_data[0 + i * conv0_pre_tensor->ne[0]]);
        }
        fprintf(stderr, "\n");
    }
    
    // Debug: print conv0 output after ReLU
    struct ggml_tensor * conv0_tensor = ggml_graph_get_tensor(gf, "conv0_out");
    if (conv0_tensor) {
        std::vector<float> conv0_data(conv0_tensor->ne[0] * conv0_tensor->ne[1]);
        ggml_backend_tensor_get(conv0_tensor, conv0_data.data(), 0, conv0_data.size() * sizeof(float));
        float c0_min = conv0_data[0], c0_max = conv0_data[0], c0_sum = 0;
        for (float v : conv0_data) {
            c0_min = std::min(c0_min, v);
            c0_max = std::max(c0_max, v);
            c0_sum += v;
        }
        fprintf(stderr, "DEBUG: conv0_out shape=[%lld, %lld], min=%.4f, max=%.4f, mean=%.4f\n",
                (long long)conv0_tensor->ne[0], (long long)conv0_tensor->ne[1],
                c0_min, c0_max, c0_sum / conv0_data.size());
        fprintf(stderr, "DEBUG: conv0_out[:10, 0] = ");
        for (int i = 0; i < 10 && i < (int)conv0_tensor->ne[1]; ++i) {
            // GGML is column-major: element at (t, c) is at index t + c * ne[0]
            fprintf(stderr, "%.4f ", conv0_data[0 + i * conv0_tensor->ne[0]]);
        }
        fprintf(stderr, "\n");
    }
    
    auto print_tensor_debug = [&gf](const char * name, const char * ref_file) {
        struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
        if (!t) {
            fprintf(stderr, "DEBUG: %s not found\n", name);
            return;
        }
        size_t n_elem = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
        std::vector<float> data(n_elem);
        ggml_backend_tensor_get(t, data.data(), 0, n_elem * sizeof(float));
        
        float min_v = data[0], max_v = data[0], sum_v = 0;
        for (float v : data) {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            sum_v += v;
        }
        fprintf(stderr, "DEBUG: %s shape=[%lld,%lld,%lld], min=%.4f, max=%.4f, mean=%.4f\n",
                name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                min_v, max_v, sum_v / n_elem);
        fprintf(stderr, "DEBUG: %s first 5: ", name);
        for (int i = 0; i < 5 && i < (int)n_elem; ++i) {
            fprintf(stderr, "%.6f ", data[i]);
        }
        fprintf(stderr, "\n");
        
        if (ref_file) {
            char path[256];
            snprintf(path, sizeof(path), "reference/debug/%s", ref_file);
            FILE * f = fopen(path, "rb");
            if (f) {
                fseek(f, 0, SEEK_END);
                size_t ref_size = ftell(f) / sizeof(float);
                fseek(f, 0, SEEK_SET);
                std::vector<float> ref(ref_size);
                fread(ref.data(), sizeof(float), ref_size, f);
                fclose(f);
                
                int64_t T = t->ne[0];
                int64_t C = t->ne[1];
                
                fprintf(stderr, "DEBUG: %s ref (t=0, c=0:5): ", name);
                for (int c = 0; c < 5 && c < C; ++c) {
                    fprintf(stderr, "%.6f ", ref[c * T + 0]);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "DEBUG: %s cpp (t=0, c=0:5): ", name);
                for (int c = 0; c < 5 && c < C; ++c) {
                    fprintf(stderr, "%.6f ", data[0 + c * T]);
                }
                fprintf(stderr, "\nDEBUG: %s T=%lld, C=%lld, indices: ", name, (long long)T, (long long)C);
                for (int c = 0; c < 5 && c < C; ++c) {
                    fprintf(stderr, "%lld ", (long long)(0 + c * T));
                }
                fprintf(stderr, "\n");
                
                float l2 = 0;
                for (int64_t tt = 0; tt < T; ++tt) {
                    for (int64_t cc = 0; cc < C; ++cc) {
                        float cpp_val = data[tt + cc * T];
                        float ref_val = ref[cc * T + tt];
                        float diff = cpp_val - ref_val;
                        l2 += diff * diff;
                    }
                }
                l2 = sqrtf(l2);
                fprintf(stderr, "DEBUG: %s L2 distance from ref: %.6f\n", name, l2);
            }
        }
    };
    
    struct ggml_tensor * mel_padded_t = ggml_graph_get_tensor(gf, "mel_padded");
    if (mel_padded_t) {
        std::vector<float> mp_data(mel_padded_t->ne[0] * mel_padded_t->ne[1]);
        ggml_backend_tensor_get(mel_padded_t, mp_data.data(), 0, mp_data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: mel_padded shape=[%lld, %lld]\n",
                (long long)mel_padded_t->ne[0], (long long)mel_padded_t->ne[1]);
        fprintf(stderr, "DEBUG: mel_padded (t=0:7, c=0): ");
        for (int t = 0; t < 7; ++t) {
            fprintf(stderr, "%.4f ", mp_data[t + 0 * mel_padded_t->ne[0]]);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "DEBUG: mel_padded (t=0, c=0:5): ");
        for (int c = 0; c < 5; ++c) {
            fprintf(stderr, "%.4f ", mp_data[0 + c * mel_padded_t->ne[0]]);
        }
        fprintf(stderr, "\n");
    }
    
    print_tensor_debug("conv0_out", "block_0.bin");
    print_tensor_debug("blk1_tdnn1", "block_1_tdnn1.bin");
    
    for (int b = 0; b < 8; ++b) {
        char name[32];
        snprintf(name, sizeof(name), "blk1_branch%d", b);
        struct ggml_tensor * branch_t = ggml_graph_get_tensor(gf, name);
        if (branch_t) {
            std::vector<float> branch_data(branch_t->ne[0] * branch_t->ne[1]);
            ggml_backend_tensor_get(branch_t, branch_data.data(), 0, branch_data.size() * sizeof(float));
            fprintf(stderr, "DEBUG: %s shape=[%lld,%lld], first 5 at t=0: ", name,
                    (long long)branch_t->ne[0], (long long)branch_t->ne[1]);
            for (int c = 0; c < 5 && c < branch_t->ne[1]; ++c) {
                fprintf(stderr, "%.4f ", branch_data[0 + c * branch_t->ne[0]]);
            }
            fprintf(stderr, "\n");
        }
    }
    
    print_tensor_debug("blk1_res2net", "block_1_res2net.bin");
    print_tensor_debug("blk1_tdnn2", "block_1_tdnn2.bin");
    print_tensor_debug("blk1_se", "block_1_se.bin");
    print_tensor_debug("block_1", "block_1.bin");
    print_tensor_debug("block_2", "block_2.bin");
    print_tensor_debug("block_3", "block_3.bin");
    print_tensor_debug("mfa_input", "mfa_input.bin");
    print_tensor_debug("mfa_out", "mfa.bin");
    print_tensor_debug("asp_tdnn", "asp_tdnn.bin");
    print_tensor_debug("asp_conv", "asp_conv.bin");
    print_tensor_debug("asp_softmax", nullptr);
    print_tensor_debug("asp_pooled", "asp.bin");
    print_tensor_debug("fc_out", "fc.bin");
    
    struct ggml_tensor * emb_tensor = ggml_graph_get_tensor(gf, "embedding");
    if (!emb_tensor) {
        error_msg_ = "Failed to find embedding tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    embedding.resize(model_.config.embedding_dim);
    ggml_backend_tensor_get(emb_tensor, embedding.data(), 0, embedding.size() * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

void free_speaker_encoder_model(speaker_encoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

} // namespace qwen3_tts
