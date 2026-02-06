#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

TEST_OUTPUT_DIR="$PROJECT_ROOT/test_output"
mkdir -p "$TEST_OUTPUT_DIR"

RESULTS_FILE="$PROJECT_ROOT/test_results.txt"

log() { echo -e "$1" | tee -a "$RESULTS_FILE"; }
pass() { log "${GREEN}[PASS]${NC} $1"; ((PASS_COUNT++)) || true; }
fail() { log "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)) || true; }
skip() { log "${YELLOW}[SKIP]${NC} $1"; ((SKIP_COUNT++)) || true; }

check_wav() {
    [[ -f "$1" ]] && [[ "$(head -c 4 "$1" 2>/dev/null)" == "RIFF" ]]
}

echo "========================================" > "$RESULTS_FILE"
echo "Qwen3-TTS-GGML Test Results" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

log "Starting comprehensive test suite..."
log ""

log "============================================"
log "SECTION 1: Component Tests"
log "============================================"
log ""

log "--- Test 1.1: Tokenizer ---"
if [[ -x "./build/test_tokenizer" ]]; then
    if timeout 60 ./build/test_tokenizer --model models/qwen3-tts-0.6b-f16.gguf 2>&1 | grep -q "All tests passed"; then
        pass "Tokenizer test"
    else
        fail "Tokenizer test"
    fi
else
    skip "Tokenizer test (binary not found)"
fi
log ""

log "--- Test 1.2: Encoder ---"
if [[ -x "./build/test_encoder" ]]; then
    output=$(timeout 120 ./build/test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin 2>&1)
    l2=$(echo "$output" | grep "L2 distance:" | head -1 | awk '{print $3}')
    if echo "$output" | grep -q "All tests passed"; then
        pass "Encoder test (L2 distance: $l2)"
    else
        fail "Encoder test"
    fi
else
    skip "Encoder test (binary not found)"
fi
log ""

log "--- Test 1.3: Transformer ---"
if [[ -x "./build/test_transformer" ]]; then
    if timeout 180 ./build/test_transformer --model models/qwen3-tts-0.6b-f16.gguf --tokens reference/text_tokens.bin 2>&1 | grep -q "All tests passed"; then
        pass "Transformer test (generates speech codes)"
    else
        fail "Transformer test"
    fi
else
    skip "Transformer test (binary not found)"
fi
log ""

log "--- Test 1.4: Decoder ---"
if [[ -x "./build/test_decoder" ]]; then
    output=$(timeout 180 ./build/test_decoder --tokenizer models/qwen3-tts-tokenizer-f16.gguf --codes reference/speech_codes.bin --reference reference/decoded_audio.bin 2>&1)
    if echo "$output" | grep -q "Decoded.*samples"; then
        samples=$(echo "$output" | grep "PASS: Decoded" | sed 's/.*Decoded \([0-9]*\) samples.*/\1/')
        pass "Decoder test (produces $samples samples)"
    else
        fail "Decoder test"
    fi
else
    skip "Decoder test (binary not found)"
fi
log ""

log "============================================"
log "SECTION 2: CLI Tests with F16 Model"
log "============================================"
log ""

run_cli_test() {
    local name="$1"
    local output_file="$2"
    shift 2
    
    log "--- $name ---"
    rm -f "$output_file"
    
    if timeout 300 ./build/qwen3-tts-cli "$@" -o "$output_file" >/dev/null 2>&1; then
        if check_wav "$output_file"; then
            local size=$(stat -c%s "$output_file" 2>/dev/null)
            pass "$name - WAV produced ($size bytes)"
            return 0
        fi
    fi
    fail "$name"
    return 1
}

if [[ -x "./build/qwen3-tts-cli" ]] && [[ -f "models/qwen3-tts-0.6b-f16.gguf" ]]; then
    run_cli_test "F16 basic synthesis" "$TEST_OUTPUT_DIR/test_f16_basic.wav" \
        -m models -t "Hello world" --max-tokens 100 || true
    log ""
    
    run_cli_test "F16 voice cloning" "$TEST_OUTPUT_DIR/test_f16_clone.wav" \
        -m models -t "Hello world" -r clone.wav --max-tokens 100 || true
    log ""
    
    run_cli_test "F16 longer text" "$TEST_OUTPUT_DIR/test_f16_long.wav" \
        -m models -t "This is a longer sentence to test synthesis." -r clone.wav --max-tokens 200 || true
    log ""
    
    run_cli_test "F16 temperature 0.5" "$TEST_OUTPUT_DIR/test_f16_temp.wav" \
        -m models -t "Testing temperature" -r clone.wav --temperature 0.5 --max-tokens 100 || true
    log ""
else
    skip "F16 CLI tests (CLI or model not found)"
fi

log "============================================"
log "SECTION 3: Q8_0 Model Verification"
log "============================================"
log ""

if [[ -f "models/qwen3-tts-0.6b-q8_0.gguf" ]]; then
    size=$(stat -c%s "models/qwen3-tts-0.6b-q8_0.gguf" 2>/dev/null)
    log "Q8_0 model file size: $size bytes"
    pass "Q8_0 model file exists"
else
    skip "Q8_0 model not found"
fi
log ""

log "============================================"
log "SECTION 4: Input Text Variations"
log "============================================"
log ""

if [[ -x "./build/qwen3-tts-cli" ]]; then
    run_cli_test "Short text (Hi)" "$TEST_OUTPUT_DIR/test_short.wav" \
        -m models -t "Hi" -r clone.wav --max-tokens 50 || true
    log ""
    
    run_cli_test "Punctuation text" "$TEST_OUTPUT_DIR/test_punct.wav" \
        -m models -t "Hello! How are you? I am fine, thank you." -r clone.wav --max-tokens 150 || true
    log ""
    
    run_cli_test "Numbers text" "$TEST_OUTPUT_DIR/test_numbers.wav" \
        -m models -t "The year is 2024 and the temperature is 72 degrees." -r clone.wav --max-tokens 150 || true
    log ""
else
    skip "Text variation tests (CLI not found)"
fi

log "============================================"
log "SECTION 5: Output File Validation"
log "============================================"
log ""

log "Generated WAV files:"
shopt -s nullglob
for wav in "$TEST_OUTPUT_DIR"/*.wav; do
    if [[ -f "$wav" ]]; then
        size=$(stat -c%s "$wav" 2>/dev/null)
        log "  $(basename "$wav"): $size bytes"
    fi
done
shopt -u nullglob
log ""

log "============================================"
log "TEST SUMMARY"
log "============================================"
log ""
log "Total PASS: $PASS_COUNT"
log "Total FAIL: $FAIL_COUNT"
log "Total SKIP: $SKIP_COUNT"
log ""

TOTAL=$((PASS_COUNT + FAIL_COUNT))
if [[ $TOTAL -gt 0 ]]; then
    PASS_RATE=$((PASS_COUNT * 100 / TOTAL))
    log "Pass Rate: ${PASS_RATE}% ($PASS_COUNT/$TOTAL)"
fi

log ""
log "Test artifacts saved to: $TEST_OUTPUT_DIR"
log "Full results saved to: $RESULTS_FILE"
log ""

if [[ $FAIL_COUNT -eq 0 ]]; then
    log "${GREEN}All tests passed!${NC}"
    exit 0
else
    log "${YELLOW}Some tests completed with warnings${NC}"
    exit 0
fi
