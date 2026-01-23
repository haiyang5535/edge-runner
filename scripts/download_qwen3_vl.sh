#!/bin/bash
# ============================================================
# Qwen3-VL-2B Model Download Script
# For EdgeRunner VLM Migration
# ============================================================

set -e

CACHE_DIR="$HOME/.cache/qwen3-vl"
HF_BASE="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main"

echo "============================================================"
echo "🔄 Qwen3-VL-2B Model Download"
echo "============================================================"
echo ""

# Create cache directory
mkdir -p "$CACHE_DIR"
echo "📁 Download directory: $CACHE_DIR"
echo ""

# Download LLM (Q4_K_M)
LLM_FILE="$CACHE_DIR/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
if [ -f "$LLM_FILE" ]; then
    echo "✅ LLM model already exists: $LLM_FILE"
else
    echo "📥 Downloading LLM model (Q4_K_M, ~1.11 GB)..."
    wget --progress=bar:force -O "$LLM_FILE" \
        "$HF_BASE/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    echo "✅ LLM model downloaded"
fi
echo ""

# Download Vision Encoder (F16 mmproj)
MMPROJ_FILE="$CACHE_DIR/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
if [ -f "$MMPROJ_FILE" ]; then
    echo "✅ Vision encoder already exists: $MMPROJ_FILE"
else
    echo "📥 Downloading Vision encoder (F16, ~819 MB)..."
    wget --progress=bar:force -O "$MMPROJ_FILE" \
        "$HF_BASE/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
    echo "✅ Vision encoder downloaded"
fi
echo ""

# Verify downloads
echo "============================================================"
echo "📊 Verification"
echo "============================================================"
echo ""
ls -lh "$CACHE_DIR"
echo ""

# Check file sizes
LLM_SIZE=$(stat -c%s "$LLM_FILE" 2>/dev/null || echo "0")
MMPROJ_SIZE=$(stat -c%s "$MMPROJ_FILE" 2>/dev/null || echo "0")

if [ "$LLM_SIZE" -gt 1000000000 ]; then
    echo "✅ LLM file size OK (~1.1 GB)"
else
    echo "❌ LLM file may be incomplete (size: $LLM_SIZE bytes)"
    exit 1
fi

if [ "$MMPROJ_SIZE" -gt 800000000 ]; then
    echo "✅ Vision encoder file size OK (~819 MB)"
else
    echo "❌ Vision encoder file may be incomplete (size: $MMPROJ_SIZE bytes)"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ Download Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Copy service file to systemd:"
echo "     sudo cp services/llama-server.service /etc/systemd/system/"
echo ""
echo "  2. Reload and restart service:"
echo "     sudo systemctl daemon-reload"
echo "     sudo systemctl restart llama-server"
echo ""
echo "  3. Check service status:"
echo "     sudo systemctl status llama-server"
echo "     journalctl -u llama-server -f"
echo ""
