#!/bin/bash
# Build the WebRTC VAD shared library from source.
# Requires: gcc, make
set -e

SRC_DIR=$(dirname "$0")/cbits
BUILD_DIR=$(dirname "$0")

# Download webrtcvad source if not present
if [ ! -d "$SRC_DIR" ]; then
    echo "Downloading webrtcvad source..."
    TMP_DIR=$(mktemp -d)
    cd "$TMP_DIR"
    curl -sL https://github.com/wiseman/py-webrtcvad/archive/refs/tags/2.0.10.tar.gz | tar xz
    mkdir -p "$BUILD_DIR/cbits"
    cp -r py-webrtcvad-2.0.10/cbits/* "$BUILD_DIR/cbits/"
    cd "$BUILD_DIR"
    rm -rf "$TMP_DIR"
fi

echo "Compiling WebRTC VAD..."
cd "$BUILD_DIR"
gcc -c -fPIC -DWEBRTC_POSIX \
  -Icbits -Icbits/webrtc/common_audio/vad/include \
  -Icbits/webrtc/common_audio/signal_processing/include \
  cbits/webrtc/common_audio/vad/webrtc_vad.c \
  cbits/webrtc/common_audio/vad/vad_core.c \
  cbits/webrtc/common_audio/vad/vad_filterbank.c \
  cbits/webrtc/common_audio/vad/vad_gmm.c \
  cbits/webrtc/common_audio/vad/vad_sp.c \
  cbits/webrtc/common_audio/signal_processing/division_operations.c \
  cbits/webrtc/common_audio/signal_processing/energy.c \
  cbits/webrtc/common_audio/signal_processing/get_scaling_square.c \
  cbits/webrtc/common_audio/signal_processing/real_fft.c \
  cbits/webrtc/common_audio/signal_processing/complex_fft.c \
  cbits/webrtc/common_audio/signal_processing/complex_bit_reverse.c \
  cbits/webrtc/common_audio/signal_processing/cross_correlation.c \
  cbits/webrtc/common_audio/signal_processing/min_max_operations.c \
  cbits/webrtc/common_audio/signal_processing/downsample_fast.c \
  cbits/webrtc/common_audio/signal_processing/resample_48khz.c \
  cbits/webrtc/common_audio/signal_processing/resample_by_2_internal.c \
  cbits/webrtc/common_audio/signal_processing/resample_fractional.c \
  cbits/webrtc/common_audio/signal_processing/spl_init.c \
  cbits/webrtc/common_audio/signal_processing/vector_scaling_operations.c

gcc -shared -o src/libwebrtcvad.so *.o
rm -f *.o
echo "Done: src/libwebrtcvad.so"
