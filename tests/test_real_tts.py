"""Quick test: verify real sherpa-onnx TTS works with downloaded models."""
import time
import sherpa_onnx
import soundfile as sf

MODEL_DIR = "models/matcha-icefall-zh-baker"
VOCOS_PATH = "models/vocos-22khz-univ.onnx"

config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model=f"{MODEL_DIR}/model-steps-3.onnx",
            vocoder=VOCOS_PATH,
            lexicon=f"{MODEL_DIR}/lexicon.txt",
            tokens=f"{MODEL_DIR}/tokens.txt",
        ),
        num_threads=2,
        debug=False,
    ),
    rule_fsts=f"{MODEL_DIR}/phone.fst,{MODEL_DIR}/date.fst,{MODEL_DIR}/number.fst",
    max_num_sentences=1,
)

assert config.validate(), "Config validation failed!"

tts = sherpa_onnx.OfflineTts(config)

# Test short sentence
text = "今天天气真不错。"
gen_cfg = sherpa_onnx.GenerationConfig()
gen_cfg.sid = 0
gen_cfg.speed = 1.0
gen_cfg.silence_scale = 0.2

t0 = time.perf_counter()
audio = tts.generate(text, gen_cfg)
t1 = time.perf_counter()

dur = len(audio.samples) / audio.sample_rate
rtf = (t1 - t0) / dur

print(f"Text: {text}")
print(f"Samples: {len(audio.samples)}, Sample rate: {audio.sample_rate}")
print(f"Audio duration: {dur:.3f}s")
print(f"Generation time: {t1-t0:.3f}s")
print(f"RTF: {rtf:.3f}")

# Test longer sentence
text2 = "某某银行的副行长和一些行政领导表示，他们去过长江和长白山，经济不断增长。"
t0 = time.perf_counter()
audio2 = tts.generate(text2, gen_cfg)
t1 = time.perf_counter()

dur2 = len(audio2.samples) / audio2.sample_rate
rtf2 = (t1 - t0) / dur2

print(f"\nText: {text2}")
print(f"Audio duration: {dur2:.3f}s")
print(f"Generation time: {t1-t0:.3f}s")
print(f"RTF: {rtf2:.3f}")

# Save to file
sf.write("/tmp/test_tts_real.wav", audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
sf.write("/tmp/test_tts_real_long.wav", audio2.samples, samplerate=audio2.sample_rate, subtype="PCM_16")
print("\nSaved /tmp/test_tts_real.wav and /tmp/test_tts_real_long.wav")
