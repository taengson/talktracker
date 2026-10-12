# Local model storage

TalkTracker keeps downloaded model files in this folder rather than in the shared Hugging Face cache.

Expected folders after setup:

```text
models/
├─ whisper-large-v3-turbo/
├─ whisper-large-v3-turbo-ct2/
├─ whisper-large-v3-turbo-ct2-cuda-fp16/
└─ nemotron-3-diarization/
```

These files are intentionally excluded from Git. The initial download requires internet access. Afterward, TalkTracker loads the models from these local folders and can run without an internet connection.

Use `scripts/windows/Download-Models.ps1` on Windows, `scripts/macos/Download-Models.sh` on macOS, or `scripts/linux/Download-Models.sh` on Linux to fetch model files here. The Nemotron download may require a Hugging Face token available as the `HF_TOKEN` environment variable.

On a Windows or Linux PC where CTranslate2 can use CUDA, the preparation script creates both the CPU `-ct2` `int8` model and the `-ct2-cuda-fp16` GPU model. This keeps a local CPU fallback ready if GPU loading fails. macOS prepares the CPU model only.
