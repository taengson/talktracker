# Local model storage

TalkTracker keeps downloaded model files in this folder rather than in the shared Hugging Face cache.

Expected folders after setup:

```text
models/
├─ whisper-large-v3-turbo/
├─ whisper-large-v3-turbo-ct2/
└─ nemotron-3-diarization/
```

These files are intentionally excluded from Git. The initial download requires internet access. Afterward, TalkTracker loads the models from these local folders and can run without an internet connection.

Use `scripts/windows/Download-Models.ps1` to fetch model files here. The Nemotron download may require a Hugging Face token available as the `HF_TOKEN` environment variable.
