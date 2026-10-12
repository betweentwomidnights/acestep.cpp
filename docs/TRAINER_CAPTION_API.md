# Trainer audio analysis and caption API

The native server exposes suggestions for both SA3 and ACE-Step dataset trainers.
The caller owns dataset files, sidecars, caption pools, triggers, confidence
thresholds and user approval. This API does not start or register a training job.
It is not Carey's `/release_task` / `/query_result` protocol: gary4local still
needs a native adapter and a shared caption provider used by both trainers.

## Local BPM and key

`POST /analyze`, multipart field `audio`, runs on CPU without loading any models.
It returns JSON with `schema_version: 1`, `ok`, nullable integer `bpm`, string
`keyscale`, `suggestion`, `bpm_source`, `key_source`, nullable
`bpm_confidence` / `key_confidence`, and actual `duration` in seconds.

`ace-audio-analyze --in audio.wav` supplies the same suggestions as standalone
JSON, without a server, model, ggml backend or Python installation. It supports
Unicode filenames on Windows. `--control-info` reports the model-free contract;
`--version` reports the package version. The CLI omits duration.
The server still uses its normal model-registry startup requirement; use the
standalone CLI on machines with no model installation at all.

The estimators are ported unchanged from sa3.cpp commit
`3b7bfbf2939983ff3411dea92e5357f04a0ec6ad`: original-rate mono downmix, Kaiser
resampling, RMS-onset BPM and FFT/chroma key estimation. Attribution and MIT
licenses ship with both Windows and macOS packages. Confidence is an estimator
score, not a calibrated probability. Silence can produce missing suggestions.
Apply Gary's filename precedence, confidence thresholds, enharmonic spelling
and LM/local tempo agreement in the application rather than silently overwriting
user metadata upstream.

HTTP input is PCM WAV (8/16/24/32 bit) or IEEE float WAV (32/64 bit), mono or
multichannel, at its original sample rate. It rejects malformed, empty, truncated
and non-finite samples with HTTP 400; uploads over ten minutes get HTTP 413.
Decode other application formats to WAV before uploading. Analysis retains
amplitude; the caption encoder's stereo input is separately clamped and resampled
to 48 kHz. Analysis downmixes every channel; captioning uses the first two channels
(duplicating mono).

## Model captioning

`POST /caption` accepts the same WAV `audio` and an optional multipart `request`
JSON field. Explicitly choose model basenames from `/props` when any model bucket
contains more than one entry:

```json
{
  "lm_model": "acestep-5Hz-lm-0.6B-Q8_0.gguf",
  "synth_model": "acestep-v15-base-Q4_K_M.gguf",
  "vae": "vae-BF16.gguf",
  "lm_seed": 42,
  "lm_temperature": 0.3,
  "lm_top_p": 1.0,
  "lm_top_k": 0
}
```

The `synth_model` supplies the FSQ audio tokenizer, not an audio synthesis run.
It need not be the dataset's training base. Selecting a different tokenizer can
change the caption substantially; pin and validate the caption model bundle
separately from the XL-base Q4 training plan. Unknown or ambiguous model choices,
invalid sampling parameters and latent-only requests get HTTP 400 before weights
are loaded. Omitted sampling uses understand defaults (temperature 0.3).

The response is `{"id":"..."}`. Poll `GET /job?id=...`; when `status` is `done`,
fetch `GET /job?id=...&result=1` as ordinary JSON. Cancel with
`POST /job?id=...&cancel=1`, as for other native jobs. Model loading, VAE/FSQ
work and LM decoding use the existing serialized queue, memory policy and backend
selection. Cancellation during VAE/FSQ is observed at the next pipeline boundary;
LM decoding checks per token. Stopping competing services remains a supervisor
responsibility.

The result contains `schema_version: 1`, `caption`, identical `prompt`, `lyrics`,
LM `bpm`, `keyscale`, `time_signature`, `language`, actual WAV `duration`, resolved
`lm_seed`, selected `lm_model` / `synth_model` / `vae`, and a separate
`local_analysis` object. No audio codes, latent attachments or adapter files are
returned. `genre` is currently empty and `is_instrumental` is null: do not infer
either from those placeholders. Lyrics and prose are model suggestions and need
the application's existing quality checks and instrumental/trigger policy.
Empty/malformed caption text or an exhausted decode budget fails the job.
Decoding is bounded by the configured KV cache as well as its 4096-token limit.
The existing `/understand` multipart response remains available unchanged.

## Validation

Model-free CTests cover the real HTTP handlers, validation errors, CLI parity,
Unicode paths, silence and malformed samples. The opt-in real-model regression
requires a representative local WAV and verifies capture-enabled decoding,
same-seed replay, queued cancellation, normalized JSON and the actual LM backend:

```powershell
python tests/test-caption-real.py --server build/Release/ace-server.exe --models models --audio music.wav --lm acestep-5Hz-lm-0.6B-Q8_0.gguf --dit acestep-v15-base-Q4_K_M.gguf --vae vae-BF16.gguf --device CUDA0 --output caption-check
```

Use a new output directory for each run. CUDA's shared runtime must be available
on PATH when testing a split package. The test deliberately clears a global
`GGML_CUDA_DISABLE_GRAPHS` override: correctness must hold with capture active.
Passing this regression proves execution and structural quality, not accurate
genre, instruments, key or lyric transcription over a production dataset.
