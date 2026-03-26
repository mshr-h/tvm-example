# Whisper TVM

Run OpenAI's Whisper with Apache TVM.

## Prerequisites

- uv
  - install via: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- tokenizers-tvm-ffi
  - install via: `uv pip install git+https://github.com/mshr-h/tokenizers-tvm-ffi.git --verbose`
- API server
  - `fastapi`, `uvicorn` and `python-multipart`
  - install via: `uv pip install fastapi uvicorn python-multipart`

## Usage

Choose targe model:

```bash
export WHISPER_SIZE=tiny
export WHISPER_SIZE=base
export WHISPER_SIZE=small
export WHISPER_SIZE=medium
export WHISPER_SIZE=large-v2
```

Compile model:

```bash
python compile_whisper_bundle.py \
  --model-id openai/whisper-${WHISPER_SIZE} \
  --output-dir artifacts_whisper_${WHISPER_SIZE} \
  --target cuda \
  --max-new-tokens 128
```

Run compiled model:

```bash
python run_whisper_bundle.py \
  --artifacts-dir ./artifacts_whisper_${WHISPER_SIZE}/ \
  --audio ./jfk.flac \
  --language auto \
  --timestamps \
  --json ./out.json
```

Run API server:

```bash
python serve_whisper_bundle_api.py \
  --artifacts-dir ./artifacts_whisper_${WHISPER_SIZE} \
  --served-model whisper-${WHISPER_SIZE} \
  --host 0.0.0.0 \
  --port 8000
```

Call API via curl:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@jfk.flac \
  -F model='whisper-${WHISPER_SIZE}' \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment'
```
