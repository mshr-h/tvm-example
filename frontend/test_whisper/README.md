# Whisper TVM

Run OpenAI's Whisper with Apache TVM.

## Prerequisites

```bash
uv pip install git+https://github.com/mshr-h/tokenizers-tvm-ffi.git --verbose
```

## Usage

Compile model.

```bash
# choose targe model
export WHISPER_SIZE=tiny
export WHISPER_SIZE=base
export WHISPER_SIZE=small
export WHISPER_SIZE=medium
export WHISPER_SIZE=large-v2

python compile_whisper_bundle.py \
  --model-id openai/whisper-${WHISPER_SIZE} \
  --output-dir artifacts_whisper_${WHISPER_SIZE} \
  --target cuda \
  --max-new-tokens 128
```

Run compiled model.

```bash
python run_whisper_bundle.py \
  --artifacts-dir ./artifacts_whisper_${WHISPER_SIZE}/ \
  --audio ./jfk.flac \
  --language auto \
  --timestamps \
  --json ./out.json
```
