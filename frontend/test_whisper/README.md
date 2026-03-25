# Whisper TVM

Apache TVM implementation of OpenAI's Whisper model.

## Usage

Compile model with TVM.

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

Download tokenizer.

```bash
uvx hf download openai/whisper-tiny tokenizer.json --local-dir whisper_tokenizer
```

Run compiled model.

```bash
python run_whisper_bundle.py \
  --artifacts-dir artifacts_whisper_${WHISPER_SIZE} \
  --flac jfk.flac \
  --device cuda \
  --tokffi-lib tokffi/build/libtokffi.so \
  --tokffi-tokenizer-dir whisper_tokenizer
```
