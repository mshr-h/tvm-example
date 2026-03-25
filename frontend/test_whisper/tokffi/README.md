# tokffi (streaming)

Minimal standalone TVM-FFI wrapper around `tokenizers-cpp`, with a streaming detokenizer.

## Exposed API

Global functions:

- `tokffi.TokenizerFromHFJSONBytes(blob: str) -> tokffi.Tokenizer`
- `tokffi.TokenizerFromSentencePieceBytes(blob: str) -> tokffi.Tokenizer`
- `tokffi.TokenizerFromPath(path: str) -> tokffi.Tokenizer`
- `tokffi.TokenizerEncode(tok: tokffi.Tokenizer, text: str) -> IntTuple`
- `tokffi.TokenizerDecode(tok: tokffi.Tokenizer, ids: IntTuple) -> str`
- `tokffi.TextStreamer(tok: tokffi.Tokenizer) -> tokffi.TextStreamer`
- `tokffi.TextStreamerPut(streamer: tokffi.TextStreamer, ids: IntTuple) -> str`
- `tokffi.TextStreamerPutOne(streamer: tokffi.TextStreamer, token_id: int) -> str`
- `tokffi.TextStreamerFinish(streamer: tokffi.TextStreamer) -> str`

## Build

```bash
git clone <your tokffi repo>
cd tokffi
uv venv
source .venv/bin/activate

git submodule add https://github.com/mlc-ai/tokenizers-cpp 3rdparty/tokenizers-cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH=/path/to/tvm-ffi/install
cmake --build build --parallel
```

## Python usage

```python
import tvm_ffi

# Ensure the shared library is loaded so the static init block runs.
tvm_ffi.load_module("build/libtokffi.so")

from tvm_ffi import get_global_func

make_tok = get_global_func("tokffi.TokenizerFromPath")
encode = get_global_func("tokffi.TokenizerEncode")
decode = get_global_func("tokffi.TokenizerDecode")

tok = make_tok("/path/to/tokenizer_dir")
ids = encode(tok, "Hello tokffi")
text = decode(tok, ids)
print(list(ids))
print(text)
```

## Notes

The streaming detokenizer follows the same broad idea as MLC-LLM's `TextStreamer`:
- decode incrementally
- only return UTF-8-valid text
- keep back a few pending tokens when the decode currently ends in the UTF-8 replacement character

`TokenizerFromPath` still only supports `tokenizer.json` and `tokenizer.model`.
