#!/usr/bin/env python3
"""Minimal Python demo for the standalone tokffi shared library.

Usage:
  uvx hf download openai/whisper-tiny tokenizer.json --local-dir whisper_tiny
  python tokffi_python_demo.py \
    --tokenizer-path whisper_tiny \
    --lib build/libtokffi.so \
    --text "Hello world"

The demo assumes the C++ side registered these global functions:
  - tokffi.TokenizerFromPath
  - tokffi.TokenizerEncode
  - tokffi.TokenizerDecode
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal Python demo for standalone tokffi")
    p.add_argument(
        "--lib",
        required=True,
        help="Path to the compiled shared library, e.g. ./build/libtokffi.so",
    )
    p.add_argument(
        "--tokenizer-path",
        required=True,
        help=(
            "Path to a tokenizer directory or file. For the minimal tokffi FromPath helper, "
            "this should typically be a directory containing tokenizer.json or tokenizer.model."
        ),
    )
    p.add_argument("--text", required=True, help="Input text to encode/decode")
    p.add_argument(
        "--print-metadata",
        action="store_true",
        help="Best-effort print of function metadata if the runtime provides it.",
    )
    return p.parse_args()


def _to_py_list(x: Any) -> list[int]:
    """Best-effort conversion of TVM-FFI IntTuple / tuple-like return values to list[int]."""
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, tuple):
        return [int(v) for v in x]
    if hasattr(x, "tolist"):
        y = x.tolist()
        if isinstance(y, list):
            return [int(v) for v in y]
    if hasattr(x, "__iter__"):
        return [int(v) for v in x]
    raise TypeError(f"Unable to convert encoded ids of type {type(x)!r} to Python list")


def main() -> int:
    args = parse_args()

    lib_path = pathlib.Path(args.lib)
    tok_path = pathlib.Path(args.tokenizer_path)
    if not lib_path.exists():
        raise FileNotFoundError(f"Shared library not found: {lib_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tok_path}")

    try:
        import tvm_ffi
    except Exception:  # pragma: no cover
        print("Failed to import tvm_ffi. Make sure tvm-ffi Python bindings are installed.", file=sys.stderr)
        raise

    # Load the shared library first so its static registration block runs.
    tvm_ffi.load_module(str(lib_path))
    print(f"Loaded module: {lib_path}")

    # Global registry lookups for the tokffi functions.
    tokenizer_from_path = tvm_ffi.get_global_func("tokffi.TokenizerFromPath")
    encode = tvm_ffi.get_global_func("tokffi.TokenizerEncode")
    decode = tvm_ffi.get_global_func("tokffi.TokenizerDecode")

    if args.print_metadata:
        for name in [
            "tokffi.TokenizerFromPath",
            "tokffi.TokenizerEncode",
            "tokffi.TokenizerDecode",
        ]:
            try:
                meta = tvm_ffi.get_global_func_metadata(name)
            except Exception:
                meta = None
            print(f"metadata[{name}] = {meta}")

    tokenizer = tokenizer_from_path(str(tok_path))
    ids = encode(tokenizer, args.text)
    decoded = decode(tokenizer, ids)

    ids_list = _to_py_list(ids)

    print(f"input_text   : {args.text!r}")
    print(f"encoded_ids  : {ids_list}")
    print(f"decoded_text : {decoded!r}")
    print(f"roundtrip_eq : {decoded == args.text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
