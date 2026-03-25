import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import tvm_ffi
from scipy.signal import resample_poly

import tvm
from tvm import relax


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run WhisperBundle inference from exported artifacts without transformers. "
            "tokenizer.json must be downloaded separately, e.g. "
            "uvx hf download openai/whisper-tiny tokenizer.json --local-dir whisper_tiny"
        )
    )
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--flac", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--tokffi-lib", type=Path, required=True)
    parser.add_argument("--tokffi-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--print-ids", action="store_true")
    return parser.parse_args()


def choose_device(device_arg: str):
    if device_arg == "cuda":
        dev = tvm.cuda(0)
        if not dev.exist:
            raise RuntimeError("CUDA was requested, but tvm.cuda(0).exist is False.")
    elif device_arg == "cpu":
        dev = tvm.cpu(0)
    else:
        dev = tvm.cuda(0) if tvm.cuda(0).exist else tvm.cpu(0)
    return dev


def to_tvm_tensor(x, dev):
    if isinstance(x, np.ndarray):
        return tvm.runtime.tensor(x, dev)
    if hasattr(x, "copyto") and hasattr(x, "numpy"):
        return x.copyto(target=dev)
    raise TypeError(f"Unsupported tensor input type: {type(x)}")


def unwrap_vm_output(x):
    while not hasattr(x, "numpy"):
        x = x[0]
    return x


def unwrap_vm_outputs(x):
    if hasattr(x, "numpy"):
        return x
    return [unwrap_vm_outputs(x[i]) for i in range(len(x))]


class TokFFITextStreamer:
    def __init__(self, tokenizer_obj, streamer_obj, put_one_fn, finish_fn):
        self._tokenizer_obj = tokenizer_obj
        self._streamer_obj = streamer_obj
        self._put_one_fn = put_one_fn
        self._finish_fn = finish_fn

    def put_one(self, token_id: int) -> str:
        return str(self._put_one_fn(self._streamer_obj, int(token_id)))

    def finish(self) -> str:
        return str(self._finish_fn(self._streamer_obj))


def build_tokffi_text_streamer(lib_path: Path, tokenizer_dir: Path):
    if not lib_path.exists():
        raise FileNotFoundError(f"tokffi shared library not found: {lib_path}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"tokffi tokenizer path not found: {tokenizer_dir}")

    tvm_ffi.load_module(str(lib_path))
    tokenizer_from_path = tvm_ffi.get_global_func("tokffi.TokenizerFromPath")
    make_streamer = tvm_ffi.get_global_func("tokffi.TextStreamer")
    put_one_fn = tvm_ffi.get_global_func("tokffi.TextStreamerPutOne")
    finish_fn = tvm_ffi.get_global_func("tokffi.TextStreamerFinish")
    tokenizer_obj = tokenizer_from_path(str(tokenizer_dir))
    streamer_obj = make_streamer(tokenizer_obj)
    return TokFFITextStreamer(tokenizer_obj, streamer_obj, put_one_fn, finish_fn)


def strip_prefix_and_eos(seq: Sequence[int], prompt_ids: list[int], eos_token_id: int):
    seq = list(seq)
    if seq[: len(prompt_ids)] == prompt_ids:
        seq = seq[len(prompt_ids) :]
    if seq and seq[-1] == eos_token_id:
        seq = seq[:-1]
    return seq


def apply_whisper_suppression(logits_np, suppress_tokens, begin_suppress_tokens, generated_token_index: int):
    logits_np = logits_np.copy()
    if suppress_tokens:
        logits_np[np.array(suppress_tokens, dtype=np.int64)] = -np.inf
    if generated_token_index == 0 and begin_suppress_tokens:
        logits_np[np.array(begin_suppress_tokens, dtype=np.int64)] = -np.inf
    return logits_np


def load_audio_from_flac(src: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(src, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def pad_or_trim_audio(audio_1d: np.ndarray, n_samples: int):
    audio_1d = np.asarray(audio_1d, dtype=np.float32).reshape(-1)
    valid = min(audio_1d.shape[0], n_samples)
    if audio_1d.shape[0] < n_samples:
        audio_fixed = np.pad(audio_1d, (0, n_samples - audio_1d.shape[0]))
    else:
        audio_fixed = audio_1d[:n_samples]
    audio_fixed = np.asarray(audio_fixed, dtype=np.float32)[None, :]
    valid_samples = np.asarray([valid], dtype=np.int32)
    return audio_fixed, valid_samples


def make_past_keep_mask_np(past_len: int, max_past_len: int) -> np.ndarray:
    mask = np.full((1, 1, 1, max_past_len), -1e9, dtype=np.float32)
    if past_len > 0:
        mask[..., :past_len] = 0.0
    return mask


def load_params_npz(params_path: Path, dev, param_count: int):
    params_npz = np.load(params_path)
    params = []
    for i in range(param_count):
        key = f"p_{i:04d}"
        if key not in params_npz:
            raise KeyError(f"Missing parameter array: {key}")
        params.append(tvm.runtime.tensor(params_npz[key], dev))
    return params


def main():
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    metadata_path = artifacts_dir / "whisper_bundle_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lib_path = artifacts_dir / metadata["lib_name"]
    params_path = artifacts_dir / metadata["params_name"]
    if not lib_path.exists():
        raise FileNotFoundError(f"bundle library not found: {lib_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"bundle params not found: {params_path}")

    dev = choose_device(args.device)
    text_streamer = build_tokffi_text_streamer(args.tokffi_lib, args.tokffi_tokenizer_dir)

    lib = tvm.runtime.load_module(str(lib_path))
    vm = relax.VirtualMachine(lib, dev)
    params = load_params_npz(params_path, dev, int(metadata["param_count"]))

    audio = load_audio_from_flac(args.flac, int(metadata["sample_rate"]))
    waveform_fixed, valid_samples = pad_or_trim_audio(audio, int(metadata["n_samples"]))

    preprocess_out = vm["preprocess"](
        to_tvm_tensor(waveform_fixed, dev),
        to_tvm_tensor(valid_samples, dev),
    )
    input_features_tvm, feature_attention_mask_tvm = unwrap_vm_outputs(preprocess_out)
    input_features_np = unwrap_vm_output(input_features_tvm).numpy().astype(np.float32)
    feature_attention_mask_np = unwrap_vm_output(feature_attention_mask_tvm).numpy().astype(np.int32)
    _ = feature_attention_mask_np  # currently not used in TVM runtime path

    encoder_hidden_tvm = vm["encode"](
        to_tvm_tensor(input_features_np, dev),
        params,
    )
    encoder_hidden_np = unwrap_vm_output(encoder_hidden_tvm).numpy().astype(np.float32)

    cross_kv_out = vm["cross_kv"](
        to_tvm_tensor(encoder_hidden_np, dev),
        params,
    )
    cross_k_cache_tvm, cross_v_cache_tvm = unwrap_vm_outputs(cross_kv_out)

    prompt_ids = [int(x) for x in metadata["prompt_ids"]]
    eos_token_id = int(metadata["eos_token_id"])
    vocab_size = int(metadata["vocab_size"])
    special_ids = set(int(x) for x in metadata["special_ids"])
    suppress_tokens = [int(x) for x in metadata.get("suppress_tokens", [])]
    begin_suppress_tokens = [int(x) for x in metadata.get("begin_suppress_tokens", [])]

    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else metadata["max_new_tokens_default"])
    max_dec_len_compiled = int(metadata["max_dec_len_compiled"])
    max_runtime_dec_len = len(prompt_ids) + max_new_tokens
    if max_runtime_dec_len > max_dec_len_compiled:
        raise ValueError(
            f"Requested runtime length {max_runtime_dec_len} exceeds compiled limit {max_dec_len_compiled}."
        )

    decoder_layers = int(metadata["decoder_layers"])
    decoder_attention_heads = int(metadata["decoder_attention_heads"])
    head_dim = int(metadata["head_dim"])
    max_past_len = max_dec_len_compiled - 1

    self_k_cache_np = np.zeros(
        (decoder_layers, 1, decoder_attention_heads, max_past_len, head_dim),
        dtype=np.float32,
    )
    self_v_cache_np = np.zeros_like(self_k_cache_np)

    last_logits_np = None
    for pos, tok in enumerate(prompt_ids):
        out = vm["decode_step"](
            to_tvm_tensor(np.asarray([[tok]], dtype=np.int32), dev),
            to_tvm_tensor(np.asarray([[pos]], dtype=np.int32), dev),
            to_tvm_tensor(self_k_cache_np, dev),
            to_tvm_tensor(self_v_cache_np, dev),
            to_tvm_tensor(make_past_keep_mask_np(pos, max_past_len), dev),
            cross_k_cache_tvm,
            cross_v_cache_tvm,
            params,
        )
        logits_tvm, new_k_tvm, new_v_tvm = unwrap_vm_outputs(out)
        last_logits_np = unwrap_vm_output(logits_tvm).numpy().astype(np.float32)
        new_k_np = unwrap_vm_output(new_k_tvm).numpy().astype(np.float32)
        new_v_np = unwrap_vm_output(new_v_tvm).numpy().astype(np.float32)
        self_k_cache_np[:, :, :, pos : pos + 1, :] = new_k_np
        self_v_cache_np[:, :, :, pos : pos + 1, :] = new_v_np

    tvm_generated_content_ids = []
    cached_tokens = len(prompt_ids)
    tvm_logits_np = last_logits_np

    for gen_idx in range(max_new_tokens):
        next_logits = apply_whisper_suppression(
            tvm_logits_np[0],
            suppress_tokens,
            begin_suppress_tokens,
            gen_idx,
        )
        next_id = int(np.argmax(next_logits))
        if not (0 <= next_id < vocab_size):
            raise RuntimeError(f"Out-of-range token id: {next_id}")
        tvm_generated_content_ids.append(next_id)

        if next_id not in special_ids:
            delta = text_streamer.put_one(next_id)
            if delta:
                print(delta, end="", flush=True)

        if next_id == eos_token_id or gen_idx == max_new_tokens - 1:
            break

        out = vm["decode_step"](
            to_tvm_tensor(np.asarray([[next_id]], dtype=np.int32), dev),
            to_tvm_tensor(np.asarray([[cached_tokens]], dtype=np.int32), dev),
            to_tvm_tensor(self_k_cache_np, dev),
            to_tvm_tensor(self_v_cache_np, dev),
            to_tvm_tensor(make_past_keep_mask_np(cached_tokens, max_past_len), dev),
            cross_k_cache_tvm,
            cross_v_cache_tvm,
            params,
        )
        logits_tvm, new_k_tvm, new_v_tvm = unwrap_vm_outputs(out)
        tvm_logits_np = unwrap_vm_output(logits_tvm).numpy().astype(np.float32)
        new_k_np = unwrap_vm_output(new_k_tvm).numpy().astype(np.float32)
        new_v_np = unwrap_vm_output(new_v_tvm).numpy().astype(np.float32)
        self_k_cache_np[:, :, :, cached_tokens : cached_tokens + 1, :] = new_k_np
        self_v_cache_np[:, :, :, cached_tokens : cached_tokens + 1, :] = new_v_np
        cached_tokens += 1

    tail = text_streamer.finish()
    if tail:
        print(tail, end="", flush=True)
    print()

    tvm_full_ids = prompt_ids + tvm_generated_content_ids
    tvm_content_ids = strip_prefix_and_eos(tvm_full_ids, prompt_ids, eos_token_id)

    if args.print_ids:
        print(f"[full ids]    {tvm_full_ids}")
        print(f"[content ids] {tvm_content_ids}")


if __name__ == "__main__":
    main()
