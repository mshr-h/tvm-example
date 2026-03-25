import argparse
import copy
import math
import tempfile
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import soundfile as sf
import torch
import tvm
import tvm_ffi
from scipy.signal import resample_poly
from transformers import AutoProcessor, WhisperForConditionalGeneration
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

SAMPLE_RATE = 16000
CHUNK_LENGTH = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH  # 480000
N_FFT = 400
HOP_LENGTH = 160
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000
N_FREQ = 1 + N_FFT // 2  # 201
REFLECT_PAD = N_FFT // 2  # 200
N_MELS = 80

HF_LANGUAGE = "en"
PROMPT_LANGUAGE = "english"
TASK = "transcribe"
NO_TIMESTAMPS = True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Whisper preprocess/encoder/cached decoder-step in tvm.relax.frontend.nn "
            "with Hugging Face weight copy and numerical comparison."
        )
    )
    parser.add_argument("--flac", type=Path, required=True)
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--tokffi-lib",
        type=Path,
        required=True,
        help="Path to the compiled standalone tokffi shared library, e.g. ./build/libtokffi.so",
    )
    parser.add_argument(
        "--tokffi-tokenizer-dir",
        type=Path,
        default=None,
        help=(
            "Optional local tokenizer directory for tokffi.TokenizerFromPath. "
            "If omitted, the script saves the Hugging Face tokenizer to a temporary "
            "directory and loads tokffi from there."
        ),
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def choose_device_and_target(device_arg: str):
    if device_arg == "cuda":
        dev = tvm.cuda(0)
        if not dev.exist:
            raise RuntimeError("CUDA was requested, but tvm.cuda(0).exist is False.")
    elif device_arg == "cpu":
        dev = tvm.cpu(0)
    else:
        dev = tvm.cuda(0) if tvm.cuda(0).exist else tvm.cpu(0)
    target = tvm.target.Target.from_device(dev)
    return dev, target


def get_s_tir_pipeline():
    return tvm.transform.Sequential(
        [
            tvm.s_tir.transform.DefaultGPUSchedule(),
            tvm.s_tir.pipeline.default_s_tir_pipeline(),
        ]
    )


def compile_nn_module_to_vm(module: nn.Module, target, dev):
    mod, named_params = module.export_tvm(spec=module.get_default_spec())
    compile_kwargs = {"target": target}
    if target.kind.name == "cuda":
        compile_kwargs["tir_pipeline"] = get_s_tir_pipeline()
    ex = tvm.compile(mod, **compile_kwargs)
    vm = relax.VirtualMachine(ex, dev)

    params_tvm = []
    for _, param in named_params:
        if param.data is None:
            raise ValueError("Encountered unbound parameter during compilation")
        params_tvm.append(param.data.copyto(target=dev))
    return vm, named_params, params_tvm


def to_tvm_tensor(x, dev):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
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


def report_diff(name: str, ref: np.ndarray, test: np.ndarray):
    ref = np.asarray(ref)
    test = np.asarray(test)
    diff = ref - test
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    print(f"[{name}] max_abs={max_abs:.8e} mean_abs={mean_abs:.8e} rmse={rmse:.8e}")


def maybe_reconstruct_full_hf_ids(hf_ids: torch.Tensor, prompt_ids: List[int], eos_token_id: int):
    if hf_ids.ndim != 2 or hf_ids.shape[0] != 1:
        return hf_ids

    starts_with_prompt = hf_ids.shape[1] > 0 and int(hf_ids[0, 0]) == prompt_ids[0]
    if starts_with_prompt:
        return hf_ids

    prompt = torch.tensor([prompt_ids], dtype=hf_ids.dtype, device=hf_ids.device)
    if hf_ids.shape[1] == 0 or int(hf_ids[0, -1]) != eos_token_id:
        eos = torch.tensor([[eos_token_id]], dtype=hf_ids.dtype, device=hf_ids.device)
        return torch.cat([prompt, hf_ids, eos], dim=1)
    return torch.cat([prompt, hf_ids], dim=1)


def strip_prefix_and_eos(seq: Sequence[int], prompt_ids: List[int], eos_token_id: int):
    seq = list(seq)
    if seq[: len(prompt_ids)] == prompt_ids:
        seq = seq[len(prompt_ids) :]
    if seq and seq[-1] == eos_token_id:
        seq = seq[:-1]
    return seq


def filter_special_ids(seq: Sequence[int], special_ids: set[int]):
    return [int(tok) for tok in seq if int(tok) not in special_ids]


def _to_py_list(x: Any) -> list[int]:
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


class TokFFITokenizer:
    def __init__(self, tokenizer_obj, encode_fn, decode_fn):
        self._tokenizer_obj = tokenizer_obj
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn

    def encode(self, text: str) -> list[int]:
        return _to_py_list(self._encode_fn(self._tokenizer_obj, text))

    def decode(self, token_ids: Sequence[int]) -> str:
        token_ids = tuple(int(tok) for tok in token_ids)
        return str(self._decode_fn(self._tokenizer_obj, token_ids))


def build_tokffi_tokenizer(processor, lib_path: Path, tokenizer_dir: Path | None):
    if not lib_path.exists():
        raise FileNotFoundError(f"tokffi shared library not found: {lib_path}")

    tmpdir = None
    if tokenizer_dir is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="whisper_tokffi_tokenizer_")
        tokenizer_dir = Path(tmpdir.name)
        processor.tokenizer.save_pretrained(tokenizer_dir)

    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"tokffi tokenizer path not found: {tokenizer_dir}")

    tvm_ffi.load_module(str(lib_path))
    tokenizer_from_path = tvm_ffi.get_global_func("tokffi.TokenizerFromPath")
    encode_fn = tvm_ffi.get_global_func("tokffi.TokenizerEncode")
    decode_fn = tvm_ffi.get_global_func("tokffi.TokenizerDecode")

    tokenizer_obj = tokenizer_from_path(str(tokenizer_dir))
    tokenizer = TokFFITokenizer(tokenizer_obj, encode_fn, decode_fn)
    return tokenizer, tmpdir


def decode_with_tokffi(tokenizer: TokFFITokenizer, token_ids: Sequence[int], special_ids: set[int]):
    token_ids = filter_special_ids(token_ids, special_ids)
    return tokenizer.decode(token_ids)


def apply_whisper_suppression(logits_np, generation_config, generated_token_index: int):
    logits_np = logits_np.copy()
    suppress_tokens = getattr(generation_config, "suppress_tokens", None)
    if suppress_tokens is not None:
        logits_np[np.array(suppress_tokens, dtype=np.int64)] = -np.inf

    begin_suppress_tokens = getattr(generation_config, "begin_suppress_tokens", None)
    if generated_token_index == 0 and begin_suppress_tokens is not None:
        logits_np[np.array(begin_suppress_tokens, dtype=np.int64)] = -np.inf
    return logits_np


def get_decoder_prompt_ids(processor, decoder_start_token_id: int):
    if hasattr(processor, "get_decoder_prompt_ids"):
        forced = processor.get_decoder_prompt_ids(
            language=PROMPT_LANGUAGE,
            task=TASK,
            no_timestamps=NO_TIMESTAMPS,
        )
    else:
        forced = processor.tokenizer.get_decoder_prompt_ids(
            language=PROMPT_LANGUAGE,
            task=TASK,
            no_timestamps=NO_TIMESTAMPS,
        )
    return [decoder_start_token_id] + [tok for _, tok in forced]


def load_audio_from_flac(src: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(src, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def pad_or_trim_audio(audio_1d: np.ndarray, n_samples: int = N_SAMPLES):
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


def add_axis0(x: Tensor):
    shape = [1] + list(x.shape)
    return op.reshape(x, shape)


# -----------------------------------------------------------------------------
# TVM-native preprocess
# -----------------------------------------------------------------------------
def make_reflect_indices(n_samples: int = N_SAMPLES, pad: int = REFLECT_PAD) -> np.ndarray:
    left = np.arange(pad, 0, -1, dtype=np.int32)
    center = np.arange(n_samples, dtype=np.int32)
    right = np.arange(n_samples - 2, n_samples - pad - 2, -1, dtype=np.int32)
    return np.concatenate([left, center, right], axis=0)


def make_periodic_hann(n_fft: int = N_FFT) -> np.ndarray:
    n = np.arange(n_fft, dtype=np.float32)
    return 0.5 - 0.5 * np.cos((2.0 * np.pi * n) / float(n_fft))


def make_stft_conv_kernels(n_fft: int = N_FFT, n_freq: int = N_FREQ):
    n = np.arange(n_fft, dtype=np.float32)
    k = np.arange(n_freq, dtype=np.float32)[:, None]
    window = make_periodic_hann(n_fft)[None, :]
    phase = (2.0 * np.pi / float(n_fft)) * (k * n[None, :])
    real = np.cos(phase) * window
    imag = -np.sin(phase) * window
    return real.astype(np.float32)[:, None, :], imag.astype(np.float32)[:, None, :]


class WhisperPreprocessTVM(nn.Module):
    def __init__(self, mel_filters: np.ndarray):
        super().__init__()
        mel_filters = np.asarray(mel_filters, dtype=np.float32)
        if mel_filters.shape == (N_MELS, N_FREQ):
            mel_filters = mel_filters.T
        if mel_filters.shape != (N_FREQ, N_MELS):
            raise ValueError(f"Unexpected mel filter shape: {mel_filters.shape}")

        real_kernel, imag_kernel = make_stft_conv_kernels()
        self.reflect_indices = Tensor.from_const(make_reflect_indices())
        self.keep_frame_indices = Tensor.from_const(np.arange(N_FRAMES, dtype=np.int32))
        self.frame_starts = Tensor.from_const(np.arange(0, N_SAMPLES, HOP_LENGTH, dtype=np.int32).reshape(1, N_FRAMES))
        self.real_kernel = Tensor.from_const(real_kernel)
        self.imag_kernel = Tensor.from_const(imag_kernel)
        self.mel_filters = Tensor.from_const(mel_filters)
        self.log_eps = Tensor.from_const(np.array(1e-10, dtype=np.float32))
        self.inv_ln10 = Tensor.from_const(np.array(1.0 / math.log(10.0), dtype=np.float32))
        self.eight = Tensor.from_const(np.array(8.0, dtype=np.float32))
        self.four = Tensor.from_const(np.array(4.0, dtype=np.float32))

    def forward(self, waveform: Tensor, valid_samples: Tensor):
        x = op.take(waveform, self.reflect_indices, axis=1)
        x = op.reshape(x, [1, 1, N_SAMPLES + 2 * REFLECT_PAD])
        real = op.conv1d(x, self.real_kernel, stride=HOP_LENGTH, padding=0)
        imag = op.conv1d(x, self.imag_kernel, stride=HOP_LENGTH, padding=0)
        power = op.add(op.square(real), op.square(imag))
        power = op.take(power, self.keep_frame_indices, axis=2)
        power_t = op.permute_dims(power, axes=[0, 2, 1])
        mel = op.matmul(power_t, self.mel_filters)
        mel = op.permute_dims(mel, axes=[0, 2, 1])
        log_spec = op.log(op.maximum(mel, self.log_eps))
        log_spec = op.multiply(log_spec, self.inv_ln10)
        max_val = op.max(log_spec, axis=[1, 2], keepdims=True)
        log_spec = op.maximum(log_spec, op.subtract(max_val, self.eight))
        input_features = op.divide(op.add(log_spec, self.four), self.four)
        valid_samples_2d = op.broadcast_to(op.reshape(valid_samples, [1, 1]), [1, N_FRAMES])
        feature_attention_mask = op.astype(op.less(self.frame_starts, valid_samples_2d), "int32")
        return input_features, feature_attention_mask

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "waveform": nn.spec.Tensor([1, N_SAMPLES], "float32"),
                    "valid_samples": nn.spec.Tensor([1], "int32"),
                    "$": {"param_mode": "none", "effect_mode": "none"},
                }
            },  # ty:ignore[invalid-argument-type]
            self,
        )


# -----------------------------------------------------------------------------
# TVM-native encoder
# -----------------------------------------------------------------------------
class WhisperAttentionTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.scale = Tensor.from_const(np.array(self.head_dim**-0.5, dtype=np.float32))

    def _reshape_qkv(self, x: Tensor):
        bsz, seq_len, _ = x.shape
        x = op.reshape(x, [bsz, seq_len, self.num_heads, self.head_dim])
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return x

    def _merge_heads(self, x: Tensor):
        bsz, num_heads, seq_len, head_dim = x.shape
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        x = op.reshape(x, [bsz, seq_len, num_heads * head_dim])
        return x

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None):
        query_states = op.multiply(self.q_proj(hidden_states), self.scale)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = self._reshape_qkv(query_states)
        key_states = self._reshape_qkv(key_states)
        value_states = self._reshape_qkv(value_states)
        attn_weights = op.matmul(query_states, op.permute_dims(key_states, axes=[0, 1, 3, 2]))
        if attention_mask is not None:
            attn_weights = op.add(attn_weights, attention_mask)
        attn_weights = op.softmax(attn_weights, axis=-1)
        attn_output = op.matmul(attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        return self.out_proj(attn_output)


class WhisperEncoderLayerTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_attn = WhisperAttentionTVM(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, hidden_states: Tensor):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=None)
        hidden_states = op.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = op.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = op.add(residual, hidden_states)
        return hidden_states


class WhisperEncoderTVM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_mel_bins = int(config.num_mel_bins)
        self.d_model = int(config.d_model)
        self.max_source_positions = int(config.max_source_positions)
        self.encoder_layers = int(config.encoder_layers)
        self.encoder_attention_heads = int(config.encoder_attention_heads)
        self.encoder_ffn_dim = int(config.encoder_ffn_dim)
        self.conv1 = nn.Conv1D(self.num_mel_bins, self.d_model, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv1D(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1, bias=True)
        self.embed_positions = nn.Embedding(self.max_source_positions, self.d_model)
        self.layers = []
        for i in range(self.encoder_layers):
            layer = WhisperEncoderLayerTVM(self.d_model, self.encoder_attention_heads, self.encoder_ffn_dim)
            setattr(self, f"layer_{i}", layer)
            self.layers.append(layer)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.position_ids = Tensor.from_const(np.arange(self.max_source_positions, dtype=np.int32)[None, :])

    def forward(self, input_features: Tensor):
        hidden_states = self.conv1(input_features)
        hidden_states = op.gelu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = op.gelu(hidden_states)
        hidden_states = op.permute_dims(hidden_states, axes=[0, 2, 1])
        hidden_states = op.add(hidden_states, self.embed_positions(self.position_ids))
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "input_features": nn.spec.Tensor([1, N_MELS, N_FRAMES], "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },  # ty:ignore[invalid-argument-type]
            self,
        )


# -----------------------------------------------------------------------------
# Cross-KV precompute for decoder
# -----------------------------------------------------------------------------
class WhisperCrossKVLayerTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _reshape_qkv(self, x: Tensor):
        bsz, seq_len, _ = x.shape
        x = op.reshape(x, [bsz, seq_len, self.num_heads, self.head_dim])
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return x

    def forward(self, encoder_hidden_states: Tensor):
        k = self._reshape_qkv(self.k_proj(encoder_hidden_states))
        v = self._reshape_qkv(self.v_proj(encoder_hidden_states))
        return k, v


class WhisperCrossKVCachedTVM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = int(config.d_model)
        self.decoder_layers = int(config.decoder_layers)
        self.decoder_attention_heads = int(config.decoder_attention_heads)
        self.max_source_positions = int(config.max_source_positions)
        self.head_dim = self.d_model // self.decoder_attention_heads
        self.layers = []
        for i in range(self.decoder_layers):
            layer = WhisperCrossKVLayerTVM(self.d_model, self.decoder_attention_heads)
            setattr(self, f"layer_{i}", layer)
            self.layers.append(layer)

    def forward(self, encoder_hidden_states: Tensor):
        all_k = []
        all_v = []
        for layer in self.layers:
            k, v = layer(encoder_hidden_states)
            all_k.append(add_axis0(k))
            all_v.append(add_axis0(v))
        return op.concat(all_k, dim=0), op.concat(all_v, dim=0)

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "encoder_hidden_states": nn.spec.Tensor([1, self.max_source_positions, self.d_model], "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },  # ty:ignore[invalid-argument-type]
            self,
        )


# -----------------------------------------------------------------------------
# Cached decoder-step (explicit cache tensors)
# -----------------------------------------------------------------------------
class WhisperSelfAttentionStepTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_past_len: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_past_len = max_past_len
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.scale = Tensor.from_const(np.array(self.head_dim**-0.5, dtype=np.float32))
        self.zero_mask = Tensor.from_const(np.zeros((1, 1, 1, 1), dtype=np.float32))

    def _reshape_qkv(self, x: Tensor):
        bsz, seq_len, _ = x.shape
        x = op.reshape(x, [bsz, seq_len, self.num_heads, self.head_dim])
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return x

    def _merge_heads(self, x: Tensor):
        bsz, num_heads, seq_len, head_dim = x.shape
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        x = op.reshape(x, [bsz, seq_len, num_heads * head_dim])
        return x

    def forward(self, hidden_states: Tensor, past_k: Tensor, past_v: Tensor, past_keep_mask: Tensor):
        query_states = op.multiply(self.q_proj(hidden_states), self.scale)
        new_k = self.k_proj(hidden_states)
        new_v = self.v_proj(hidden_states)
        query_states = self._reshape_qkv(query_states)
        new_k = self._reshape_qkv(new_k)
        new_v = self._reshape_qkv(new_v)
        all_k = op.concat([past_k, new_k], dim=2)
        all_v = op.concat([past_v, new_v], dim=2)
        attn_mask = op.concat([past_keep_mask, self.zero_mask], dim=-1)
        attn_weights = op.matmul(query_states, op.permute_dims(all_k, axes=[0, 1, 3, 2]))
        attn_weights = op.add(attn_weights, attn_mask)
        attn_weights = op.softmax(attn_weights, axis=-1)
        attn_output = op.matmul(attn_weights, all_v)
        attn_output = self._merge_heads(attn_output)
        return self.out_proj(attn_output), new_k, new_v


class WhisperCrossAttentionStepTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.scale = Tensor.from_const(np.array(self.head_dim**-0.5, dtype=np.float32))

    def _reshape_qkv(self, x: Tensor):
        bsz, seq_len, _ = x.shape
        x = op.reshape(x, [bsz, seq_len, self.num_heads, self.head_dim])
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return x

    def _merge_heads(self, x: Tensor):
        bsz, num_heads, seq_len, head_dim = x.shape
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        x = op.reshape(x, [bsz, seq_len, num_heads * head_dim])
        return x

    def forward(self, hidden_states: Tensor, cross_k: Tensor, cross_v: Tensor):
        query_states = op.multiply(self.q_proj(hidden_states), self.scale)
        query_states = self._reshape_qkv(query_states)
        attn_weights = op.matmul(query_states, op.permute_dims(cross_k, axes=[0, 1, 3, 2]))
        attn_weights = op.softmax(attn_weights, axis=-1)
        attn_output = op.matmul(attn_weights, cross_v)
        attn_output = self._merge_heads(attn_output)
        return self.out_proj(attn_output)


class WhisperDecoderLayerStepTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, max_past_len: int):
        super().__init__()
        self.self_attn = WhisperSelfAttentionStepTVM(embed_dim, num_heads, max_past_len)
        self.encoder_attn = WhisperCrossAttentionStepTVM(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(
        self,
        hidden_states: Tensor,
        past_k: Tensor,
        past_v: Tensor,
        past_keep_mask: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, new_k, new_v = self.self_attn(hidden_states, past_k, past_v, past_keep_mask)
        hidden_states = op.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(hidden_states, cross_k, cross_v)
        hidden_states = op.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = op.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = op.add(residual, hidden_states)
        return hidden_states, new_k, new_v


class WhisperDecoderCachedStepTVM(nn.Module):
    def __init__(self, config, max_dec_len: int):
        super().__init__()
        self.vocab_size = int(config.vocab_size)
        self.d_model = int(config.d_model)
        self.max_target_positions = int(config.max_target_positions)
        self.decoder_layers = int(config.decoder_layers)
        self.decoder_attention_heads = int(config.decoder_attention_heads)
        self.decoder_ffn_dim = int(config.decoder_ffn_dim)
        self.max_source_positions = int(config.max_source_positions)
        self.max_dec_len = max_dec_len
        self.max_past_len = max_dec_len - 1
        self.head_dim = self.d_model // self.decoder_attention_heads
        if max_dec_len > self.max_target_positions:
            raise ValueError("max_dec_len exceeds Whisper config max_target_positions")
        self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)
        self.embed_positions = nn.Embedding(self.max_target_positions, self.d_model)
        self.layers = []
        self.layer_take_indices = []
        for i in range(self.decoder_layers):
            layer = WhisperDecoderLayerStepTVM(
                self.d_model, self.decoder_attention_heads, self.decoder_ffn_dim, self.max_past_len
            )
            setattr(self, f"layer_{i}", layer)
            self.layers.append(layer)
            self.layer_take_indices.append(Tensor.from_const(np.asarray([i], dtype=np.int32)))
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.proj_out = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def _take_layer(self, stacked: Tensor, layer_index: Tensor, out_shape):
        x = op.take(stacked, layer_index, axis=0)
        return op.reshape(x, out_shape)

    def forward(
        self,
        token_id: Tensor,
        position_id: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        past_keep_mask: Tensor,
        cross_k_cache: Tensor,
        cross_v_cache: Tensor,
    ):
        hidden_states = self.embed_tokens(token_id)
        hidden_states = op.add(hidden_states, self.embed_positions(position_id))
        new_k_list = []
        new_v_list = []
        for i, layer in enumerate(self.layers):
            idx = self.layer_take_indices[i]
            past_k = self._take_layer(
                self_k_cache, idx, [1, self.decoder_attention_heads, self.max_past_len, self.head_dim]
            )
            past_v = self._take_layer(
                self_v_cache, idx, [1, self.decoder_attention_heads, self.max_past_len, self.head_dim]
            )
            cross_k = self._take_layer(
                cross_k_cache, idx, [1, self.decoder_attention_heads, self.max_source_positions, self.head_dim]
            )
            cross_v = self._take_layer(
                cross_v_cache, idx, [1, self.decoder_attention_heads, self.max_source_positions, self.head_dim]
            )
            hidden_states, new_k, new_v = layer(hidden_states, past_k, past_v, past_keep_mask, cross_k, cross_v)
            new_k_list.append(add_axis0(new_k))
            new_v_list.append(add_axis0(new_v))
        hidden_states = self.layer_norm(hidden_states)
        logits = self.proj_out(hidden_states)
        logits = op.reshape(logits, [1, self.vocab_size])
        return logits, op.concat(new_k_list, dim=0), op.concat(new_v_list, dim=0)

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "token_id": nn.spec.Tensor([1, 1], "int32"),
                    "position_id": nn.spec.Tensor([1, 1], "int32"),
                    "self_k_cache": nn.spec.Tensor(
                        [self.decoder_layers, 1, self.decoder_attention_heads, self.max_past_len, self.head_dim],
                        "float32",
                    ),
                    "self_v_cache": nn.spec.Tensor(
                        [self.decoder_layers, 1, self.decoder_attention_heads, self.max_past_len, self.head_dim],
                        "float32",
                    ),
                    "past_keep_mask": nn.spec.Tensor([1, 1, 1, self.max_past_len], "float32"),
                    "cross_k_cache": nn.spec.Tensor(
                        [
                            self.decoder_layers,
                            1,
                            self.decoder_attention_heads,
                            self.max_source_positions,
                            self.head_dim,
                        ],
                        "float32",
                    ),
                    "cross_v_cache": nn.spec.Tensor(
                        [
                            self.decoder_layers,
                            1,
                            self.decoder_attention_heads,
                            self.max_source_positions,
                            self.head_dim,
                        ],
                        "float32",
                    ),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },  # ty:ignore[invalid-argument-type]
            self,
        )


# -----------------------------------------------------------------------------
# Weight copy
# -----------------------------------------------------------------------------
def bind_param_from_torch(param: nn.Parameter, tensor: torch.Tensor):
    arr = tensor.detach().cpu().numpy()
    if str(arr.dtype) != param.dtype:
        arr = arr.astype(param.dtype)
    param.data = tvm.runtime.tensor(arr)


def copy_encoder_weights_from_hf(tvm_encoder: WhisperEncoderTVM, hf_model: WhisperForConditionalGeneration):
    hf_state = hf_model.state_dict()
    bind_param_from_torch(tvm_encoder.conv1.weight, hf_state["model.encoder.conv1.weight"])
    bind_param_from_torch(tvm_encoder.conv1.bias, hf_state["model.encoder.conv1.bias"])
    bind_param_from_torch(tvm_encoder.conv2.weight, hf_state["model.encoder.conv2.weight"])
    bind_param_from_torch(tvm_encoder.conv2.bias, hf_state["model.encoder.conv2.bias"])
    bind_param_from_torch(tvm_encoder.embed_positions.weight, hf_state["model.encoder.embed_positions.weight"])
    bind_param_from_torch(tvm_encoder.layer_norm.weight, hf_state["model.encoder.layer_norm.weight"])
    bind_param_from_torch(tvm_encoder.layer_norm.bias, hf_state["model.encoder.layer_norm.bias"])
    for i, layer in enumerate(tvm_encoder.layers):
        prefix = f"model.encoder.layers.{i}."
        bind_param_from_torch(layer.self_attn.q_proj.weight, hf_state[prefix + "self_attn.q_proj.weight"])
        bind_param_from_torch(layer.self_attn.q_proj.bias, hf_state[prefix + "self_attn.q_proj.bias"])
        bind_param_from_torch(layer.self_attn.k_proj.weight, hf_state[prefix + "self_attn.k_proj.weight"])
        bind_param_from_torch(layer.self_attn.v_proj.weight, hf_state[prefix + "self_attn.v_proj.weight"])
        bind_param_from_torch(layer.self_attn.v_proj.bias, hf_state[prefix + "self_attn.v_proj.bias"])
        bind_param_from_torch(layer.self_attn.out_proj.weight, hf_state[prefix + "self_attn.out_proj.weight"])
        bind_param_from_torch(layer.self_attn.out_proj.bias, hf_state[prefix + "self_attn.out_proj.bias"])
        bind_param_from_torch(layer.self_attn_layer_norm.weight, hf_state[prefix + "self_attn_layer_norm.weight"])
        bind_param_from_torch(layer.self_attn_layer_norm.bias, hf_state[prefix + "self_attn_layer_norm.bias"])
        bind_param_from_torch(layer.fc1.weight, hf_state[prefix + "fc1.weight"])
        bind_param_from_torch(layer.fc1.bias, hf_state[prefix + "fc1.bias"])
        bind_param_from_torch(layer.fc2.weight, hf_state[prefix + "fc2.weight"])
        bind_param_from_torch(layer.fc2.bias, hf_state[prefix + "fc2.bias"])
        bind_param_from_torch(layer.final_layer_norm.weight, hf_state[prefix + "final_layer_norm.weight"])
        bind_param_from_torch(layer.final_layer_norm.bias, hf_state[prefix + "final_layer_norm.bias"])


def copy_cross_kv_weights_from_hf(tvm_cross: WhisperCrossKVCachedTVM, hf_model: WhisperForConditionalGeneration):
    hf_state = hf_model.state_dict()
    for i, layer in enumerate(tvm_cross.layers):
        prefix = f"model.decoder.layers.{i}.encoder_attn."
        bind_param_from_torch(layer.k_proj.weight, hf_state[prefix + "k_proj.weight"])
        bind_param_from_torch(layer.v_proj.weight, hf_state[prefix + "v_proj.weight"])
        bind_param_from_torch(layer.v_proj.bias, hf_state[prefix + "v_proj.bias"])


def copy_decoder_step_weights_from_hf(
    tvm_decoder: WhisperDecoderCachedStepTVM, hf_model: WhisperForConditionalGeneration
):
    hf_state = hf_model.state_dict()
    bind_param_from_torch(tvm_decoder.embed_tokens.weight, hf_state["model.decoder.embed_tokens.weight"])
    bind_param_from_torch(tvm_decoder.embed_positions.weight, hf_state["model.decoder.embed_positions.weight"])
    bind_param_from_torch(tvm_decoder.layer_norm.weight, hf_state["model.decoder.layer_norm.weight"])
    bind_param_from_torch(tvm_decoder.layer_norm.bias, hf_state["model.decoder.layer_norm.bias"])
    bind_param_from_torch(tvm_decoder.proj_out.weight, hf_state["proj_out.weight"])
    for i, layer in enumerate(tvm_decoder.layers):
        prefix = f"model.decoder.layers.{i}."
        bind_param_from_torch(layer.self_attn.q_proj.weight, hf_state[prefix + "self_attn.q_proj.weight"])
        bind_param_from_torch(layer.self_attn.q_proj.bias, hf_state[prefix + "self_attn.q_proj.bias"])
        bind_param_from_torch(layer.self_attn.k_proj.weight, hf_state[prefix + "self_attn.k_proj.weight"])
        bind_param_from_torch(layer.self_attn.v_proj.weight, hf_state[prefix + "self_attn.v_proj.weight"])
        bind_param_from_torch(layer.self_attn.v_proj.bias, hf_state[prefix + "self_attn.v_proj.bias"])
        bind_param_from_torch(layer.self_attn.out_proj.weight, hf_state[prefix + "self_attn.out_proj.weight"])
        bind_param_from_torch(layer.self_attn.out_proj.bias, hf_state[prefix + "self_attn.out_proj.bias"])
        bind_param_from_torch(layer.encoder_attn.q_proj.weight, hf_state[prefix + "encoder_attn.q_proj.weight"])
        bind_param_from_torch(layer.encoder_attn.q_proj.bias, hf_state[prefix + "encoder_attn.q_proj.bias"])
        bind_param_from_torch(layer.encoder_attn.out_proj.weight, hf_state[prefix + "encoder_attn.out_proj.weight"])
        bind_param_from_torch(layer.encoder_attn.out_proj.bias, hf_state[prefix + "encoder_attn.out_proj.bias"])
        bind_param_from_torch(layer.self_attn_layer_norm.weight, hf_state[prefix + "self_attn_layer_norm.weight"])
        bind_param_from_torch(layer.self_attn_layer_norm.bias, hf_state[prefix + "self_attn_layer_norm.bias"])
        bind_param_from_torch(layer.encoder_attn_layer_norm.weight, hf_state[prefix + "encoder_attn_layer_norm.weight"])
        bind_param_from_torch(layer.encoder_attn_layer_norm.bias, hf_state[prefix + "encoder_attn_layer_norm.bias"])
        bind_param_from_torch(layer.fc1.weight, hf_state[prefix + "fc1.weight"])
        bind_param_from_torch(layer.fc1.bias, hf_state[prefix + "fc1.bias"])
        bind_param_from_torch(layer.fc2.weight, hf_state[prefix + "fc2.weight"])
        bind_param_from_torch(layer.fc2.bias, hf_state[prefix + "fc2.bias"])
        bind_param_from_torch(layer.final_layer_norm.weight, hf_state[prefix + "final_layer_norm.weight"])
        bind_param_from_torch(layer.final_layer_norm.bias, hf_state[prefix + "final_layer_norm.bias"])


# -----------------------------------------------------------------------------
# Host-side cached-step runner
# -----------------------------------------------------------------------------
def tvm_cached_step(
    decoder_vm,
    decoder_params_tvm,
    dev,
    token_id_np,
    position_id_np,
    self_k_cache_np,
    self_v_cache_np,
    past_keep_mask_np,
    cross_k_cache_tvm,
    cross_v_cache_tvm,
):
    out = decoder_vm["forward"](
        to_tvm_tensor(token_id_np, dev),
        to_tvm_tensor(position_id_np, dev),
        to_tvm_tensor(self_k_cache_np, dev),
        to_tvm_tensor(self_v_cache_np, dev),
        to_tvm_tensor(past_keep_mask_np, dev),
        cross_k_cache_tvm,
        cross_v_cache_tvm,
        decoder_params_tvm,
    )
    logits_tvm, new_k_tvm, new_v_tvm = unwrap_vm_outputs(out)
    logits_np = unwrap_vm_output(logits_tvm).numpy().astype(np.float32)
    new_k_np = unwrap_vm_output(new_k_tvm).numpy().astype(np.float32)
    new_v_np = unwrap_vm_output(new_v_tvm).numpy().astype(np.float32)
    return logits_np, new_k_np, new_v_np


def prefill_prompt_tvm(
    prompt_ids: List[int],
    decoder_vm,
    decoder_params_tvm,
    dev,
    cross_k_cache_tvm,
    cross_v_cache_tvm,
    decoder_layers: int,
    num_heads: int,
    max_past_len: int,
    head_dim: int,
):
    self_k_cache_np = np.zeros((decoder_layers, 1, num_heads, max_past_len, head_dim), dtype=np.float32)
    self_v_cache_np = np.zeros((decoder_layers, 1, num_heads, max_past_len, head_dim), dtype=np.float32)
    last_logits_np = None
    for pos, tok in enumerate(prompt_ids):
        token_id_np = np.asarray([[tok]], dtype=np.int32)
        position_id_np = np.asarray([[pos]], dtype=np.int32)
        past_keep_mask_np = make_past_keep_mask_np(pos, max_past_len)
        logits_np, new_k_np, new_v_np = tvm_cached_step(
            decoder_vm=decoder_vm,
            decoder_params_tvm=decoder_params_tvm,
            dev=dev,
            token_id_np=token_id_np,
            position_id_np=position_id_np,
            self_k_cache_np=self_k_cache_np,
            self_v_cache_np=self_v_cache_np,
            past_keep_mask_np=past_keep_mask_np,
            cross_k_cache_tvm=cross_k_cache_tvm,
            cross_v_cache_tvm=cross_v_cache_tvm,
        )
        self_k_cache_np[:, :, :, pos : pos + 1, :] = new_k_np
        self_v_cache_np[:, :, :, pos : pos + 1, :] = new_v_np
        last_logits_np = logits_np
    return self_k_cache_np, self_v_cache_np, last_logits_np


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    dev, target = choose_device_and_target(args.device)

    processor = AutoProcessor.from_pretrained(args.model_id)
    hf_model = WhisperForConditionalGeneration.from_pretrained(args.model_id).eval()
    tokffi_tokenizer, _tokffi_tokenizer_tmpdir = build_tokffi_tokenizer(
        processor, args.tokffi_lib, args.tokffi_tokenizer_dir
    )
    special_ids = set(int(x) for x in processor.tokenizer.all_special_ids)

    audio = load_audio_from_flac(args.flac)
    waveform_fixed, valid_samples = pad_or_trim_audio(audio)

    eos_token_id = int(hf_model.config.eos_token_id)
    decoder_start_token_id = int(hf_model.config.decoder_start_token_id)
    vocab_size = int(hf_model.config.vocab_size)
    prompt_ids = get_decoder_prompt_ids(processor, decoder_start_token_id)
    max_dec_len = len(prompt_ids) + int(args.max_new_tokens)

    preprocess_model = WhisperPreprocessTVM(np.asarray(processor.feature_extractor.mel_filters, dtype=np.float32))
    encoder_model = WhisperEncoderTVM(hf_model.config)
    cross_kv_model = WhisperCrossKVCachedTVM(hf_model.config)
    decoder_model = WhisperDecoderCachedStepTVM(hf_model.config, max_dec_len=max_dec_len)

    copy_encoder_weights_from_hf(encoder_model, hf_model)
    copy_cross_kv_weights_from_hf(cross_kv_model, hf_model)
    copy_decoder_step_weights_from_hf(decoder_model, hf_model)

    preprocess_vm, _, preprocess_params_tvm = compile_nn_module_to_vm(preprocess_model, target, dev)
    encoder_vm, _, encoder_params_tvm = compile_nn_module_to_vm(encoder_model, target, dev)
    cross_kv_vm, _, cross_kv_params_tvm = compile_nn_module_to_vm(cross_kv_model, target, dev)
    decoder_vm, _, decoder_params_tvm = compile_nn_module_to_vm(decoder_model, target, dev)

    # 1) Preprocess compare
    preprocess_out = preprocess_vm["forward"](
        to_tvm_tensor(waveform_fixed, dev),
        to_tvm_tensor(valid_samples, dev),
        *preprocess_params_tvm,
    )
    input_features_tvm, feature_attention_mask_tvm = unwrap_vm_outputs(preprocess_out)
    input_features_np = unwrap_vm_output(input_features_tvm).numpy().astype(np.float32)
    feature_attention_mask_np = unwrap_vm_output(feature_attention_mask_tvm).numpy().astype(np.int32)

    hf_inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
    hf_proc_features = hf_inputs.input_features.detach().cpu().numpy().astype(np.float32)
    hf_proc_mask = hf_inputs.attention_mask.detach().cpu().numpy().astype(np.int32)
    report_diff("preprocess.features", hf_proc_features, input_features_np)
    print(f"[preprocess.mask.equal] {np.array_equal(hf_proc_mask, feature_attention_mask_np)}")

    # 2) Encoder compare
    input_features_torch = torch.from_numpy(input_features_np)
    with torch.no_grad():
        hf_encoder_hidden = hf_model.model.encoder(input_features_torch, return_dict=False)[0]

    encoder_hidden_tvm = encoder_vm["forward"](to_tvm_tensor(input_features_np, dev), encoder_params_tvm)
    encoder_hidden_np = unwrap_vm_output(encoder_hidden_tvm).numpy().astype(np.float32)
    report_diff("encoder.hidden", hf_encoder_hidden.detach().cpu().numpy().astype(np.float32), encoder_hidden_np)

    # 3) Cross-KV compare (layer 0)
    cross_cache_out = cross_kv_vm["forward"](to_tvm_tensor(encoder_hidden_np, dev), cross_kv_params_tvm)
    cross_k_cache_tvm, cross_v_cache_tvm = unwrap_vm_outputs(cross_cache_out)
    cross_k_cache_np = unwrap_vm_output(cross_k_cache_tvm).numpy().astype(np.float32)
    cross_v_cache_np = unwrap_vm_output(cross_v_cache_tvm).numpy().astype(np.float32)

    with torch.no_grad():
        hf_cross_k_l0 = hf_model.model.decoder.layers[0].encoder_attn.k_proj(hf_encoder_hidden)
        hf_cross_v_l0 = hf_model.model.decoder.layers[0].encoder_attn.v_proj(hf_encoder_hidden)
        bsz, src_len, d_model = hf_cross_k_l0.shape
        num_heads = int(hf_model.config.decoder_attention_heads)
        head_dim = d_model // num_heads
        hf_cross_k_l0 = hf_cross_k_l0.view(bsz, src_len, num_heads, head_dim).transpose(1, 2).contiguous()
        hf_cross_v_l0 = hf_cross_v_l0.view(bsz, src_len, num_heads, head_dim).transpose(1, 2).contiguous()
    report_diff("cross_k.layer0", hf_cross_k_l0.detach().cpu().numpy().astype(np.float32), cross_k_cache_np[0])
    report_diff("cross_v.layer0", hf_cross_v_l0.detach().cpu().numpy().astype(np.float32), cross_v_cache_np[0])

    # 4) HF prompt prefill reference
    prompt_torch = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        hf_prefill = hf_model.model.decoder(
            input_ids=prompt_torch,
            attention_mask=torch.ones_like(prompt_torch),
            encoder_hidden_states=hf_encoder_hidden,
            use_cache=True,
            return_dict=True,
        )
        hf_prefill_logits = hf_model.proj_out(hf_prefill.last_hidden_state)[:, -1, :]
        hf_past = hf_prefill.past_key_values

    # 5) TVM prompt prefill + logits compare
    decoder_layers = int(hf_model.config.decoder_layers)
    head_dim = int(hf_model.config.d_model) // num_heads
    max_past_len = max_dec_len - 1
    self_k_cache_np, self_v_cache_np, tvm_prefill_logits_np = prefill_prompt_tvm(
        prompt_ids=prompt_ids,
        decoder_vm=decoder_vm,
        decoder_params_tvm=decoder_params_tvm,
        dev=dev,
        cross_k_cache_tvm=cross_k_cache_tvm,
        cross_v_cache_tvm=cross_v_cache_tvm,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        max_past_len=max_past_len,
        head_dim=head_dim,
    )
    report_diff("prefill.logits", hf_prefill_logits.detach().cpu().numpy().astype(np.float32), tvm_prefill_logits_np)

    # 6) HF generate + first cached step compare
    gen_cfg = copy.deepcopy(hf_model.generation_config)
    gen_cfg.max_length = None
    if hasattr(gen_cfg, "forced_decoder_ids"):
        gen_cfg.forced_decoder_ids = None
    hf_generate_kwargs = dict(
        input_features=input_features_torch,
        attention_mask=torch.from_numpy(feature_attention_mask_np).to(torch.long),
        generation_config=gen_cfg,
        language=HF_LANGUAGE,
        task=TASK,
        max_new_tokens=int(args.max_new_tokens),
        return_dict_in_generate=True,
    )
    try:
        with torch.no_grad():
            hf_out = hf_model.generate(force_unique_generate_call=True, **hf_generate_kwargs)
    except TypeError:
        with torch.no_grad():
            hf_out = hf_model.generate(**hf_generate_kwargs)
    hf_full_ids = hf_out.sequences if hasattr(hf_out, "sequences") else hf_out
    hf_full_ids = maybe_reconstruct_full_hf_ids(hf_full_ids, prompt_ids, eos_token_id)
    hf_content_ids = strip_prefix_and_eos(hf_full_ids[0].tolist(), prompt_ids, eos_token_id)
    hf_text = decode_with_tokffi(tokffi_tokenizer, hf_content_ids, special_ids)

    if len(hf_content_ids) > 0:
        first_generated_id = int(hf_content_ids[0])
        with torch.no_grad():
            hf_first_step = hf_model.model.decoder(
                input_ids=torch.tensor([[first_generated_id]], dtype=torch.long),
                encoder_hidden_states=hf_encoder_hidden,
                past_key_values=hf_past,
                use_cache=True,
                return_dict=True,
            )
            hf_first_step_logits = hf_model.proj_out(hf_first_step.last_hidden_state)[:, -1, :]
        past_keep_mask_np = make_past_keep_mask_np(len(prompt_ids), max_past_len)
        tvm_first_step_logits_np, _, _ = tvm_cached_step(
            decoder_vm=decoder_vm,
            decoder_params_tvm=decoder_params_tvm,
            dev=dev,
            token_id_np=np.asarray([[first_generated_id]], dtype=np.int32),
            position_id_np=np.asarray([[len(prompt_ids)]], dtype=np.int32),
            self_k_cache_np=self_k_cache_np,
            self_v_cache_np=self_v_cache_np,
            past_keep_mask_np=past_keep_mask_np,
            cross_k_cache_tvm=cross_k_cache_tvm,
            cross_v_cache_tvm=cross_v_cache_tvm,
        )
        report_diff(
            "first_cached_step.logits",
            hf_first_step_logits.detach().cpu().numpy().astype(np.float32),
            tvm_first_step_logits_np,
        )

    # 7) TVM greedy decode using cached decoder-step
    tvm_generated_content_ids = []
    tvm_logits_np = tvm_prefill_logits_np
    cached_tokens = len(prompt_ids)
    for gen_idx in range(int(args.max_new_tokens)):
        next_logits = apply_whisper_suppression(tvm_logits_np[0], hf_model.generation_config, gen_idx)
        next_id = int(np.argmax(next_logits))
        if not (0 <= next_id < vocab_size):
            raise RuntimeError(f"Out-of-range token id: {next_id}")
        tvm_generated_content_ids.append(next_id)
        if next_id == eos_token_id or gen_idx == int(args.max_new_tokens) - 1:
            break
        past_keep_mask_np = make_past_keep_mask_np(cached_tokens, max_past_len)
        tvm_logits_np, new_k_np, new_v_np = tvm_cached_step(
            decoder_vm=decoder_vm,
            decoder_params_tvm=decoder_params_tvm,
            dev=dev,
            token_id_np=np.asarray([[next_id]], dtype=np.int32),
            position_id_np=np.asarray([[cached_tokens]], dtype=np.int32),
            self_k_cache_np=self_k_cache_np,
            self_v_cache_np=self_v_cache_np,
            past_keep_mask_np=past_keep_mask_np,
            cross_k_cache_tvm=cross_k_cache_tvm,
            cross_v_cache_tvm=cross_v_cache_tvm,
        )
        self_k_cache_np[:, :, :, cached_tokens : cached_tokens + 1, :] = new_k_np
        self_v_cache_np[:, :, :, cached_tokens : cached_tokens + 1, :] = new_v_np
        cached_tokens += 1

    tvm_full_ids = np.asarray([prompt_ids + tvm_generated_content_ids], dtype=np.int64)
    tvm_content_ids = strip_prefix_and_eos(tvm_full_ids[0].tolist(), prompt_ids, eos_token_id)
    tvm_text = decode_with_tokffi(tokffi_tokenizer, tvm_content_ids, special_ids)

    print(f"[HF text]  {hf_text}")
    print(f"[TVM text] {tvm_text}")
    print(f"[HF full IDs ] {hf_full_ids[0].tolist()}")
    print(f"[TVM full IDs] {tvm_full_ids[0].tolist()}")
    print(f"[HF text IDs ] {hf_content_ids}")
    print(f"[TVM text IDs] {tvm_content_ids}")
    print(f"[match text] {hf_text == tvm_text}")
    print(f"[match content ids] {hf_content_ids == tvm_content_ids}")


if __name__ == "__main__":
    main()
