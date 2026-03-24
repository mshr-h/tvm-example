import argparse
import copy
import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor, WhisperForConditionalGeneration
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

import tvm
from tvm import relax

# ----------------------------------------------------------------------
# Static-shape setup
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Whisper preprocess/encoder/cached decoder-step in tvm.relax.frontend.nn "
            "with Hugging Face weight copy and numerical comparison."
        )
    )
    parser.add_argument(
        "--flac",
        type=Path,
        required=True,
        help=(
            "Path to the input FLAC file. The script reads it, converts it to mono float32, and resamples it to 16 kHz."
        ),
    )
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device. auto prefers CUDA when available.",
    )
    return parser.parse_args()


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
    for name, param in named_params:
        if param.data is None:
            raise ValueError(f"Parameter {name} is not set.")
        params_tvm.append(param.data.copyto(target=dev))
    return vm, named_params, params_tvm


def unwrap_vm_output(x):
    while not hasattr(x, "numpy"):
        x = x[0]
    return x


def unwrap_vm_outputs(x):
    if hasattr(x, "numpy"):
        return x
    return [unwrap_vm_outputs(x[i]) for i in range(len(x))]


def to_tvm_tensor(x, dev):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return tvm.runtime.tensor(x, dev)
    if hasattr(x, "copyto") and hasattr(x, "numpy"):
        return x.copyto(target=dev)
    raise TypeError(f"Unsupported tensor input type: {type(x)}")


def report_diff(name: str, ref: np.ndarray, test: np.ndarray):
    ref = np.asarray(ref)
    test = np.asarray(test)
    diff = ref - test
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    print(f"[{name}] max_abs={max_abs:.8e} mean_abs={mean_abs:.8e} rmse={rmse:.8e}")


def maybe_reconstruct_full_hf_ids(hf_ids: torch.Tensor, prompt_ids: List[int], eos_token_id: int):
    """
    Some Transformers versions may still return content-only IDs even when
    return_dict_in_generate=True. Reconstruct the full sequence if needed so
    HF and TVM can be compared directly.
    """
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


# ----------------------------------------------------------------------
# TVM-native preprocess
# ----------------------------------------------------------------------
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

    real = real.astype(np.float32)[:, None, :]
    imag = imag.astype(np.float32)[:, None, :]
    return real, imag


class WhisperPreprocessTVM(nn.Module):
    def __init__(self, mel_filters: np.ndarray):
        super().__init__()

        mel_filters = np.asarray(mel_filters, dtype=np.float32)
        if mel_filters.shape == (N_MELS, N_FREQ):
            mel_filters = mel_filters.T
        if mel_filters.shape != (N_FREQ, N_MELS):
            raise ValueError(
                f"Expected mel_filters shape {(N_FREQ, N_MELS)} or {(N_MELS, N_FREQ)}, got {mel_filters.shape}"
            )

        real_kernel, imag_kernel = make_stft_conv_kernels()
        reflect_indices = make_reflect_indices()
        keep_frame_indices = np.arange(N_FRAMES, dtype=np.int32)
        frame_starts = np.arange(0, N_SAMPLES, HOP_LENGTH, dtype=np.int32).reshape(1, N_FRAMES)

        self.reflect_indices = Tensor.from_const(reflect_indices)
        self.keep_frame_indices = Tensor.from_const(keep_frame_indices)
        self.frame_starts = Tensor.from_const(frame_starts)

        self.real_kernel = Tensor.from_const(real_kernel)
        self.imag_kernel = Tensor.from_const(imag_kernel)
        self.mel_filters = Tensor.from_const(mel_filters)

        self.log_eps = Tensor.from_scalar(1e-10, "float32")
        self.inv_ln10 = Tensor.from_scalar(float(1.0 / math.log(10.0)), "float32")
        self.eight = Tensor.from_scalar(8.0, "float32")
        self.four = Tensor.from_scalar(4.0, "float32")

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
        feature_attention_mask = op.astype(
            op.less(self.frame_starts, valid_samples_2d),
            "int32",
        )
        return input_features, feature_attention_mask

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "waveform": nn.spec.Tensor([1, N_SAMPLES], "float32"),
                    "valid_samples": nn.spec.Tensor([1], "int32"),
                    "$": {"param_mode": "none", "effect_mode": "none"},
                }
            },
            self,
        )


# ----------------------------------------------------------------------
# TVM-native Whisper encoder
# ----------------------------------------------------------------------
class WhisperAttentionTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.scale = Tensor.from_scalar(float(self.head_dim**-0.5), "float32")

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

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ):
        if key_value_states is None:
            key_value_states = hidden_states

        query_states = self.q_proj(hidden_states)
        query_states = op.multiply(query_states, self.scale)
        key_states = self.k_proj(key_value_states)
        value_states = self.v_proj(key_value_states)

        query_states = self._reshape_qkv(query_states)
        key_states = self._reshape_qkv(key_states)
        value_states = self._reshape_qkv(value_states)

        attn_weights = op.matmul(query_states, op.permute_dims(key_states, axes=[0, 1, 3, 2]))
        if attention_mask is not None:
            attn_weights = op.add(attn_weights, attention_mask)
        attn_weights = op.softmax(attn_weights, axis=-1)

        attn_output = op.matmul(attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return attn_output


class WhisperEncoderLayerTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_attn = WhisperAttentionTVM(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.activation_fn = nn.GELU()

    def forward(self, hidden_states: Tensor):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, key_value_states=None, attention_mask=None)
        hidden_states = op.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
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

        self.conv1 = nn.Conv1D(
            in_channels=self.num_mel_bins,
            out_channels=self.d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv1D(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )
        self.embed_positions = nn.Embedding(self.max_source_positions, self.d_model)
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayerTVM(
                    embed_dim=self.d_model,
                    num_heads=self.encoder_attention_heads,
                    ffn_dim=self.encoder_ffn_dim,
                )
                for _ in range(self.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.activation_fn = nn.GELU()
        self.position_ids = Tensor.from_const(np.arange(self.max_source_positions, dtype=np.int64)[None, :])

    def forward(self, input_features: Tensor):
        hidden_states = self.conv1(input_features)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = op.permute_dims(hidden_states, axes=[0, 2, 1])

        pos_embeds = self.embed_positions(self.position_ids)
        hidden_states = op.add(hidden_states, pos_embeds)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {"forward": {"input_features": nn.spec.Tensor([1, N_MELS, N_FRAMES], "float32")}},
            self,
        )


# ----------------------------------------------------------------------
# TVM-native cached decoder-step
# ----------------------------------------------------------------------
class WhisperCrossKVLayerTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, enc_len: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.enc_len = enc_len

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, encoder_hidden_states: Tensor):
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)

        k = op.reshape(k, [1, self.enc_len, self.num_heads, self.head_dim])
        v = op.reshape(v, [1, self.enc_len, self.num_heads, self.head_dim])
        k = op.permute_dims(k, axes=[0, 2, 1, 3])
        v = op.permute_dims(v, axes=[0, 2, 1, 3])
        return k, v


class WhisperCrossKVTVM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = int(config.decoder_layers)
        self.d_model = int(config.d_model)
        self.num_heads = int(config.decoder_attention_heads)
        self.enc_len = int(config.max_source_positions)
        self.head_dim = self.d_model // self.num_heads

        self.layers = nn.ModuleList(
            [
                WhisperCrossKVLayerTVM(
                    embed_dim=self.d_model,
                    num_heads=self.num_heads,
                    enc_len=self.enc_len,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, encoder_hidden_states: Tensor):
        all_k = []
        all_v = []
        for layer in self.layers:
            k, v = layer(encoder_hidden_states)
            all_k.append(op.reshape(k, [1, 1, self.num_heads, self.enc_len, self.head_dim]))
            all_v.append(op.reshape(v, [1, 1, self.num_heads, self.enc_len, self.head_dim]))
        cross_k = nn.concat(all_k, dim=0)
        cross_v = nn.concat(all_v, dim=0)
        return cross_k, cross_v

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {"forward": {"encoder_hidden_states": nn.spec.Tensor([1, self.enc_len, self.d_model], "float32")}},
            self,
        )


class WhisperSelfAttentionStepTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_cache_len: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.scale = Tensor.from_scalar(float(self.head_dim**-0.5), "float32")
        self.positions_4d = Tensor.from_const(
            np.arange(self.max_cache_len, dtype=np.int64).reshape(1, 1, self.max_cache_len, 1)
        )
        self.positions_mask = Tensor.from_const(
            np.arange(self.max_cache_len, dtype=np.int64).reshape(1, 1, 1, self.max_cache_len)
        )
        self.zero_f32 = Tensor.from_scalar(0.0, "float32")
        self.one_f32 = Tensor.from_scalar(1.0, "float32")
        self.neg_inf = Tensor.from_scalar(-1e9, "float32")

    def _reshape_qkv(self, x: Tensor):
        x = op.reshape(x, [1, 1, self.num_heads, self.head_dim])
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return x

    def _merge_heads(self, x: Tensor):
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        x = op.reshape(x, [1, 1, self.embed_dim])
        return x

    def _update_cache(self, cache: Tensor, new_value: Tensor, position_ids: Tensor):
        pos = op.reshape(position_ids, [1, 1, 1, 1])
        pos_mask_bool = op.equal(self.positions_4d, pos)
        pos_mask = op.astype(pos_mask_bool, "float32")
        pos_mask = op.broadcast_to(pos_mask, [1, self.num_heads, self.max_cache_len, self.head_dim])
        new_full = op.broadcast_to(new_value, [1, self.num_heads, self.max_cache_len, self.head_dim])
        keep_mask = op.subtract(self.one_f32, pos_mask)
        return op.add(op.multiply(cache, keep_mask), op.multiply(new_full, pos_mask))

    def _make_self_mask(self, position_ids: Tensor):
        pos = op.reshape(position_ids, [1, 1, 1, 1])
        valid = op.less_equal(self.positions_mask, pos)
        zeros = op.broadcast_to(self.zero_f32, [1, 1, 1, self.max_cache_len])
        negs = op.broadcast_to(self.neg_inf, [1, 1, 1, self.max_cache_len])
        return op.where(valid, zeros, negs)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
    ):
        q = self.q_proj(hidden_states)
        q = op.multiply(q, self.scale)
        q = self._reshape_qkv(q)

        k_new = self.k_proj(hidden_states)
        v_new = self.v_proj(hidden_states)
        k_new = self._reshape_qkv(k_new)
        v_new = self._reshape_qkv(v_new)

        new_k_cache = self._update_cache(k_cache, k_new, position_ids)
        new_v_cache = self._update_cache(v_cache, v_new, position_ids)

        attn_weights = op.matmul(q, op.permute_dims(new_k_cache, axes=[0, 1, 3, 2]))
        attn_weights = op.add(attn_weights, self._make_self_mask(position_ids))
        attn_weights = op.softmax(attn_weights, axis=-1)

        attn_output = op.matmul(attn_weights, new_v_cache)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return attn_output, new_k_cache, new_v_cache


class WhisperCrossAttentionStepTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, enc_len: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.enc_len = enc_len

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.scale = Tensor.from_scalar(float(self.head_dim**-0.5), "float32")

    def _reshape_q(self, x: Tensor):
        x = op.reshape(x, [1, 1, self.num_heads, self.head_dim])
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return x

    def _merge_heads(self, x: Tensor):
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        x = op.reshape(x, [1, 1, self.embed_dim])
        return x

    def forward(self, hidden_states: Tensor, cross_k_cache: Tensor, cross_v_cache: Tensor):
        q = self.q_proj(hidden_states)
        q = op.multiply(q, self.scale)
        q = self._reshape_q(q)

        attn_weights = op.matmul(q, op.permute_dims(cross_k_cache, axes=[0, 1, 3, 2]))
        attn_weights = op.softmax(attn_weights, axis=-1)
        attn_output = op.matmul(attn_weights, cross_v_cache)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return attn_output


class WhisperDecoderLayerStepTVM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        enc_len: int,
        max_cache_len: int,
    ):
        super().__init__()
        self.self_attn = WhisperSelfAttentionStepTVM(embed_dim, num_heads, max_cache_len)
        self.encoder_attn = WhisperCrossAttentionStepTVM(embed_dim, num_heads, enc_len)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.activation_fn = nn.GELU()

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k_cache: Tensor,
        cross_v_cache: Tensor,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        attn_out, new_self_k, new_self_v = self.self_attn(
            hidden_states,
            position_ids,
            self_k_cache,
            self_v_cache,
        )
        hidden_states = op.add(residual, attn_out)

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        cross_attn_out = self.encoder_attn(hidden_states, cross_k_cache, cross_v_cache)
        hidden_states = op.add(residual, cross_attn_out)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = op.add(residual, hidden_states)
        return hidden_states, new_self_k, new_self_v


class WhisperDecoderStepLMHeadTVM(nn.Module):
    def __init__(self, config, max_dec_len: int):
        super().__init__()
        self.vocab_size = int(config.vocab_size)
        self.d_model = int(config.d_model)
        self.max_target_positions = int(config.max_target_positions)
        self.max_source_positions = int(config.max_source_positions)
        self.decoder_layers = int(config.decoder_layers)
        self.decoder_attention_heads = int(config.decoder_attention_heads)
        self.decoder_ffn_dim = int(config.decoder_ffn_dim)
        self.max_dec_len = int(max_dec_len)
        self.head_dim = self.d_model // self.decoder_attention_heads

        if self.max_dec_len > self.max_target_positions:
            raise ValueError(
                f"max_dec_len={self.max_dec_len} exceeds config.max_target_positions={self.max_target_positions}"
            )

        self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)
        self.embed_positions = nn.Embedding(self.max_target_positions, self.d_model)
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayerStepTVM(
                    embed_dim=self.d_model,
                    num_heads=self.decoder_attention_heads,
                    ffn_dim=self.decoder_ffn_dim,
                    enc_len=self.max_source_positions,
                    max_cache_len=self.max_dec_len,
                )
                for _ in range(self.decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.proj_out = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def forward(
        self,
        input_token_ids: Tensor,
        position_ids: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k_cache: Tensor,
        cross_v_cache: Tensor,
    ):
        hidden_states = self.embed_tokens(input_token_ids)
        pos_embeds = self.embed_positions(position_ids)
        hidden_states = op.add(hidden_states, pos_embeds)

        self_k_layers = nn.split(self_k_cache, self.decoder_layers, axis=0)
        self_v_layers = nn.split(self_v_cache, self.decoder_layers, axis=0)
        cross_k_layers = nn.split(cross_k_cache, self.decoder_layers, axis=0)
        cross_v_layers = nn.split(cross_v_cache, self.decoder_layers, axis=0)

        new_self_k = []
        new_self_v = []

        for layer_idx, layer in enumerate(self.layers):
            layer_self_k = op.reshape(
                self_k_layers[layer_idx],
                [1, self.decoder_attention_heads, self.max_dec_len, self.head_dim],
            )
            layer_self_v = op.reshape(
                self_v_layers[layer_idx],
                [1, self.decoder_attention_heads, self.max_dec_len, self.head_dim],
            )
            layer_cross_k = op.reshape(
                cross_k_layers[layer_idx],
                [
                    1,
                    self.decoder_attention_heads,
                    self.max_source_positions,
                    self.head_dim,
                ],
            )
            layer_cross_v = op.reshape(
                cross_v_layers[layer_idx],
                [
                    1,
                    self.decoder_attention_heads,
                    self.max_source_positions,
                    self.head_dim,
                ],
            )

            hidden_states, layer_new_k, layer_new_v = layer(
                hidden_states,
                position_ids,
                layer_self_k,
                layer_self_v,
                layer_cross_k,
                layer_cross_v,
            )

            new_self_k.append(
                op.reshape(
                    layer_new_k,
                    [
                        1,
                        1,
                        self.decoder_attention_heads,
                        self.max_dec_len,
                        self.head_dim,
                    ],
                )
            )
            new_self_v.append(
                op.reshape(
                    layer_new_v,
                    [
                        1,
                        1,
                        self.decoder_attention_heads,
                        self.max_dec_len,
                        self.head_dim,
                    ],
                )
            )

        hidden_states = self.layer_norm(hidden_states)
        logits = self.proj_out(hidden_states)

        new_self_k_cache = nn.concat(new_self_k, dim=0)
        new_self_v_cache = nn.concat(new_self_v, dim=0)
        return logits, new_self_k_cache, new_self_v_cache

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "input_token_ids": nn.spec.Tensor([1, 1], "int64"),
                    "position_ids": nn.spec.Tensor([1, 1], "int64"),
                    "self_k_cache": nn.spec.Tensor(
                        [
                            self.decoder_layers,
                            1,
                            self.decoder_attention_heads,
                            self.max_dec_len,
                            self.head_dim,
                        ],
                        "float32",
                    ),
                    "self_v_cache": nn.spec.Tensor(
                        [
                            self.decoder_layers,
                            1,
                            self.decoder_attention_heads,
                            self.max_dec_len,
                            self.head_dim,
                        ],
                        "float32",
                    ),
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
                }
            },
            self,
        )


# ----------------------------------------------------------------------
# Weight copy
# ----------------------------------------------------------------------
def set_param_from_hf(param: nn.Parameter, tensor: torch.Tensor):
    array = tensor.detach().cpu().numpy()
    if str(array.dtype) != param.dtype:
        array = array.astype(param.dtype)
    param.data = array


def copy_encoder_weights_from_hf(tvm_encoder: WhisperEncoderTVM, hf_model: WhisperForConditionalGeneration):
    hf_state = hf_model.state_dict()
    tvm_state = tvm_encoder.state_dict()
    for name, param in tvm_state.items():
        hf_name = "model.encoder." + name
        if hf_name not in hf_state:
            raise KeyError(f"Missing HF encoder weight: {hf_name}")
        set_param_from_hf(param, hf_state[hf_name])


def copy_cross_kv_weights_from_hf(tvm_cross_kv: WhisperCrossKVTVM, hf_model: WhisperForConditionalGeneration):
    hf_state = hf_model.state_dict()
    tvm_state = tvm_cross_kv.state_dict()
    for name, param in tvm_state.items():
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "layers":
            raise KeyError(f"Unexpected cross-kv state name: {name}")
        layer_idx = parts[1]
        local_name = ".".join(parts[2:])
        hf_name = f"model.decoder.layers.{layer_idx}.encoder_attn.{local_name}"
        if hf_name not in hf_state:
            raise KeyError(f"Missing HF cross-kv weight: {hf_name}")
        set_param_from_hf(param, hf_state[hf_name])


def copy_decoder_step_weights_from_hf(
    tvm_decoder: WhisperDecoderStepLMHeadTVM,
    hf_model: WhisperForConditionalGeneration,
):
    hf_state = hf_model.state_dict()
    tvm_state = tvm_decoder.state_dict()
    for name, param in tvm_state.items():
        if name.startswith("proj_out."):
            hf_name = name
        else:
            hf_name = "model.decoder." + name
        if hf_name not in hf_state:
            raise KeyError(f"Missing HF decoder-step weight: {hf_name}")
        set_param_from_hf(param, hf_state[hf_name])


# ----------------------------------------------------------------------
# HF reference helpers
# ----------------------------------------------------------------------
def collect_hf_cross_kv(hf_model: WhisperForConditionalGeneration, encoder_hidden_states: torch.Tensor):
    all_k = []
    all_v = []
    with torch.no_grad():
        for layer in hf_model.model.decoder.layers:
            attn = layer.encoder_attn
            bsz, src_len, _ = encoder_hidden_states.shape

            k = attn.k_proj(encoder_hidden_states)
            v = attn.v_proj(encoder_hidden_states)

            k = k.view(bsz, src_len, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
            v = v.view(bsz, src_len, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()

            all_k.append(k)
            all_v.append(v)

    return torch.stack(all_k, dim=0), torch.stack(all_v, dim=0)


def run_hf_cached_decode(
    hf_model: WhisperForConditionalGeneration,
    processor,
    encoder_hidden_states: torch.Tensor,
    prompt_ids: List[int],
    max_new_tokens: int,
):
    hf_past = None
    last_logits_np = None

    with torch.no_grad():
        for pos, tok in enumerate(prompt_ids):
            token = torch.tensor([[tok]], dtype=torch.long)
            position_ids = torch.tensor([[pos]], dtype=torch.long)
            out = hf_model.model.decoder(
                input_ids=token,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=hf_past,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            hf_past = out.past_key_values
            logits = hf_model.proj_out(out.last_hidden_state)
            last_logits_np = logits.detach().cpu().numpy().astype(np.float32)

    if last_logits_np is None:
        raise RuntimeError("HF cached decode prefill produced no logits.")

    prefill_logits_np = last_logits_np.copy()
    generated = list(prompt_ids)

    for gen_idx in range(max_new_tokens):
        next_logits = apply_whisper_suppression(
            last_logits_np[0, 0],
            hf_model.generation_config,
            gen_idx,
        )
        next_id = int(np.argmax(next_logits))
        generated.append(next_id)

        if next_id == int(hf_model.config.eos_token_id):
            break

        position = len(generated) - 1
        with torch.no_grad():
            token = torch.tensor([[next_id]], dtype=torch.long)
            position_ids = torch.tensor([[position]], dtype=torch.long)
            out = hf_model.model.decoder(
                input_ids=token,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=hf_past,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            hf_past = out.past_key_values
            logits = hf_model.proj_out(out.last_hidden_state)
            last_logits_np = logits.detach().cpu().numpy().astype(np.float32)

    generated_tensor = torch.tensor([generated], dtype=torch.long)
    text = processor.batch_decode(generated_tensor, skip_special_tokens=True)[0]
    return generated_tensor, text, prefill_logits_np


def run_tvm_cached_decode(
    step_vm,
    step_params_tvm,
    dev,
    prompt_ids: List[int],
    max_new_tokens: int,
    num_layers: int,
    num_heads: int,
    max_dec_len: int,
    head_dim: int,
    vocab_size: int,
    eos_token_id: int,
    generation_config,
    cross_k_cache_tvm,
    cross_v_cache_tvm,
):
    self_k_cache_np = np.zeros((num_layers, 1, num_heads, max_dec_len, head_dim), dtype=np.float32)
    self_v_cache_np = np.zeros((num_layers, 1, num_heads, max_dec_len, head_dim), dtype=np.float32)

    self_k_cache_tvm = to_tvm_tensor(self_k_cache_np, dev)
    self_v_cache_tvm = to_tvm_tensor(self_v_cache_np, dev)
    last_logits_tvm = None

    for pos, tok in enumerate(prompt_ids):
        tok_tvm = to_tvm_tensor(np.array([[tok]], dtype=np.int64), dev)
        pos_tvm = to_tvm_tensor(np.array([[pos]], dtype=np.int64), dev)
        step_out = step_vm["forward"](
            tok_tvm,
            pos_tvm,
            self_k_cache_tvm,
            self_v_cache_tvm,
            cross_k_cache_tvm,
            cross_v_cache_tvm,
            *step_params_tvm,
        )
        last_logits_tvm, self_k_cache_tvm, self_v_cache_tvm = unwrap_vm_outputs(step_out)

    if last_logits_tvm is None:
        raise RuntimeError("TVM cached decode prefill produced no logits.")

    prefill_logits_np = unwrap_vm_output(last_logits_tvm).numpy().astype(np.float32)
    generated = list(prompt_ids)

    for gen_idx in range(max_new_tokens):
        next_logits = apply_whisper_suppression(
            prefill_logits_np[0, 0] if gen_idx == 0 else unwrap_vm_output(last_logits_tvm).numpy()[0, 0],
            generation_config,
            gen_idx,
        )
        next_id = int(np.argmax(next_logits))
        if not (0 <= next_id < vocab_size):
            raise RuntimeError(f"Out-of-range token id: {next_id}")

        generated.append(next_id)
        if next_id == eos_token_id:
            break

        position = len(generated) - 1
        if position >= max_dec_len:
            raise RuntimeError(f"Decoder length overflow: position={position}, max_dec_len={max_dec_len}")

        tok_tvm = to_tvm_tensor(np.array([[next_id]], dtype=np.int64), dev)
        pos_tvm = to_tvm_tensor(np.array([[position]], dtype=np.int64), dev)
        step_out = step_vm["forward"](
            tok_tvm,
            pos_tvm,
            self_k_cache_tvm,
            self_v_cache_tvm,
            cross_k_cache_tvm,
            cross_v_cache_tvm,
            *step_params_tvm,
        )
        last_logits_tvm, self_k_cache_tvm, self_v_cache_tvm = unwrap_vm_outputs(step_out)

    generated_np = np.asarray([generated], dtype=np.int64)
    return generated_np, prefill_logits_np


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    dev, target = choose_device_and_target(args.device)

    print(f"[device] {dev}")
    print(f"[target] {target}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    hf_model = WhisperForConditionalGeneration.from_pretrained(args.model_id).eval()

    audio = load_audio_from_flac(args.flac)
    waveform_fixed, valid_samples = pad_or_trim_audio(audio)

    eos_token_id = int(hf_model.config.eos_token_id)
    decoder_start_token_id = int(hf_model.config.decoder_start_token_id)
    vocab_size = int(hf_model.config.vocab_size)
    num_layers = int(hf_model.config.decoder_layers)
    num_heads = int(hf_model.config.decoder_attention_heads)
    head_dim = int(hf_model.config.d_model // hf_model.config.decoder_attention_heads)
    enc_len = int(hf_model.config.max_source_positions)

    prompt_ids = get_decoder_prompt_ids(processor, decoder_start_token_id)
    max_dec_len = len(prompt_ids) + int(args.max_new_tokens)

    preprocess_model = WhisperPreprocessTVM(np.asarray(processor.feature_extractor.mel_filters, dtype=np.float32))
    encoder_model = WhisperEncoderTVM(hf_model.config)
    cross_kv_model = WhisperCrossKVTVM(hf_model.config)
    decoder_step_model = WhisperDecoderStepLMHeadTVM(hf_model.config, max_dec_len=max_dec_len)

    copy_encoder_weights_from_hf(encoder_model, hf_model)
    copy_cross_kv_weights_from_hf(cross_kv_model, hf_model)
    copy_decoder_step_weights_from_hf(decoder_step_model, hf_model)

    preprocess_vm, _, preprocess_params_tvm = compile_nn_module_to_vm(preprocess_model, target, dev)
    encoder_vm, _, encoder_params_tvm = compile_nn_module_to_vm(encoder_model, target, dev)
    cross_kv_vm, _, cross_kv_params_tvm = compile_nn_module_to_vm(cross_kv_model, target, dev)
    decoder_step_vm, _, decoder_step_params_tvm = compile_nn_module_to_vm(decoder_step_model, target, dev)

    # --------------------------------------------------------------
    # 1) Preprocess compare
    # --------------------------------------------------------------
    waveform_tvm = to_tvm_tensor(waveform_fixed, dev)
    valid_samples_tvm = to_tvm_tensor(valid_samples, dev)
    preprocess_out = preprocess_vm["forward"](
        waveform_tvm,
        valid_samples_tvm,
        *preprocess_params_tvm,
    )
    input_features_tvm, attention_mask_tvm = unwrap_vm_outputs(preprocess_out)
    input_features_np = unwrap_vm_output(input_features_tvm).numpy().astype(np.float32)
    attention_mask_np = unwrap_vm_output(attention_mask_tvm).numpy().astype(np.int32)

    hf_inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )
    hf_proc_features = hf_inputs.input_features.detach().cpu().numpy().astype(np.float32)
    hf_proc_mask = hf_inputs.attention_mask.detach().cpu().numpy().astype(np.int32)

    report_diff("preprocess features", hf_proc_features, input_features_np)
    print(f"[preprocess mask equal] {np.array_equal(hf_proc_mask, attention_mask_np)}")

    # --------------------------------------------------------------
    # 2) Encoder compare
    # --------------------------------------------------------------
    input_features_torch = torch.from_numpy(input_features_np)
    with torch.no_grad():
        hf_encoder_hidden = hf_model.model.encoder(
            input_features_torch,
            return_dict=False,
        )[0]

    encoder_hidden_tvm = encoder_vm["forward"](
        to_tvm_tensor(input_features_np, dev),
        *encoder_params_tvm,
    )
    encoder_hidden_np = unwrap_vm_output(encoder_hidden_tvm).numpy().astype(np.float32)

    report_diff(
        "encoder hidden",
        hf_encoder_hidden.detach().cpu().numpy().astype(np.float32),
        encoder_hidden_np,
    )

    # --------------------------------------------------------------
    # 3) Cross K/V compare (isolated on HF encoder hidden)
    # --------------------------------------------------------------
    hf_cross_k, hf_cross_v = collect_hf_cross_kv(hf_model, hf_encoder_hidden)

    tvm_cross_out_on_hf_encoder = cross_kv_vm["forward"](
        to_tvm_tensor(hf_encoder_hidden, dev),
        *cross_kv_params_tvm,
    )
    tvm_cross_k_on_hf, tvm_cross_v_on_hf = unwrap_vm_outputs(tvm_cross_out_on_hf_encoder)
    tvm_cross_k_on_hf_np = unwrap_vm_output(tvm_cross_k_on_hf).numpy().astype(np.float32)
    tvm_cross_v_on_hf_np = unwrap_vm_output(tvm_cross_v_on_hf).numpy().astype(np.float32)

    report_diff(
        "cross_k_cache (HF encoder hidden)",
        hf_cross_k.detach().cpu().numpy().astype(np.float32),
        tvm_cross_k_on_hf_np,
    )
    report_diff(
        "cross_v_cache (HF encoder hidden)",
        hf_cross_v.detach().cpu().numpy().astype(np.float32),
        tvm_cross_v_on_hf_np,
    )

    # --------------------------------------------------------------
    # 4) HF generate() reference using TVM-preprocessed features
    # --------------------------------------------------------------
    gen_cfg = copy.deepcopy(hf_model.generation_config)
    gen_cfg.max_length = None
    if hasattr(gen_cfg, "forced_decoder_ids"):
        gen_cfg.forced_decoder_ids = None

    hf_generate_kwargs = dict(
        input_features=input_features_torch,
        attention_mask=torch.from_numpy(attention_mask_np).to(torch.long),
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

    hf_generate_full_ids = hf_out.sequences if hasattr(hf_out, "sequences") else hf_out
    hf_generate_full_ids = maybe_reconstruct_full_hf_ids(
        hf_generate_full_ids,
        prompt_ids,
        eos_token_id,
    )
    hf_generate_text = processor.batch_decode(hf_generate_full_ids, skip_special_tokens=True)[0]

    # --------------------------------------------------------------
    # 5) HF manual cached-step decode reference
    # --------------------------------------------------------------
    hf_manual_ids, hf_manual_text, hf_prefill_logits_np = run_hf_cached_decode(
        hf_model,
        processor,
        hf_encoder_hidden,
        prompt_ids,
        int(args.max_new_tokens),
    )

    # --------------------------------------------------------------
    # 6) TVM cached-step decode (isolated: HF encoder hidden -> TVM cross cache -> TVM decode)
    # --------------------------------------------------------------
    tvm_isolated_ids_np, tvm_prefill_logits_np = run_tvm_cached_decode(
        step_vm=decoder_step_vm,
        step_params_tvm=decoder_step_params_tvm,
        dev=dev,
        prompt_ids=prompt_ids,
        max_new_tokens=int(args.max_new_tokens),
        num_layers=num_layers,
        num_heads=num_heads,
        max_dec_len=max_dec_len,
        head_dim=head_dim,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        generation_config=hf_model.generation_config,
        cross_k_cache_tvm=tvm_cross_k_on_hf,
        cross_v_cache_tvm=tvm_cross_v_on_hf,
    )
    tvm_isolated_text = processor.batch_decode(torch.from_numpy(tvm_isolated_ids_np), skip_special_tokens=True)[0]

    report_diff(
        "decoder-step prefill logits (HF encoder hidden)",
        hf_prefill_logits_np,
        tvm_prefill_logits_np,
    )

    # --------------------------------------------------------------
    # 7) TVM cached-step full pipeline decode (TVM encoder hidden -> TVM cross cache -> TVM decode)
    # --------------------------------------------------------------
    encoder_hidden_tvm_runtime = to_tvm_tensor(encoder_hidden_np, dev)
    tvm_cross_out_full = cross_kv_vm["forward"](
        encoder_hidden_tvm_runtime,
        *cross_kv_params_tvm,
    )
    tvm_cross_k_full, tvm_cross_v_full = unwrap_vm_outputs(tvm_cross_out_full)

    tvm_full_ids_np, _ = run_tvm_cached_decode(
        step_vm=decoder_step_vm,
        step_params_tvm=decoder_step_params_tvm,
        dev=dev,
        prompt_ids=prompt_ids,
        max_new_tokens=int(args.max_new_tokens),
        num_layers=num_layers,
        num_heads=num_heads,
        max_dec_len=max_dec_len,
        head_dim=head_dim,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        generation_config=hf_model.generation_config,
        cross_k_cache_tvm=tvm_cross_k_full,
        cross_v_cache_tvm=tvm_cross_v_full,
    )
    tvm_full_text = processor.batch_decode(torch.from_numpy(tvm_full_ids_np), skip_special_tokens=True)[0]

    # --------------------------------------------------------------
    # 8) Final comparisons
    # --------------------------------------------------------------
    hf_generate_content_ids = strip_prefix_and_eos(hf_generate_full_ids[0].tolist(), prompt_ids, eos_token_id)
    hf_manual_content_ids = strip_prefix_and_eos(hf_manual_ids[0].tolist(), prompt_ids, eos_token_id)
    tvm_isolated_content_ids = strip_prefix_and_eos(tvm_isolated_ids_np[0].tolist(), prompt_ids, eos_token_id)
    tvm_full_content_ids = strip_prefix_and_eos(tvm_full_ids_np[0].tolist(), prompt_ids, eos_token_id)

    print(f"[HF generate text] {hf_generate_text}")
    print(f"[HF manual text ] {hf_manual_text}")
    print(f"[TVM isolated  ] {tvm_isolated_text}")
    print(f"[TVM full      ] {tvm_full_text}")
    print(f"[HF generate full IDs] {hf_generate_full_ids[0].tolist()}")
    print(f"[HF manual full IDs ] {hf_manual_ids[0].tolist()}")
    print(f"[TVM isolated IDs  ] {tvm_isolated_ids_np[0].tolist()}")
    print(f"[TVM full IDs      ] {tvm_full_ids_np[0].tolist()}")
    print(f"[HF generate text IDs] {hf_generate_content_ids}")
    print(f"[HF manual text IDs ] {hf_manual_content_ids}")
    print(f"[TVM isolated IDs   ] {tvm_isolated_content_ids}")
    print(f"[TVM full text IDs  ] {tvm_full_content_ids}")
    print(f"[match manual full (isolated)] {hf_manual_ids[0].tolist() == tvm_isolated_ids_np[0].tolist()}")
    print(f"[match manual content (isolated)] {hf_manual_content_ids == tvm_isolated_content_ids}")
    print(f"[match manual full (full pipeline)] {hf_manual_ids[0].tolist() == tvm_full_ids_np[0].tolist()}")
    print(f"[match manual content (full pipeline)] {hf_manual_content_ids == tvm_full_content_ids}")
    print(f"[match generate vs TVM text] {hf_generate_text == tvm_full_text}")

    print(
        f"[cache shapes] self=({num_layers}, 1, {num_heads}, {max_dec_len}, {head_dim}) cross=({num_layers}, 1, {num_heads}, {enc_len}, {head_dim})"
    )


if __name__ == "__main__":
    main()
