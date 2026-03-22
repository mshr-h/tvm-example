import argparse
import copy
import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
import soundfile as sf
import torch
import tvm
from scipy.signal import resample_poly
from transformers import AutoProcessor, WhisperForConditionalGeneration
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

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
        description="Whisper preprocess/encoder/decoder in tvm.relax.frontend.nn "
        "with Hugging Face weight copy and numerical comparison."
    )
    parser.add_argument(
        "--flac",
        type=Path,
        required=True,
        help="Path to the input FLAC file. The script reads it, converts to mono float32, and resamples to 16 kHz.",
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


def maybe_reconstruct_full_hf_ids(
    hf_ids: torch.Tensor, prompt_ids: List[int], eos_token_id: int
):
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
    """
    A small host-side approximation of Whisper's default suppression behavior so
    greedy TVM decoding matches HF generate() more closely.
    """
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
def make_reflect_indices(
    n_samples: int = N_SAMPLES, pad: int = REFLECT_PAD
) -> np.ndarray:
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

    real = real.astype(np.float32)[:, None, :]  # [F, 1, N_FFT]
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
        frame_starts = np.arange(0, N_SAMPLES, HOP_LENGTH, dtype=np.int32).reshape(
            1, N_FRAMES
        )

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
        # waveform: [1, 480000] float32
        # valid_samples: [1] int32
        x = op.take(waveform, self.reflect_indices, axis=1)  # [1, 480400]
        x = op.unsqueeze(x, dim=1)  # [1, 1, 480400]

        real = op.conv1d(x, self.real_kernel, stride=HOP_LENGTH, padding=0)
        imag = op.conv1d(x, self.imag_kernel, stride=HOP_LENGTH, padding=0)
        power = op.add(op.square(real), op.square(imag))  # [1, 201, 3001]
        power = op.take(power, self.keep_frame_indices, axis=2)  # [1, 201, 3000]

        power_t = op.permute_dims(power, axes=[0, 2, 1])  # [1, 3000, 201]
        mel = op.matmul(power_t, self.mel_filters)  # [1, 3000, 80]
        mel = op.permute_dims(mel, axes=[0, 2, 1])  # [1, 80, 3000]

        log_spec = op.log(op.maximum(mel, self.log_eps))
        log_spec = op.multiply(log_spec, self.inv_ln10)  # ln -> log10

        max_val = op.max(log_spec, axis=[1, 2], keepdims=True)
        log_spec = op.maximum(log_spec, op.subtract(max_val, self.eight))
        input_features = op.divide(op.add(log_spec, self.four), self.four)

        valid_samples_2d = op.broadcast_to(
            op.reshape(valid_samples, [1, 1]), [1, N_FRAMES]
        )
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
                    "$": {
                        "param_mode": "none",
                        "effect_mode": "none",
                    },
                }
            },
            self,
        )


# ----------------------------------------------------------------------
# TVM-native Whisper encoder / decoder
# ----------------------------------------------------------------------
class WhisperAttentionTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, is_cross_attention: bool):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_cross_attention = is_cross_attention

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

        attn_weights = op.matmul(
            query_states,
            op.permute_dims(key_states, axes=[0, 1, 3, 2]),
        )

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
        self.self_attn = WhisperAttentionTVM(
            embed_dim, num_heads, is_cross_attention=False
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.activation_fn = nn.GELU()

    def forward(self, hidden_states: Tensor):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, key_value_states=None, attention_mask=None
        )
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
        self.position_ids = Tensor.from_const(
            np.arange(self.max_source_positions, dtype=np.int64)[None, :]
        )

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
            {
                "forward": {
                    "input_features": nn.spec.Tensor([1, N_MELS, N_FRAMES], "float32"),
                }
            },
            self,
        )


class WhisperDecoderLayerTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_attn = WhisperAttentionTVM(
            embed_dim, num_heads, is_cross_attention=False
        )
        self.encoder_attn = WhisperAttentionTVM(
            embed_dim, num_heads, is_cross_attention=True
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.activation_fn = nn.GELU()

    def forward(
        self,
        hidden_states: Tensor,
        self_attention_mask: Tensor,
        encoder_hidden_states: Tensor,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            key_value_states=None,
            attention_mask=self_attention_mask,
        )
        hidden_states = op.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=None,
        )
        hidden_states = op.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = op.add(residual, hidden_states)
        return hidden_states


class WhisperDecoderLMHeadTVM(nn.Module):
    def __init__(self, config, max_dec_len: int):
        super().__init__()
        self.vocab_size = int(config.vocab_size)
        self.d_model = int(config.d_model)
        self.max_target_positions = int(config.max_target_positions)
        self.decoder_layers = int(config.decoder_layers)
        self.decoder_attention_heads = int(config.decoder_attention_heads)
        self.decoder_ffn_dim = int(config.decoder_ffn_dim)
        self.max_source_positions = int(config.max_source_positions)
        self.max_dec_len = int(max_dec_len)

        if self.max_dec_len > self.max_target_positions:
            raise ValueError(
                f"max_dec_len={self.max_dec_len} exceeds config.max_target_positions={self.max_target_positions}"
            )

        self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)
        self.embed_positions = nn.Embedding(self.max_target_positions, self.d_model)
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayerTVM(
                    embed_dim=self.d_model,
                    num_heads=self.decoder_attention_heads,
                    ffn_dim=self.decoder_ffn_dim,
                )
                for _ in range(self.decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.proj_out = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self.position_ids = Tensor.from_const(
            np.arange(self.max_dec_len, dtype=np.int64)[None, :]
        )

        causal_keep = np.tril(
            np.ones((1, 1, self.max_dec_len, self.max_dec_len), dtype=np.int32)
        )
        self.causal_keep = Tensor.from_const(causal_keep)
        self.zero_i32 = Tensor.from_scalar(0, "int32")
        self.zero_f32 = Tensor.from_scalar(0.0, "float32")
        self.neg_inf = Tensor.from_scalar(-1e9, "float32")

    def _make_self_attention_mask(self, decoder_attention_mask: Tensor):
        # decoder_attention_mask: [1, T] with 1 for valid tokens, 0 for pad
        key_keep = op.reshape(decoder_attention_mask, [1, 1, 1, self.max_dec_len])
        allowed = op.multiply(self.causal_keep, key_keep)
        allowed = op.greater(allowed, self.zero_i32)
        return op.where(allowed, self.zero_f32, self.neg_inf)

    def forward(
        self,
        decoder_input_ids: Tensor,
        decoder_attention_mask: Tensor,
        encoder_hidden_states: Tensor,
    ):
        hidden_states = self.embed_tokens(decoder_input_ids)
        pos_embeds = self.embed_positions(self.position_ids)
        hidden_states = op.add(hidden_states, pos_embeds)

        self_attention_mask = self._make_self_attention_mask(decoder_attention_mask)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                self_attention_mask=self_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.layer_norm(hidden_states)
        logits = self.proj_out(hidden_states)
        return logits

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "decoder_input_ids": nn.spec.Tensor([1, self.max_dec_len], "int64"),
                    "decoder_attention_mask": nn.spec.Tensor(
                        [1, self.max_dec_len], "int32"
                    ),
                    "encoder_hidden_states": nn.spec.Tensor(
                        [1, self.max_source_positions, self.d_model], "float32"
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


def copy_encoder_weights_from_hf(
    tvm_encoder: WhisperEncoderTVM, hf_model: WhisperForConditionalGeneration
):
    hf_state = hf_model.state_dict()
    tvm_state = tvm_encoder.state_dict()
    for name, param in tvm_state.items():
        hf_name = "model.encoder." + name
        if hf_name not in hf_state:
            raise KeyError(f"Missing HF encoder weight: {hf_name}")
        set_param_from_hf(param, hf_state[hf_name])


def copy_decoder_weights_from_hf(
    tvm_decoder: WhisperDecoderLMHeadTVM,
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
            raise KeyError(f"Missing HF decoder weight: {hf_name}")
        set_param_from_hf(param, hf_state[hf_name])


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

    pad_token_id = (
        hf_model.config.pad_token_id if hf_model.config.pad_token_id is not None else 0
    )
    eos_token_id = int(hf_model.config.eos_token_id)
    decoder_start_token_id = int(hf_model.config.decoder_start_token_id)
    vocab_size = int(hf_model.config.vocab_size)

    prompt_ids = get_decoder_prompt_ids(processor, decoder_start_token_id)
    max_dec_len = len(prompt_ids) + int(args.max_new_tokens)

    preprocess_model = WhisperPreprocessTVM(
        np.asarray(processor.feature_extractor.mel_filters, dtype=np.float32)
    )
    encoder_model = WhisperEncoderTVM(hf_model.config)
    decoder_model = WhisperDecoderLMHeadTVM(hf_model.config, max_dec_len=max_dec_len)

    copy_encoder_weights_from_hf(encoder_model, hf_model)
    copy_decoder_weights_from_hf(decoder_model, hf_model)

    preprocess_vm, _, preprocess_params_tvm = compile_nn_module_to_vm(
        preprocess_model, target, dev
    )
    encoder_vm, _, encoder_params_tvm = compile_nn_module_to_vm(
        encoder_model, target, dev
    )
    decoder_vm, _, decoder_params_tvm = compile_nn_module_to_vm(
        decoder_model, target, dev
    )

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
    hf_proc_features = (
        hf_inputs.input_features.detach().cpu().numpy().astype(np.float32)
    )
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
    # 3) Decoder logits compare (isolated, using HF encoder hidden)
    # --------------------------------------------------------------
    decoder_ids_np = np.full((1, max_dec_len), pad_token_id, dtype=np.int64)
    decoder_mask_np = np.zeros((1, max_dec_len), dtype=np.int32)
    decoder_ids_np[0, : len(prompt_ids)] = np.asarray(prompt_ids, dtype=np.int64)
    decoder_mask_np[0, : len(prompt_ids)] = 1

    decoder_ids_torch = torch.from_numpy(decoder_ids_np)
    decoder_mask_torch = torch.from_numpy(decoder_mask_np).to(torch.long)

    with torch.no_grad():
        hf_decoder_hidden = hf_model.model.decoder(
            input_ids=decoder_ids_torch,
            attention_mask=decoder_mask_torch,
            encoder_hidden_states=hf_encoder_hidden,
            return_dict=False,
        )[0]
        hf_logits = hf_model.proj_out(hf_decoder_hidden)

    tvm_logits_on_hf_encoder = decoder_vm["forward"](
        to_tvm_tensor(decoder_ids_np, dev),
        to_tvm_tensor(decoder_mask_np, dev),
        to_tvm_tensor(hf_encoder_hidden, dev),
        *decoder_params_tvm,
    )
    tvm_logits_on_hf_encoder_np = (
        unwrap_vm_output(tvm_logits_on_hf_encoder).numpy().astype(np.float32)
    )

    valid_prefix = len(prompt_ids)
    report_diff(
        "decoder logits (valid prefix, HF encoder hidden)",
        hf_logits[:, :valid_prefix].detach().cpu().numpy().astype(np.float32),
        tvm_logits_on_hf_encoder_np[:, :valid_prefix],
    )

    # --------------------------------------------------------------
    # 4) HF full generate reference using TVM-preprocessed features
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
            hf_out = hf_model.generate(
                force_unique_generate_call=True, **hf_generate_kwargs
            )
    except TypeError:
        with torch.no_grad():
            hf_out = hf_model.generate(**hf_generate_kwargs)

    hf_full_ids = hf_out.sequences if hasattr(hf_out, "sequences") else hf_out
    hf_full_ids = maybe_reconstruct_full_hf_ids(hf_full_ids, prompt_ids, eos_token_id)
    hf_text = processor.batch_decode(hf_full_ids, skip_special_tokens=True)[0]

    # --------------------------------------------------------------
    # 5) TVM greedy decode (encoder once + no-cache full decoder)
    # --------------------------------------------------------------
    decoder_ids_np = np.full((1, max_dec_len), pad_token_id, dtype=np.int64)
    decoder_mask_np = np.zeros((1, max_dec_len), dtype=np.int32)
    decoder_ids_np[0, : len(prompt_ids)] = np.asarray(prompt_ids, dtype=np.int64)
    decoder_mask_np[0, : len(prompt_ids)] = 1
    cur_len = len(prompt_ids)

    encoder_hidden_tvm_runtime = to_tvm_tensor(encoder_hidden_np, dev)

    for gen_idx in range(int(args.max_new_tokens)):
        tvm_logits = decoder_vm["forward"](
            to_tvm_tensor(decoder_ids_np, dev),
            to_tvm_tensor(decoder_mask_np, dev),
            encoder_hidden_tvm_runtime,
            *decoder_params_tvm,
        )
        tvm_logits_np = unwrap_vm_output(tvm_logits).numpy().astype(np.float32)

        next_logits = tvm_logits_np[0, cur_len - 1]
        next_logits = apply_whisper_suppression(
            next_logits, hf_model.generation_config, gen_idx
        )

        next_id = int(np.argmax(next_logits))
        if not (0 <= next_id < vocab_size):
            raise RuntimeError(f"Out-of-range token id: {next_id}")

        decoder_ids_np[0, cur_len] = next_id
        decoder_mask_np[0, cur_len] = 1
        cur_len += 1

        if next_id == eos_token_id:
            break

    tvm_full_ids = decoder_ids_np[:, :cur_len]
    tvm_text = processor.batch_decode(tvm_full_ids, skip_special_tokens=True)[0]

    hf_content_ids = strip_prefix_and_eos(
        hf_full_ids[0].tolist(), prompt_ids, eos_token_id
    )
    tvm_content_ids = strip_prefix_and_eos(
        tvm_full_ids[0].tolist(), prompt_ids, eos_token_id
    )

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
