import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

import tvm

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
            "Compile WhisperBundle into a single Relax executable (.so) plus native TVM params (.params) "
            "and metadata.json. "
            "This step requires transformers/torch."
        )
    )
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def build_target(name: str):
    if name == "cuda":
        return tvm.target.Target("cuda")
    return tvm.target.Target("llvm")


def get_s_tir_pipeline():
    return tvm.transform.Sequential(
        [
            tvm.s_tir.transform.DefaultGPUSchedule(),
            tvm.s_tir.pipeline.default_s_tir_pipeline(),
        ]
    )


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


# -----------------------------------------------------------------------------
# TVM-native cross-KV and cached decoder-step
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
        self.head_dim = self.d_model // self.decoder_attention_heads
        self.max_source_positions = int(config.max_source_positions)
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


# -----------------------------------------------------------------------------
# WhisperBundle
# -----------------------------------------------------------------------------
class WhisperBundle(nn.Module):
    def __init__(self, config, mel_filters: np.ndarray, max_dec_len: int):
        super().__init__()
        self.preprocess_mod = WhisperPreprocessTVM(mel_filters)
        self.encoder_mod = WhisperEncoderTVM(config)
        self.cross_kv_mod = WhisperCrossKVCachedTVM(config)
        self.decoder_step_mod = WhisperDecoderCachedStepTVM(config, max_dec_len=max_dec_len)

        self.decoder_layers = int(config.decoder_layers)
        self.decoder_attention_heads = int(config.decoder_attention_heads)
        self.head_dim = int(config.d_model) // int(config.decoder_attention_heads)
        self.max_source_positions = int(config.max_source_positions)
        self.max_dec_len = int(max_dec_len)
        self.max_past_len = self.max_dec_len - 1

    def preprocess(self, waveform: Tensor, valid_samples: Tensor):
        return self.preprocess_mod(waveform, valid_samples)

    def encode(self, input_features: Tensor):
        return self.encoder_mod(input_features)

    def cross_kv(self, encoder_hidden_states: Tensor):
        return self.cross_kv_mod(encoder_hidden_states)

    def decode_step(
        self,
        token_id: Tensor,
        position_id: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        past_keep_mask: Tensor,
        cross_k_cache: Tensor,
        cross_v_cache: Tensor,
    ):
        return self.decoder_step_mod(
            token_id,
            position_id,
            self_k_cache,
            self_v_cache,
            past_keep_mask,
            cross_k_cache,
            cross_v_cache,
        )

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "preprocess": {
                    "waveform": nn.spec.Tensor([1, N_SAMPLES], "float32"),
                    "valid_samples": nn.spec.Tensor([1], "int32"),
                    "$": {"param_mode": "none", "effect_mode": "none"},
                },
                "encode": {
                    "input_features": nn.spec.Tensor([1, N_MELS, N_FRAMES], "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "cross_kv": {
                    "encoder_hidden_states": nn.spec.Tensor(
                        [1, self.max_source_positions, self.decoder_attention_heads * self.head_dim],
                        "float32",
                    ),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "decode_step": {
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
                },
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


def save_params_tvm(named_params, out_path: Path):
    params = {}
    names = []
    for name, param in named_params:
        if param.data is None:
            raise ValueError(f"Parameter {name} is not bound")
        params[name] = param.data
        names.append(name)
    tvm.runtime.save_param_dict_to_file(params, str(out_path))
    return names


def export_executable(executable, out_path: Path):
    if hasattr(executable, "export_library"):
        executable.export_library(str(out_path))
        return
    if hasattr(executable, "mod") and hasattr(executable.mod, "export_library"):
        executable.mod.export_library(str(out_path))
        return
    raise AttributeError("Compiled executable does not expose export_library")


def main():
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_id)
    hf_model = WhisperForConditionalGeneration.from_pretrained(args.model_id).eval()

    decoder_start_token_id = int(hf_model.config.decoder_start_token_id)
    prompt_ids = get_decoder_prompt_ids(processor, decoder_start_token_id)
    max_dec_len = len(prompt_ids) + int(args.max_new_tokens)

    bundle = WhisperBundle(
        hf_model.config,
        mel_filters=np.asarray(processor.feature_extractor.mel_filters, dtype=np.float32),
        max_dec_len=max_dec_len,
    )
    copy_encoder_weights_from_hf(bundle.encoder_mod, hf_model)
    copy_cross_kv_weights_from_hf(bundle.cross_kv_mod, hf_model)
    copy_decoder_step_weights_from_hf(bundle.decoder_step_mod, hf_model)

    target = build_target(args.target)
    mod, named_params = bundle.export_tvm(spec=bundle.get_default_spec())
    compile_kwargs = {"target": target}
    if target.kind.name == "cuda":
        compile_kwargs["tir_pipeline"] = get_s_tir_pipeline()
    executable = tvm.compile(mod, **compile_kwargs)

    lib_path = out_dir / "whisper_bundle.so"
    params_path = out_dir / "whisper_bundle.params"
    metadata_path = out_dir / "whisper_bundle_metadata.json"

    export_executable(executable, lib_path)
    param_names = save_params_tvm(named_params, params_path)

    metadata = {
        "lib_name": lib_path.name,
        "params_name": params_path.name,
        "sample_rate": SAMPLE_RATE,
        "n_samples": N_SAMPLES,
        "max_new_tokens_default": int(args.max_new_tokens),
        "max_dec_len_compiled": int(max_dec_len),
        "eos_token_id": int(hf_model.config.eos_token_id),
        "vocab_size": int(hf_model.config.vocab_size),
        "prompt_ids": [int(x) for x in prompt_ids],
        "special_ids": [int(x) for x in processor.tokenizer.all_special_ids],
        "suppress_tokens": [int(x) for x in getattr(hf_model.generation_config, "suppress_tokens", []) or []],
        "begin_suppress_tokens": [
            int(x) for x in getattr(hf_model.generation_config, "begin_suppress_tokens", []) or []
        ],
        "decoder_layers": int(hf_model.config.decoder_layers),
        "decoder_attention_heads": int(hf_model.config.decoder_attention_heads),
        "head_dim": int(hf_model.config.d_model) // int(hf_model.config.decoder_attention_heads),
        "param_count": len(param_names),
        "param_names": param_names,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote: {lib_path}")
    print(f"wrote: {params_path}")
    print(f"wrote: {metadata_path}")
    print(f"param_count: {len(param_names)}")


if __name__ == "__main__":
    main()
