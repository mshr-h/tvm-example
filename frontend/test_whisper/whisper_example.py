import copy
import math

import numpy as np
import torch
from torch.export import export
from transformers import AutoProcessor, WhisperForConditionalGeneration
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.torch import from_exported_program

import tvm
from tvm import relax

MODEL_ID = "openai/whisper-tiny"

# HF generate() には ISO っぽい短い指定を使い、
# prompt ids 側は get_decoder_prompt_ids() に合わせて長い名前を使う。
HF_LANGUAGE = "en"
PROMPT_LANGUAGE = "english"
TASK = "transcribe"
NO_TIMESTAMPS = True

MAX_NEW_TOKENS = 128
VERIFY_TVM_PREPROCESS = True

# Whisper audio constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH  # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000
N_FREQ = 1 + N_FFT // 2  # 201
REFLECT_PAD = N_FFT // 2  # 200
N_MELS = 80


processor = AutoProcessor.from_pretrained(MODEL_ID)
hf_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).eval()

# ------------------------------------------------------------------
# 0) Host-side audio load + fixed-shape preparation
#    Static-shape compile 前提なので、waveform は [1, 480000] にそろえる。
#    "前処理本体" は TVM 側で実行し、ここでは shape regularization だけ行う。
# ------------------------------------------------------------------
audio = np.load("audio_16khz_mono.npy").astype(np.float32).reshape(-1)


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


waveform_fixed, valid_samples = pad_or_trim_audio(audio)

PAD_TOKEN_ID = hf_model.config.pad_token_id if hf_model.config.pad_token_id is not None else 0
EOS_TOKEN_ID = hf_model.config.eos_token_id
VOCAB_SIZE = hf_model.config.vocab_size
NUM_LAYERS = hf_model.config.decoder_layers
NUM_HEADS = hf_model.config.decoder_attention_heads
HEAD_DIM = hf_model.config.d_model // NUM_HEADS


def get_decoder_prompt_ids(processor, language, task, no_timestamps):
    if hasattr(processor, "get_decoder_prompt_ids"):
        forced = processor.get_decoder_prompt_ids(
            language=language,
            task=task,
            no_timestamps=no_timestamps,
        )
    else:
        forced = processor.tokenizer.get_decoder_prompt_ids(
            language=language,
            task=task,
            no_timestamps=no_timestamps,
        )
    return [hf_model.config.decoder_start_token_id] + [tok for _, tok in forced]


# ------------------------------------------------------------------
# TVM helpers
# ------------------------------------------------------------------
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
    return tvm.runtime.tensor(x, dev)


def get_s_tir_pipeline():
    return tvm.transform.Sequential(
        [
            tvm.s_tir.transform.DefaultGPUSchedule(),
            tvm.s_tir.pipeline.default_s_tir_pipeline(),
        ]
    )


def compile_to_vm(torch_module, example_args, target, dev):
    with torch.no_grad():
        exported_program = export(torch_module, example_args)
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        mod, params = relax.frontend.detach_params(mod)

    ex = tvm.compile(mod, target=target, tir_pipeline=get_s_tir_pipeline())
    vm = relax.VirtualMachine(ex, dev)
    params_tvm = [tvm.runtime.tensor(p, dev) for p in params["main"]]
    return vm, params_tvm


def compile_nn_module_to_vm(module: nn.Module, target, dev):
    mod, named_params = module.export_tvm(spec=module.get_default_spec())
    if len(named_params) != 0:
        raise ValueError(
            "Expected the TVM preprocessing module to have no runtime params, "
            f"but got: {[name for name, _ in named_params]}"
        )
    ex = tvm.compile(mod, target=target, tir_pipeline=get_s_tir_pipeline())
    vm = relax.VirtualMachine(ex, dev)
    return vm


# ------------------------------------------------------------------
# 1) TVM-native preprocessing via tvm.relax.frontend.nn
#    reflect-pad -> STFT(= conv1d(real/imag)) -> power -> mel -> log10/clip/scale
# ------------------------------------------------------------------
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
        # waveform: [1, 480000] float32
        # valid_samples: [1] int32

        x = op.take(waveform, self.reflect_indices, axis=1)  # [1, 480400]
        x = op.unsqueeze(x, dim=1)  # [1, 1, 480400]

        real = op.conv1d(x, self.real_kernel, stride=HOP_LENGTH, padding=0)
        imag = op.conv1d(x, self.imag_kernel, stride=HOP_LENGTH, padding=0)
        power = op.add(op.square(real), op.square(imag))  # [1, 201, 3001]
        power = op.take(power, self.keep_frame_indices, axis=2)  # [1, 201, 3000]

        # [1, 201, 3000] -> [1, 3000, 201] @ [201, 80] -> [1, 3000, 80]
        power_t = op.permute_dims(power, axes=[0, 2, 1])
        mel = op.matmul(power_t, self.mel_filters)
        mel = op.permute_dims(mel, axes=[0, 2, 1])  # [1, 80, 3000]

        log_spec = op.log(op.maximum(mel, self.log_eps))
        log_spec = op.multiply(log_spec, self.inv_ln10)  # natural log -> log10

        max_val = op.max(log_spec, axis=[1, 2], keepdims=True)
        log_spec = op.maximum(log_spec, op.subtract(max_val, self.eight))
        input_features = op.divide(op.add(log_spec, self.four), self.four)

        # HF feature extractor の attention_mask[:, ::hop_length] と同じ発想
        valid_samples_2d = op.broadcast_to(op.reshape(valid_samples, [1, 1]), [1, N_FRAMES])
        feature_attention_mask = op.astype(
            op.less(self.frame_starts, valid_samples_2d),
            "int32",
        )
        return input_features, feature_attention_mask

    def get_default_spec(self):
        mod_spec = {
            "forward": {
                "waveform": nn.spec.Tensor([1, N_SAMPLES], "float32"),
                "valid_samples": nn.spec.Tensor([1], "int32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            }
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)


# ------------------------------------------------------------------
# 2) HF helpers
# ------------------------------------------------------------------
def maybe_reconstruct_full_hf_ids(hf_ids, prompt_ids, eos_token_id):
    """
    Whisper generate() の戻りが transformers 版によって
    full sequence だったり content-only だったりするので吸収する。
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


def strip_prefix_and_eos(seq, prompt_ids, eos_token_id):
    if isinstance(seq, torch.Tensor):
        seq = seq.tolist()
    seq = list(seq)
    if seq[: len(prompt_ids)] == prompt_ids:
        seq = seq[len(prompt_ids) :]
    if seq and seq[-1] == eos_token_id:
        seq = seq[:-1]
    return seq


def apply_whisper_suppression(logits_np, generation_config, generated_token_index):
    """
    HF generate() の Whisper 既定に少し寄せるための簡易版。
    greedy の smoke test 用として十分なことが多い。
    """
    logits_np = logits_np.copy()

    suppress_tokens = getattr(generation_config, "suppress_tokens", None)
    if suppress_tokens is not None:
        logits_np[np.array(suppress_tokens, dtype=np.int64)] = -np.inf

    begin_suppress_tokens = getattr(generation_config, "begin_suppress_tokens", None)
    if generated_token_index == 0 and begin_suppress_tokens is not None:
        logits_np[np.array(begin_suppress_tokens, dtype=np.int64)] = -np.inf

    return logits_np


prompt_ids = get_decoder_prompt_ids(processor, PROMPT_LANGUAGE, TASK, NO_TIMESTAMPS)
MAX_DEC_LEN = len(prompt_ids) + MAX_NEW_TOKENS
assert MAX_DEC_LEN <= hf_model.config.max_target_positions, (
    f"MAX_DEC_LEN={MAX_DEC_LEN} exceeds max_target_positions={hf_model.config.max_target_positions}"
)


class WhisperEncoderOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features):
        return self.encoder(input_features, return_dict=False)[0]


class WhisperCrossKVOnly(torch.nn.Module):
    """
    cross-attention の K/V を 1 回だけ前計算する。
    出力 shape:
      [num_layers, batch, num_heads, encoder_seq_len, head_dim]
    """

    def __init__(self, model):
        super().__init__()
        self.layers = model.model.decoder.layers

    def forward(self, encoder_hidden_states):
        all_k = []
        all_v = []
        batch_size, seq_len, _ = encoder_hidden_states.shape

        for layer in self.layers:
            attn = layer.encoder_attn
            k = attn.k_proj(encoder_hidden_states)
            v = attn.v_proj(encoder_hidden_states)

            k = k.view(batch_size, seq_len, attn.num_heads, attn.head_dim)
            v = v.view(batch_size, seq_len, attn.num_heads, attn.head_dim)

            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

            all_k.append(k)
            all_v.append(v)

        return torch.stack(all_k, dim=0), torch.stack(all_v, dim=0)


class WhisperDecoderStepSelfCached(torch.nn.Module):
    """
    1 token ずつ進める decoder-step。
    self-attention の K/V は明示テンソル cache として入出力し、
    cross-attention の K/V は外で事前計算して入力として受ける。

    入力:
      input_token_ids : [B, 1]
      position_ids    : [B, 1]
      self_k_cache    : [L, B, H, MAX_DEC_LEN, D]
      self_v_cache    : [L, B, H, MAX_DEC_LEN, D]
      cross_k_cache   : [L, B, H, ENC_LEN, D]
      cross_v_cache   : [L, B, H, ENC_LEN, D]

    出力:
      logits          : [B, 1, vocab]
      new_self_k      : [L, B, H, MAX_DEC_LEN, D]
      new_self_v      : [L, B, H, MAX_DEC_LEN, D]
    """

    def __init__(self, model, max_cache_len):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.max_cache_len = max_cache_len

    def _update_cache(self, cache, new_value, position_ids):
        # cache:     [B, H, MAX, D]
        # new_value: [B, H, 1,   D]
        bsz = position_ids.shape[0]
        positions = torch.arange(self.max_cache_len, device=position_ids.device, dtype=position_ids.dtype).view(
            1, 1, self.max_cache_len, 1
        )
        pos_mask = (positions == position_ids.view(bsz, 1, 1, 1)).to(new_value.dtype)
        return cache * (1.0 - pos_mask) + new_value.expand(-1, -1, self.max_cache_len, -1) * pos_mask

    def _make_self_mask(self, position_ids, dtype, device):
        bsz = position_ids.shape[0]
        positions = torch.arange(self.max_cache_len, device=device, dtype=position_ids.dtype).view(
            1, 1, 1, self.max_cache_len
        )
        valid = positions <= position_ids.view(bsz, 1, 1, 1)
        zeros = torch.zeros((bsz, 1, 1, self.max_cache_len), device=device, dtype=dtype)
        negs = torch.full(
            (bsz, 1, 1, self.max_cache_len),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
        return torch.where(valid, zeros, negs)

    def _attend(self, q, k, v, out_proj, attention_mask):
        attn_weights = torch.matmul(q, k.transpose(2, 3))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        bsz, _, q_len, _ = attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return out_proj(attn_output)

    def forward(
        self,
        input_token_ids,
        position_ids,
        self_k_cache,
        self_v_cache,
        cross_k_cache,
        cross_v_cache,
    ):
        inputs_embeds = self.decoder.embed_tokens(input_token_ids) * self.decoder.embed_scale
        positions = self.decoder.embed_positions(input_token_ids, position_ids=position_ids)
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)

        new_self_k = []
        new_self_v = []

        for layer_idx, layer in enumerate(self.decoder.layers):
            # --------------------------
            # self-attention block
            # --------------------------
            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)

            self_attn = layer.self_attn
            bsz, tgt_len, _ = hidden_states.shape

            q = self_attn.q_proj(hidden_states) * self_attn.scaling
            q = q.view(bsz, tgt_len, self_attn.num_heads, self_attn.head_dim)
            q = q.transpose(1, 2).contiguous()  # [B, H, 1, D]

            k_new = self_attn.k_proj(hidden_states)
            v_new = self_attn.v_proj(hidden_states)

            k_new = k_new.view(bsz, tgt_len, self_attn.num_heads, self_attn.head_dim)
            v_new = v_new.view(bsz, tgt_len, self_attn.num_heads, self_attn.head_dim)

            k_new = k_new.transpose(1, 2).contiguous()  # [B, H, 1, D]
            v_new = v_new.transpose(1, 2).contiguous()  # [B, H, 1, D]

            k_cache = self._update_cache(self_k_cache[layer_idx], k_new, position_ids)
            v_cache = self._update_cache(self_v_cache[layer_idx], v_new, position_ids)

            self_mask = self._make_self_mask(position_ids, q.dtype, q.device)
            self_attn_out = self._attend(
                q,
                k_cache,
                v_cache,
                self_attn.out_proj,
                self_mask,
            )
            hidden_states = residual + self_attn_out

            # --------------------------
            # cross-attention block
            # --------------------------
            residual = hidden_states
            hidden_states = layer.encoder_attn_layer_norm(hidden_states)

            cross_attn = layer.encoder_attn
            q = cross_attn.q_proj(hidden_states) * cross_attn.scaling
            q = q.view(bsz, tgt_len, cross_attn.num_heads, cross_attn.head_dim)
            q = q.transpose(1, 2).contiguous()

            cross_attn_out = self._attend(
                q,
                cross_k_cache[layer_idx],
                cross_v_cache[layer_idx],
                cross_attn.out_proj,
                None,
            )
            hidden_states = residual + cross_attn_out

            # --------------------------
            # FFN block
            # --------------------------
            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.activation_fn(layer.fc1(hidden_states))
            hidden_states = layer.fc2(hidden_states)
            hidden_states = residual + hidden_states

            new_self_k.append(k_cache)
            new_self_v.append(v_cache)

        hidden_states = self.decoder.layer_norm(hidden_states)
        logits = self.proj_out(hidden_states)

        return logits, torch.stack(new_self_k, dim=0), torch.stack(new_self_v, dim=0)


# ------------------------------------------------------------------
# 3) Compile / run TVM preprocessing
# ------------------------------------------------------------------
dev = tvm.cuda(0) if tvm.cuda(0).exist else tvm.cpu(0)
target = tvm.target.Target.from_device(dev)

preprocess_model = WhisperPreprocessTVM(np.asarray(processor.feature_extractor.mel_filters, dtype=np.float32))
preprocess_vm = compile_nn_module_to_vm(preprocess_model, target, dev)

waveform_tvm = to_tvm_tensor(waveform_fixed, dev)
valid_samples_tvm = to_tvm_tensor(valid_samples, dev)
preprocess_out = preprocess_vm["forward"](waveform_tvm, valid_samples_tvm)
input_features_tvm, attention_mask_tvm = unwrap_vm_outputs(preprocess_out)

input_features_np = unwrap_vm_output(input_features_tvm).numpy().astype(np.float32)
attention_mask_np = unwrap_vm_output(attention_mask_tvm).numpy().astype(np.int32)

input_features = torch.from_numpy(input_features_np)
attention_mask = torch.from_numpy(attention_mask_np).to(torch.long)

if VERIFY_TVM_PREPROCESS:
    hf_inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )
    hf_proc_features = hf_inputs.input_features.detach().cpu().numpy().astype(np.float32)
    hf_proc_mask = hf_inputs.attention_mask.detach().cpu().numpy().astype(np.int32)

    preproc_max_abs_diff = np.max(np.abs(hf_proc_features - input_features_np))
    preproc_mask_equal = np.array_equal(hf_proc_mask, attention_mask_np)

    print(f"[preprocess max abs diff] {preproc_max_abs_diff:.8f}")
    print(f"[preprocess mask equal] {preproc_mask_equal}")

# ------------------------------------------------------------------
# 4) HF reference (TVM-preprocessed features をそのまま投入)
# ------------------------------------------------------------------
gen_cfg = copy.deepcopy(hf_model.generation_config)
gen_cfg.max_length = None
if hasattr(gen_cfg, "forced_decoder_ids"):
    gen_cfg.forced_decoder_ids = None

hf_generate_kwargs = dict(
    input_features=input_features,
    generation_config=gen_cfg,
    language=HF_LANGUAGE,
    task=TASK,
    max_new_tokens=MAX_NEW_TOKENS,
    return_dict_in_generate=True,
    attention_mask=attention_mask,
)

try:
    with torch.no_grad():
        hf_out = hf_model.generate(
            force_unique_generate_call=True,
            **hf_generate_kwargs,
        )
except TypeError:
    with torch.no_grad():
        hf_out = hf_model.generate(**hf_generate_kwargs)

hf_full_ids = hf_out.sequences if hasattr(hf_out, "sequences") else hf_out
hf_full_ids = maybe_reconstruct_full_hf_ids(hf_full_ids, prompt_ids, EOS_TOKEN_ID)
hf_text = processor.batch_decode(hf_full_ids, skip_special_tokens=True)[0]

# ------------------------------------------------------------------
# 5) TVM compile: encoder
# ------------------------------------------------------------------
encoder_module = WhisperEncoderOnly(hf_model).eval()
encoder_vm, encoder_params_tvm = compile_to_vm(
    encoder_module,
    (input_features,),
    target,
    dev,
)

with torch.no_grad():
    encoder_hidden_trace = encoder_module(input_features)

# ------------------------------------------------------------------
# 6) TVM compile: cross K/V precompute
# ------------------------------------------------------------------
cross_kv_module = WhisperCrossKVOnly(hf_model).eval()
cross_kv_vm, cross_kv_params_tvm = compile_to_vm(
    cross_kv_module,
    (encoder_hidden_trace,),
    target,
    dev,
)

with torch.no_grad():
    cross_k_trace, cross_v_trace = cross_kv_module(encoder_hidden_trace)

ENC_LEN = cross_k_trace.shape[-2]

# ------------------------------------------------------------------
# 7) TVM compile: decoder-step (self cache + precomputed cross K/V)
# ------------------------------------------------------------------
step_module = WhisperDecoderStepSelfCached(hf_model, MAX_DEC_LEN).eval()

token_trace = torch.tensor([[prompt_ids[0]]], dtype=torch.long)
position_trace = torch.tensor([[0]], dtype=torch.long)
self_k_cache_trace = torch.zeros(
    (NUM_LAYERS, 1, NUM_HEADS, MAX_DEC_LEN, HEAD_DIM),
    dtype=encoder_hidden_trace.dtype,
)
self_v_cache_trace = torch.zeros(
    (NUM_LAYERS, 1, NUM_HEADS, MAX_DEC_LEN, HEAD_DIM),
    dtype=encoder_hidden_trace.dtype,
)

step_vm, step_params_tvm = compile_to_vm(
    step_module,
    (
        token_trace,
        position_trace,
        self_k_cache_trace,
        self_v_cache_trace,
        cross_k_trace,
        cross_v_trace,
    ),
    target,
    dev,
)

# ------------------------------------------------------------------
# 8) Run encoder once
# ------------------------------------------------------------------
features_tvm = to_tvm_tensor(input_features, dev)
encoder_hidden_tvm = encoder_vm["main"](features_tvm, *encoder_params_tvm)
encoder_hidden_tvm = unwrap_vm_output(encoder_hidden_tvm)

# ------------------------------------------------------------------
# 9) Run cross K/V precompute once
# ------------------------------------------------------------------
cross_out = cross_kv_vm["main"](encoder_hidden_tvm, *cross_kv_params_tvm)
cross_k_tvm, cross_v_tvm = unwrap_vm_outputs(cross_out)

# ------------------------------------------------------------------
# 10) Host-side decoder loop
#     prompt を token-by-token で prefill して self cache を埋める
# ------------------------------------------------------------------
self_k_cache_tvm = to_tvm_tensor(self_k_cache_trace, dev)
self_v_cache_tvm = to_tvm_tensor(self_v_cache_trace, dev)

last_logits_tvm = None

for pos, tok in enumerate(prompt_ids):
    tok_tvm = to_tvm_tensor(np.array([[tok]], dtype=np.int64), dev)
    pos_tvm = to_tvm_tensor(np.array([[pos]], dtype=np.int64), dev)
    step_out = step_vm["main"](
        tok_tvm,
        pos_tvm,
        self_k_cache_tvm,
        self_v_cache_tvm,
        cross_k_tvm,
        cross_v_tvm,
        *step_params_tvm,
    )
    last_logits_tvm, self_k_cache_tvm, self_v_cache_tvm = unwrap_vm_outputs(step_out)

generated_full_ids = list(prompt_ids)

for gen_idx in range(MAX_NEW_TOKENS):
    logits_np = unwrap_vm_output(last_logits_tvm).numpy()[0, 0]
    logits_np = apply_whisper_suppression(logits_np, hf_model.generation_config, gen_idx)

    next_id = int(logits_np.argmax())
    assert 0 <= next_id < VOCAB_SIZE, f"out-of-range next_id={next_id}"

    generated_full_ids.append(next_id)

    if next_id == EOS_TOKEN_ID:
        break

    pos = len(generated_full_ids) - 1
    assert pos < MAX_DEC_LEN, f"decoder length overflow: pos={pos}, max={MAX_DEC_LEN}"

    tok_tvm = to_tvm_tensor(np.array([[next_id]], dtype=np.int64), dev)
    pos_tvm = to_tvm_tensor(np.array([[pos]], dtype=np.int64), dev)
    step_out = step_vm["main"](
        tok_tvm,
        pos_tvm,
        self_k_cache_tvm,
        self_v_cache_tvm,
        cross_k_tvm,
        cross_v_tvm,
        *step_params_tvm,
    )
    last_logits_tvm, self_k_cache_tvm, self_v_cache_tvm = unwrap_vm_outputs(step_out)

generated = torch.tensor([generated_full_ids], dtype=torch.long)
tvm_text = processor.batch_decode(generated, skip_special_tokens=True)[0]

hf_text_ids = strip_prefix_and_eos(hf_full_ids[0], prompt_ids, EOS_TOKEN_ID)
tvm_text_ids = strip_prefix_and_eos(generated[0], prompt_ids, EOS_TOKEN_ID)

print(f"[HF]          {hf_text}")
print(f"[TVM cached]  {tvm_text}")
print(f"[HF full IDs ] {hf_full_ids[0].tolist()}")
print(f"[TVM full IDs] {generated[0].tolist()}")
print(f"[HF text IDs ] {hf_text_ids}")
print(f"[TVM text IDs] {tvm_text_ids}")
print(f"[match full] {hf_full_ids[0].tolist() == generated[0].tolist()}")
print(f"[match text] {hf_text_ids == tvm_text_ids}")
print(f"[ENC_LEN] {ENC_LEN}")
print(f"[cache shape] {(NUM_LAYERS, 1, NUM_HEADS, MAX_DEC_LEN, HEAD_DIM)}")
