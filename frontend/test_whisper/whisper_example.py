import copy

import numpy as np
import torch
from torch.export import export
from transformers import AutoProcessor, WhisperForConditionalGeneration
from tvm.relax.frontend.torch import from_exported_program

import tvm
from tvm import relax

MODEL_ID = "openai/whisper-tiny"
LANGUAGE = "en"
TASK = "transcribe"
NO_TIMESTAMPS = True
MAX_NEW_TOKENS = 128

processor = AutoProcessor.from_pretrained(MODEL_ID)
hf_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).eval()

audio = np.load("audio_16khz_mono.npy").astype(np.float32)
inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
    return_attention_mask=True,
)
input_features = inputs.input_features  # [B, 80, T]
attention_mask = getattr(inputs, "attention_mask", None)

PAD_TOKEN_ID = hf_model.config.pad_token_id if hf_model.config.pad_token_id is not None else 0
EOS_TOKEN_ID = hf_model.config.eos_token_id
VOCAB_SIZE = hf_model.config.vocab_size


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


def unwrap_vm_output(x):
    while not hasattr(x, "numpy"):
        x = x[0]
    return x


def to_tvm_tensor(x, dev):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return tvm.runtime.tensor(x, dev)


def compile_to_vm(torch_module, example_args, target, dev):
    with torch.no_grad():
        exported_program = export(torch_module, example_args)
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        mod, params = relax.frontend.detach_params(mod)

    s_tir_pipeline = tvm.transform.Sequential(
        [
            tvm.s_tir.transform.DefaultGPUSchedule(),
            tvm.s_tir.pipeline.default_s_tir_pipeline(),
        ]
    )

    ex = tvm.compile(mod, target=target, tir_pipeline=s_tir_pipeline)
    vm = relax.VirtualMachine(ex, dev)
    params_tvm = [tvm.runtime.tensor(p, dev) for p in params["main"]]
    return vm, params_tvm


def maybe_reconstruct_full_hf_ids(hf_ids, prompt_ids, eos_token_id):
    """
    Current Whisper generate() can return full sequences (prompt + eos) when
    return_dict_in_generate=True, but older versions may still hand back the
    content-only tensor. Reconstruct if needed so HF and TVM can be compared
    directly.
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


prompt_ids = get_decoder_prompt_ids(processor, LANGUAGE, TASK, NO_TIMESTAMPS)
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


class WhisperDecoderNoCache(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.proj_out = model.proj_out

    def forward(self, encoder_hidden_states, decoder_input_ids, decoder_attention_mask):
        out = self.model(
            encoder_outputs=(encoder_hidden_states,),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=False,
        )
        decoder_hidden = out[0]
        logits = self.proj_out(decoder_hidden)
        return logits


# ------------------------------------------------------------------
# 1) HF reference: explicit language/task + attention_mask
# ------------------------------------------------------------------
gen_cfg = copy.deepcopy(hf_model.generation_config)
gen_cfg.max_length = None
if hasattr(gen_cfg, "forced_decoder_ids"):
    gen_cfg.forced_decoder_ids = None

hf_generate_kwargs = dict(
    input_features=input_features,
    generation_config=gen_cfg,
    language=LANGUAGE,
    task=TASK,
    max_new_tokens=MAX_NEW_TOKENS,
    return_dict_in_generate=True,
)
if attention_mask is not None:
    hf_generate_kwargs["attention_mask"] = attention_mask

# If the installed transformers exposes this flag, it guarantees a single
# internal generate() call and returns prompt + eos, which is useful for testing.
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
# 2) TVM compile: encoder
# ------------------------------------------------------------------
dev = tvm.cuda(0) if tvm.cuda(0).exist else tvm.cpu(0)
target = tvm.target.Target.from_device(dev)

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
# 3) TVM compile: decoder (no-cache, static padded input)
# ------------------------------------------------------------------
decoder_ids_trace = torch.full((1, MAX_DEC_LEN), PAD_TOKEN_ID, dtype=torch.long)
decoder_mask_trace = torch.zeros((1, MAX_DEC_LEN), dtype=torch.long)
decoder_ids_trace[0, : len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long)
decoder_mask_trace[0, : len(prompt_ids)] = 1

decoder_module = WhisperDecoderNoCache(hf_model).eval()
decoder_vm, decoder_params_tvm = compile_to_vm(
    decoder_module,
    (encoder_hidden_trace, decoder_ids_trace, decoder_mask_trace),
    target,
    dev,
)

# ------------------------------------------------------------------
# 4) Run encoder once
# ------------------------------------------------------------------
features_tvm = to_tvm_tensor(input_features, dev)
encoder_hidden_tvm = encoder_vm["main"](features_tvm, *encoder_params_tvm)
encoder_hidden_tvm = unwrap_vm_output(encoder_hidden_tvm)

# ------------------------------------------------------------------
# 5) Host-side decoder loop
# ------------------------------------------------------------------
decoder_ids = torch.full((1, MAX_DEC_LEN), PAD_TOKEN_ID, dtype=torch.long)
decoder_mask = torch.zeros((1, MAX_DEC_LEN), dtype=torch.long)
decoder_ids[0, : len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long)
decoder_mask[0, : len(prompt_ids)] = 1
cur_len = len(prompt_ids)

for _ in range(MAX_NEW_TOKENS):
    dec_ids_tvm = to_tvm_tensor(decoder_ids, dev)
    dec_mask_tvm = to_tvm_tensor(decoder_mask, dev)

    logits = decoder_vm["main"](
        encoder_hidden_tvm,
        dec_ids_tvm,
        dec_mask_tvm,
        *decoder_params_tvm,
    )
    logits = unwrap_vm_output(logits)

    next_id = int(logits.numpy()[0, cur_len - 1].argmax())
    assert 0 <= next_id < VOCAB_SIZE, f"out-of-range next_id={next_id}"
    assert cur_len < MAX_DEC_LEN, f"decoder length overflow: cur_len={cur_len}, max={MAX_DEC_LEN}"

    decoder_ids[0, cur_len] = next_id
    decoder_mask[0, cur_len] = 1
    cur_len += 1

    if next_id == EOS_TOKEN_ID:
        break

generated = decoder_ids[:, :cur_len]
tvm_text = processor.batch_decode(generated, skip_special_tokens=True)[0]

hf_text_ids = strip_prefix_and_eos(hf_full_ids[0], prompt_ids, EOS_TOKEN_ID)
tvm_text_ids = strip_prefix_and_eos(generated[0], prompt_ids, EOS_TOKEN_ID)

print(f"[HF]         {hf_text}")
print(f"[TVM]        {tvm_text}")
print(f"[HF full IDs ] {hf_full_ids[0].tolist()}")
print(f"[TVM full IDs] {generated[0].tolist()}")
print(f"[HF text IDs ] {hf_text_ids}")
print(f"[TVM text IDs] {tvm_text_ids}")
print(f"[match full] {hf_full_ids[0].tolist() == generated[0].tolist()}")
print(f"[match text] {hf_text_ids == tvm_text_ids}")
