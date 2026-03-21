import numpy as np
import torch
from torch.export import export
from transformers import AutoProcessor, WhisperForConditionalGeneration
from tvm.relax.frontend.torch import from_exported_program

import tvm
from tvm import relax

MODEL_ID = "openai/whisper-tiny"

# 必要なら Whisper の context tokens を明示する
USE_FORCED_PROMPT = False
FORCED_LANGUAGE = "english"
FORCED_TASK = "transcribe"
NO_TIMESTAMPS = True

MAX_NEW_TOKENS = 128

processor = AutoProcessor.from_pretrained(MODEL_ID)
hf_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).eval()

audio = np.load("audio_16khz_mono.npy").astype(np.float32)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features  # [B, 80, T]

PAD_TOKEN_ID = hf_model.config.pad_token_id if hf_model.config.pad_token_id is not None else 0


def build_prompt_ids():
    prompt_ids = [hf_model.config.decoder_start_token_id]
    forced_decoder_ids = None

    if USE_FORCED_PROMPT:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=FORCED_LANGUAGE,
            task=FORCED_TASK,
            no_timestamps=NO_TIMESTAMPS,
        )
        # get_decoder_prompt_ids() は [(position, token_id), ...] を返す
        prompt_ids.extend([token_id for _, token_id in forced_decoder_ids])

    return prompt_ids, forced_decoder_ids


prompt_ids, forced_decoder_ids = build_prompt_ids()
MAX_DEC_LEN = len(prompt_ids) + MAX_NEW_TOKENS
assert MAX_DEC_LEN <= hf_model.config.max_target_positions, (
    f"MAX_DEC_LEN={MAX_DEC_LEN} exceeds max_target_positions={hf_model.config.max_target_positions}"
)


class WhisperEncoderOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features):
        # BaseModelOutput を返すので 0 番目を取り出す
        encoder_out = self.encoder(input_features)
        return encoder_out[0]  # encoder_hidden_states


class WhisperDecoderNoCache(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.proj_out = model.proj_out

    def forward(self, encoder_hidden_states, decoder_input_ids, decoder_attention_mask):
        # encoder は外で実行済みなので encoder_outputs を注入する
        out = self.model(
            encoder_outputs=(encoder_hidden_states,),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=False,
        )
        decoder_hidden = out[0]  # [B, T_dec, d_model]
        logits = self.proj_out(decoder_hidden)  # [B, T_dec, vocab]
        return logits


def unwrap_vm_output(x):
    # TVM のバージョンによっては NDArray か Array[NDArray] が返る
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


# ------------------------------------------------------------------
# 1) HF reference (比較用)
# ------------------------------------------------------------------
generate_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
if forced_decoder_ids is not None:
    generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

with torch.no_grad():
    hf_generated = hf_model.generate(input_features, **generate_kwargs)
hf_text = processor.batch_decode(hf_generated, skip_special_tokens=True)[0]

# ------------------------------------------------------------------
# 2) TVM compile: encoder
# ------------------------------------------------------------------
dev = tvm.cuda(0)
assert dev.exist, "TVM から CUDA device が見えていません"
target = tvm.target.Target.from_device(dev)

encoder_module = WhisperEncoderOnly(hf_model).eval()
encoder_vm, encoder_params_tvm = compile_to_vm(
    encoder_module,
    (input_features,),
    target,
    dev,
)

# decoder の trace shape 用に encoder 出力 shape を 1 回だけ得る
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

    # 現在位置の logits を使う
    next_logits = logits.numpy()[0, cur_len - 1]
    next_id = int(next_logits.argmax())

    decoder_ids[0, cur_len] = next_id
    decoder_mask[0, cur_len] = 1
    cur_len += 1

    if next_id == hf_model.config.eos_token_id:
        break

generated = decoder_ids[:, :cur_len]
tvm_text = processor.batch_decode(generated, skip_special_tokens=True)[0]

print(f"[HF]  {hf_text}")
print(f"[TVM] {tvm_text}")
print(f"[HF IDs ] {hf_generated[0].tolist()}")
print(f"[TVM IDs] {generated[0].tolist()}")
