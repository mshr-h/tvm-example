import numpy as np
import torch
from torch.export import export
from transformers import AutoProcessor, WhisperForConditionalGeneration
from tvm.relax.frontend.torch import from_exported_program

import tvm
from tvm import relax

MODEL_ID = "openai/whisper-tiny"

processor = AutoProcessor.from_pretrained(MODEL_ID)
hf_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).eval()

audio = np.load("audio_16khz_mono.npy").astype(np.float32)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features  # [B, 80, T]

with torch.no_grad():
    hf_generated = hf_model.generate(input_features, max_new_tokens=128)
hf_text = processor.batch_decode(hf_generated, skip_special_tokens=True)[0]

# Fixed max decoder length for static-shape compilation
MAX_DEC_LEN = 128
PAD_TOKEN_ID = (
    hf_model.config.pad_token_id if hf_model.config.pad_token_id is not None else 0
)


class WhisperNoCache(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        out = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=False,
        )
        return out[0]  # logits


wrapped = WhisperNoCache(hf_model)

# Trace with fixed-size decoder_input_ids (padded to MAX_DEC_LEN)
decoder_ids_trace = torch.full((1, MAX_DEC_LEN), PAD_TOKEN_ID, dtype=torch.long)
decoder_ids_trace[0, 0] = hf_model.config.decoder_start_token_id

with torch.no_grad():
    exported_program = export(wrapped, (input_features, decoder_ids_trace))
    mod = from_exported_program(exported_program, keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)

dev = tvm.cuda(0) if tvm.cuda(0).exist else tvm.cpu(0)
target = tvm.target.Target.from_device(dev)

s_tir_pipeline = tvm.transform.Sequential(
    [
        tvm.s_tir.transform.DefaultGPUSchedule(),
        tvm.s_tir.pipeline.default_s_tir_pipeline(),
    ]
)

ex = tvm.compile(mod, target=target, tir_pipeline=s_tir_pipeline)
vm = relax.VirtualMachine(ex, dev)

params_tvm = [tvm.runtime.tensor(p, dev) for p in params["main"]]
features_tvm = tvm.runtime.tensor(input_features.numpy(), dev)

# Autoregressive decoding with fixed-length padded input
decoder_ids = torch.full((1, MAX_DEC_LEN), PAD_TOKEN_ID, dtype=torch.long)
decoder_ids[0, 0] = hf_model.config.decoder_start_token_id
cur_len = 1

for _ in range(MAX_DEC_LEN - 1):
    dec_tvm = tvm.runtime.tensor(decoder_ids.numpy(), dev)
    logits = vm["main"](features_tvm, dec_tvm, *params_tvm)

    if not hasattr(logits, "numpy"):
        logits = logits[0]

    # Take logits at current position (last non-pad token)
    next_id = int(logits.numpy()[0, cur_len - 1].argmax())
    decoder_ids[0, cur_len] = next_id
    cur_len += 1

    if next_id == hf_model.config.eos_token_id:
        break

generated = decoder_ids[:, :cur_len]
tvm_text = processor.batch_decode(generated, skip_special_tokens=True)[0]

print(f"[HuggingFace] {hf_text}")
print(f"[TVM]         {tvm_text}")
