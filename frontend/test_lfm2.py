import pytest
import torch
from torch.export import export, Dim
from transformers import AutoModelForCausalLM, AutoTokenizer
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.frontend import detach_params  # パラメータ分離用 :contentReference[oaicite:5]{index=5}

def test_lfm2():
    model_id = "LiquidAI/LFM2-350M"  # 350M, 700M, 1.2B
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,  # float16 / bfloat16 / float32
        device_map=None,
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    class LFM2Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask=None):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,      # まずは prefill のみ
                logits_to_keep=1,     # 最後トークンだけ計算 (LFM2 config の機能) :contentReference[oaicite:1]{index=1}
            )
            return out.logits[:, -1, :]  # (B, vocab)

    lfm2 = LFM2Wrapper(hf_model)

    batch = 1
    seq_len = 128
    dummy_ids = torch.ones((batch, seq_len), dtype=torch.long)

    dynamic_shapes = {
        "input_ids": {
            0: Dim("batch", min=1, max=4),
            1: Dim("seqlen", min=1, max=2048),
        }
    }

    exported_program = export(
        lfm2,
        (dummy_ids,),
        # dynamic_shapes=dynamic_shapes,
    )

    mod = from_exported_program(exported_program, keep_params_as_input=False)
    mod, params = detach_params(mod)
    print(mod)

if __name__ == "__main__":
    pytest.main(["-s", __file__])
