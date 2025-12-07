import argparse

import torch
import tvm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


def test_lfm2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False"
            )
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")

    model_id = "LiquidAI/LFM2-350M"  # 350M, 700M, 1.2B
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,  # float16 / bfloat16 / float32
        device_map=None,
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "あいうえ" * 42 + "か"  # 128 tokens
    print(f"Prompt: {prompt}...")
    inputs = tokenizer(prompt, return_tensors="pt")

    hf_model.to("cpu")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    print("Input tokens:", inputs["input_ids"])
    print("Running HuggingFace model on cpu...")
    with torch.no_grad():
        outputs = hf_model(**inputs)
    print(f"Output logits shape: {outputs.logits.shape}, dtype: {outputs.logits.dtype}")

    class LFM2Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask=None):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,  # まずは prefill のみ
                logits_to_keep=1,  # 最後トークンだけ計算 (LFM2 config の機能) :contentReference[oaicite:1]{index=1}
            )
            return out.logits[:, -1, :]  # (B, vocab)

    lfm2 = LFM2Wrapper(hf_model).eval()

    batch = 1
    seq_len = 128
    example_args = torch.ones((batch, seq_len), dtype=torch.long)

    dynamic_shapes = {
        "input_ids": {
            0: torch.export.Dim("batch", min=1, max=4),
            1: torch.export.Dim("seqlen", min=1, max=2048),
        }
    }

    exported_program = torch.export.export(
        lfm2,
        (example_args,),
        # dynamic_shapes=dynamic_shapes, # no dynamic shape support yet on Torch side
    )

    # PyTorch
    print("Running PyTorch model on cpu...")
    expected: torch.Tensor = lfm2(**inputs)
    print(f"Expected logits shape: {expected.shape}, dtype: {expected.dtype}")

    # Relax
    tvm_device = tvm.cpu() if args.device == "cpu" else tvm.cuda()
    target = tvm.target.Target.from_device(tvm_device)
    mod = from_exported_program(exported_program)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)

    print("Compiling Relax module...")
    exe = tvm.compile(mod, target, relax_pipeline="default")

    print(f"Running Relax model on {args.device}...")
    vm = relax.VirtualMachine(exe, tvm_device)
    tvm_args = [
        tvm.runtime.from_dlpack(x.contiguous().to(torch_device)) for x in example_args
    ]


if __name__ == "__main__":
    test_lfm2()
