import argparse

import torch
import tvm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tvm import dlight, relax
from tvm.relax import register_pipeline
from tvm.relax.frontend import detach_params
from tvm.relax.frontend.torch import from_exported_program


@register_pipeline("opt_llm")
def _pipeline():
    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(
        mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext
    ) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                relax.backend.DispatchSampling(),
                relax.backend.DispatchSortScan(),
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                relax.transform.DeadCodeElimination(),
                dlight.ApplyDefaultSchedule(
                    dlight.gpu.Matmul(),
                    dlight.gpu.GEMV(),
                    dlight.gpu.Reduction(),
                    dlight.gpu.GeneralReduction(),
                    dlight.gpu.Fallback(),
                ),
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                relax.transform.StaticPlanBlockMemory(),
                relax.transform.RewriteCUDAGraph(),
                relax.transform.LowerAllocTensor(),
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.ComputePrimValue(),
                relax.transform.VMShapeLower(),
                relax.transform.AttachGlobalSymbol(),
                tvm.tir.transform.DefaultGPUSchedule(),
            ]
        )
        mod = seq(mod)
        return mod

    return _pipeline


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
    example_input_ids = torch.ones((batch, seq_len), dtype=torch.long)
    example_attention_mask = torch.ones((batch, seq_len), dtype=torch.long)

    dynamic_shapes = {
        "input_ids": {
            0: torch.export.Dim("batch", min=1, max=4),
            1: torch.export.Dim("seqlen", min=1, max=2048),
        }
    }

    exported_program = torch.export.export(
        lfm2,
        (example_input_ids, example_attention_mask),
        # dynamic_shapes=dynamic_shapes, # no dynamic shape support yet on Torch side
    )

    # PyTorch
    print("Running PyTorch model on cpu...")
    expected: torch.Tensor = lfm2(**inputs)
    print(f"Expected logits shape: {expected.shape}, dtype: {expected.dtype}")

    # Relax
    tvm_device = tvm.cpu() if args.device == "cpu" else tvm.cuda()
    target = tvm.target.Target.from_device(tvm_device)
    mod = from_exported_program(exported_program, keep_params_as_input=True)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    mod, params = detach_params(mod)

    exe = tvm.compile(mod, target, relax_pipeline="opt_llm")
    vm = relax.VirtualMachine(exe, tvm_device)
    tvm_args = [
        tvm.runtime.from_dlpack(inputs["input_ids"].contiguous().to(torch_device))
    ]
    if "attention_mask" in inputs:
        tvm_args.append(
            tvm.runtime.from_dlpack(inputs["attention_mask"].contiguous().to(torch_device))
        )

    main_params = mod["main"].params
    weight_vars = main_params[2:]
    bound_params = params.get("main", [])
    if bound_params and len(bound_params) != len(weight_vars):
        raise RuntimeError(
            f"Mismatch between Relax weight vars ({len(weight_vars)}) and detached params "
            f"({len(bound_params)})"
        )
    name_to_param = {
        var.name_hint: arr for var, arr in zip(weight_vars, bound_params)
    }
    param_list = [
        name_to_param[var.name_hint].copyto(tvm_device) for var in weight_vars
    ]
    tvm_outputs = vm["main"](*tvm_args, *param_list)

    # check if the outputs match
    rtol = 1e-4
    atol = 1e-4
    if isinstance(expected, dict):
        for i, key in enumerate(expected.keys()):
            actual = torch.from_numpy(tvm_outputs[i].numpy())
            torch.testing.assert_close(
                actual, expected[key], rtol=rtol, atol=atol, check_dtype=False
            )
    else:
        actuals = torch.from_numpy(tvm_outputs[0].numpy())
        torch.testing.assert_close(
            actuals, expected, rtol=rtol, atol=atol, check_dtype=False
        )
    print("Outputs match between TVM Relax and PyTorch!")


if __name__ == "__main__":
    test_lfm2()
