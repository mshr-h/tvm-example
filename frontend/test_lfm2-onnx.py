import os
from typing import Dict, List

import numpy as np
import onnx
import torch
import tvm
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.runtime import Tensor as TVMTensor

HF_ONNX_REPO = "onnx-community/LFM2-350M-ENJP-MT-ONNX"
HF_PT_REPO = "LiquidAI/LFM2-350M-ENJP-MT"
DEFAULT_PROMPT = "Translate English to Japanese: The weather is nice today."

PAST_CONV_INPUTS = [
    "past_conv.0",
    "past_conv.1",
    "past_conv.3",
    "past_conv.4",
    "past_conv.6",
    "past_conv.7",
    "past_conv.9",
    "past_conv.11",
    "past_conv.13",
    "past_conv.15",
]
PAST_KEY_VALUE_INDICES = [2, 5, 8, 10, 12, 14]


def parse_args() -> dict:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the PyTorch reference on. Relax currently runs on CPU.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt used to create test tokens",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for both models",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-3,
        help="Relative tolerance for output comparison",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-1,
        help="Absolute tolerance for output comparison",
    )
    return vars(parser.parse_args())


def _sanitize(name: str) -> str:
    return name.replace(".", "_")


def _build_shape_dict(batch_size: int, seq_len: int, past_seq_len: int) -> Dict[str, List[int]]:
    shape_dict: Dict[str, List[int]] = {
        "input_ids": [batch_size, seq_len],
        "attention_mask": [batch_size, seq_len],
    }
    shape_dict.update({name: [batch_size, 1024, 3] for name in PAST_CONV_INPUTS})
    for idx in PAST_KEY_VALUE_INDICES:
        shape = [batch_size, 8, past_seq_len, 64]
        shape_dict[f"past_key_values.{idx}.key"] = shape
        shape_dict[f"past_key_values.{idx}.value"] = shape
    return shape_dict


def _prepare_cache_inputs(batch_size: int, past_seq_len: int) -> Dict[str, np.ndarray]:
    cache_inputs: Dict[str, np.ndarray] = {}
    for name in PAST_CONV_INPUTS:
        cache_inputs[_sanitize(name)] = np.zeros((batch_size, 1024, 3), dtype=np.float16)
    for idx in PAST_KEY_VALUE_INDICES:
        shape = (batch_size, 8, past_seq_len, 64)
        cache_inputs[_sanitize(f"past_key_values.{idx}.key")] = np.zeros(shape, dtype=np.float16)
        cache_inputs[_sanitize(f"past_key_values.{idx}.value")] = np.zeros(shape, dtype=np.float16)
    return cache_inputs


def _build_relax_vm(device: str, batch_size: int, seq_len: int) -> tuple:
    onnx_path = hf_hub_download(
        repo_id=HF_ONNX_REPO,
        filename="onnx/model_fp16.onnx",
    )
    onnx_data_path = hf_hub_download(
        repo_id=HF_ONNX_REPO,
        filename="onnx/model_fp16.onnx_data",
    )

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    onnx.external_data_helper.load_external_data_for_model(
        onnx_model, os.path.dirname(onnx_data_path)
    )

    past_seq_len = seq_len
    shape_dict = _build_shape_dict(batch_size, seq_len, past_seq_len)
    dtype_dict = {
        "input_ids": "int32",
        "attention_mask": "int32",
    }

    print("Converting ONNX model to Relax IR...")
    mod, _ = from_onnx(
        onnx_model,
        shape_dict=shape_dict,
        dtype_dict=dtype_dict,
        keep_params_in_input=False,
        sanitize_input_names=True,
    )
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)

    tvm_device = tvm.cpu() if device == "cpu" else tvm.cuda()
    target = tvm.target.Target.from_device(tvm_device)
    print("Compiling Relax module...")
    exe = tvm.compile(mod, target, relax_pipeline="default")
    vm = relax.VirtualMachine(exe, tvm_device)
    return vm, mod, tvm_device


def _prepare_vm_inputs(
    mod: tvm.IRModule, torch_inputs: Dict[str, torch.Tensor], seq_len: int, tvm_device
) -> List[TVMTensor]:
    batch_size = torch_inputs["input_ids"].shape[0]
    cache_data = _prepare_cache_inputs(batch_size, seq_len)

    numpy_inputs: Dict[str, np.ndarray] = {
        "input_ids": torch_inputs["input_ids"].to(torch.int32).cpu().numpy(),
        "attention_mask": torch_inputs["attention_mask"].to(torch.int32).cpu().numpy(),
    }
    numpy_inputs.update(cache_data)

    vm_inputs: List[tvm.nd.NDArray] = []
    for var in mod["main"].params:
        name = var.name_hint
        if name not in numpy_inputs:
            raise KeyError(f"Missing value for Relax input '{name}'")
        vm_inputs.append(tvm.runtime.tensor(numpy_inputs[name], tvm_device))
    return vm_inputs


def _run_pytorch_reference(prompt: str, seq_len: int, device: torch.device) -> tuple:
    print(f"Running PyTorch reference model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(HF_PT_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        HF_PT_REPO,
        torch_dtype=torch.float16,
        device_map=None,
    )
    model.to(device).eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    logits = outputs.logits.detach().cpu()
    inputs_cpu = {k: v.detach().cpu() for k, v in inputs.items()}
    print(f"PyTorch logits shape: {tuple(logits.shape)}")
    return logits, inputs_cpu


def main():
    args_dict = parse_args()
    if args_dict["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    torch_device = torch.device(args_dict["device"])
    pt_logits, torch_inputs = _run_pytorch_reference(
        args_dict["prompt"], args_dict["seq_len"], torch_device
    )

    relax_device = args_dict["device"]
    if relax_device == "cuda":
        print(
            "Relax CUDA target is not yet supported for this model (concat cache kernel). "
            "Falling back to CPU for TVM execution."
        )
        relax_device = "cpu"

    vm, mod, tvm_device = _build_relax_vm(
        relax_device, torch_inputs["input_ids"].shape[0], args_dict["seq_len"]
    )
    vm_inputs = _prepare_vm_inputs(mod, torch_inputs, args_dict["seq_len"], tvm_device)

    print(f"Running Relax VM on {relax_device}...")
    vm_outputs = vm["main"](*vm_inputs)
    if hasattr(vm_outputs, "numpy"):
        relax_logits = vm_outputs.numpy()
    elif isinstance(vm_outputs, (list, tuple)):
        relax_logits = vm_outputs[0].numpy()
    elif hasattr(vm_outputs, "__getitem__"):
        relax_logits = vm_outputs[0].numpy()
    else:
        raise TypeError(f"Unexpected Relax output type: {type(vm_outputs)}")
    print(f"Relax logits shape: {relax_logits.shape}")

    ref_logits = pt_logits.numpy()
    relax_tensor = torch.from_numpy(relax_logits)
    ref_tensor = torch.from_numpy(ref_logits)
    diff_tensor = (relax_tensor - ref_tensor).abs()
    max_diff = float(diff_tensor.max().item())
    mean_diff = float(diff_tensor.mean().item())
    try:
        torch.testing.assert_close(
            relax_tensor,
            ref_tensor,
            rtol=args_dict["rtol"],
            atol=args_dict["atol"],
            check_dtype=False,
        )
        print(
            f"Outputs match within tolerance (rtol={args_dict['rtol']}, atol={args_dict['atol']}). "
            f"Max abs diff: {max_diff:.4f}, mean abs diff: {mean_diff:.4f}"
        )
    except AssertionError as err:
        print("Outputs do not match within the requested tolerance.")
        print(f"Max abs diff: {max_diff:.4f}, mean abs diff: {mean_diff:.4f}")
        print(str(err))


if __name__ == "__main__":
    main()
