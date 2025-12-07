import argparse
import json
import math
from dataclasses import dataclass

import torch
import tvm
import tvm.testing.utils
from huggingface_hub import hf_hub_download
from tvm import dlight, relax
from tvm.relax import register_pipeline
from tvm.relax.frontend.torch import from_exported_program


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


def norm(x):
    return torch.nn.functional.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = torch.nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = torch.nn.Linear(
            self.n_embd, self.n_kv_head * self.head_dim, bias=False
        )
        self.c_v = torch.nn.Linear(
            self.n_embd, self.n_kv_head * self.head_dim, bias=False
        )
        self.c_proj = torch.nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)
        if kv_cache is None or Tq == Tk:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
        elif Tq == 1:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False
            )
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask
            )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = torch.nn.functional.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = torch.nn.ModuleDict(
            {
                "wte": torch.nn.Embedding(config.vocab_size, config.n_embd),
                "h": torch.nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.transformer.wte.to(dtype=torch.bfloat16)

    def init_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0 : T0 + T], self.sin[:, T0 : T0 + T]
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        return logits


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


def test_nanochat():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    repo_id = "sdobson/nanochat"
    model_file = "model_000650.pt"
    meta_file = "meta_000650.json"

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False"
            )
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")

    print("Downloading model and metadata...")
    local_pt_path = hf_hub_download(repo_id=repo_id, filename=model_file)
    local_meta_path = hf_hub_download(repo_id=repo_id, filename=meta_file)

    with open(local_meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    model_config_kwargs = meta_data["model_config"]
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)

    print("Loading model weights...")
    model_data = torch.load(local_pt_path, map_location="cpu", weights_only=True)
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    model_data = {
        k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()
    }

    model.to_empty(device="cpu")
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    class GPTForExport(torch.nn.Module):
        def __init__(self, gpt):
            super().__init__()
            self.gpt = gpt

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            return self.gpt(idx)

    export_model = GPTForExport(model).to("cpu")
    export_model.eval()

    print("Exporting model to ExportedProgram...")
    B = 1
    T = model_config.sequence_len
    example_args = (torch.zeros((B, T), dtype=torch.long, device="cpu"),)

    exported_program = torch.export.export(
        export_model,
        example_args,
    )

    # rms_norm is not yet supported in TVM, so we define custom converter
    from tvm.relax.frontend.torch.exported_program_translator import (
        ExportedProgramImporter,
    )

    def _rms_norm(node: torch.fx.Node, self: ExportedProgramImporter) -> relax.Var:
        x = self.env[node.args[0]]
        torch_dtype = node.args[0].meta["tensor_meta"].dtype
        normalized_shape = node.args[1]
        weight = self.env.get(node.args[2], None) if len(node.args) > 2 else None
        eps = node.args[3] if len(node.args) > 3 else None

        N = len(self.shape_of(x))
        D = len(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else 1
        axes = list(range(N - D, N))

        if weight is None:
            weight = self._convert_torch_tensor_to_relax(
                torch.ones(list(normalized_shape), dtype=torch_dtype)
            )
        eps = torch.finfo(torch_dtype).eps if eps is None else 0.00001

        return self.block_builder.emit(relax.op.nn.rms_norm(x, weight, axes, eps))

    # Relax
    tvm_device = tvm.cpu() if args.device == "cpu" else tvm.cuda()
    target = tvm.target.Target.from_device(tvm_device)
    print("Converting ExportedProgram to Relax model...")
    mod = from_exported_program(
        exported_program,
        custom_convert_map={"rms_norm.default": _rms_norm},
        run_ep_decomposition=False,
    )
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)

    print("Compiling Relax module...")
    exe = tvm.compile(mod, target, relax_pipeline="opt_llm")

    print("Exporting TVM executable...")
    exe.export_library("nanochat_tvm_executable.so")

    print(f"Running Relax model on {args.device}...")
    vm = relax.VirtualMachine(exe, tvm_device)
    tvm_args = [
        tvm.runtime.from_dlpack(x.contiguous().to(torch_device)) for x in example_args
    ]
    tvm_outputs = vm["main"](*tvm_args)

    # PyTorch
    print("Running PyTorch model on cpu...")
    expected: torch.Tensor = exported_program.module()(*example_args)

    # check if the outputs match
    rtol = 1e-4
    atol = 1e-4
    if isinstance(expected, dict):
        for i, key in enumerate(expected.keys()):
            actual = torch.from_numpy(tvm_outputs[i].numpy())
            tvm.testing.utils.assert_allclose(
                actual.detach().numpy(),
                expected[key].detach().numpy(),
                rtol=rtol,
                atol=atol,
            )
    else:
        actuals = torch.from_numpy(tvm_outputs[0].numpy())
        tvm.testing.utils.assert_allclose(
            actuals.detach().numpy(), expected.detach().numpy(), rtol=rtol, atol=atol
        )
    print("Outputs match between TVM Relax and PyTorch!")

    print(f"Benchmarking Relax model on {args.device}...")
    report = vm.time_evaluator("main", tvm_device, number=5, repeat=3)(*tvm_args)
    print(report)


if __name__ == "__main__":
    test_nanochat()
