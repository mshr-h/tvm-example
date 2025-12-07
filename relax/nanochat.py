import argparse
import dataclasses
import json
import math

import numpy as np
import torch
import tvm
from huggingface_hub import hf_hub_download
from tvm import dlight, relax
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


@dataclasses.dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 65536
    n_layer: int = 20
    n_head: int = 10
    n_kv_head: int = 10
    n_embd: int = 1280


def rms_norm(x: Tensor, eps: float = 1e-5) -> Tensor:
    # x: (..., C)
    # rms = sqrt(mean(x^2, dim=-1, keepdims=True) + eps)
    # y = x / rms
    x2 = op.multiply(x, x)
    # 最後の次元で平均
    denom = nn.Tensor.from_scalar(x.shape[-1], dtype=x.dtype)
    mean = op.divide(op.sum(x2, axis=-1, keepdims=True), denom)
    eps_tensor = nn.Tensor.from_scalar(eps, dtype=mean.dtype)
    rms = op.sqrt(op.add(mean, eps_tensor))
    return op.divide(x, rms)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, H, T, D)
    # cos/sin: (1, 1, T, D/2) を想定
    b, h, t, d = x.shape

    x1, x2 = op.split(x, 2, axis=-1)  # (..., D/2), (..., D/2)
    # y1 = x1 * cos + x2 * sin
    # y2 = -x1 * sin + x2 * cos
    y1 = op.add(op.multiply(x1, cos), op.multiply(x2, sin))
    y2 = op.add(op.multiply(op.negative(x1), sin), op.multiply(x2, cos))
    return op.concat((y1, y2), dim=-1)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    # x: (B, n_kv_heads, T, D) -> (B, n_kv_heads * n_rep, T, D)
    if n_rep == 1:
        return x
    return op.repeat(x, repeats=n_rep, axis=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # PyTorch 側と同じ名前で Linear を定義する
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def _causal_attention(
        self,
        q: Tensor,  # (B, H, T, D)
        k: Tensor,  # (B, H, T, D)
        v: Tensor,  # (B, H, T, D)
    ) -> Tensor:  # (B, H, T, D)
        b, h, t, d = q.shape

        # (B*H, T, D)
        q2 = op.reshape(q, (b * h, t, d))
        k2 = op.reshape(k, (b * h, t, d))
        v2 = op.reshape(v, (b * h, t, d))

        # scores: (B*H, T, T)
        scores = op.matmul(q2, op.permute_dims(k2, [0, 2, 1]))
        scale = 1.0 / math.sqrt(d)
        scale_tensor = nn.Tensor.from_scalar(scale, dtype=scores.dtype)
        scores = op.multiply(scores, scale_tensor)

        # causal mask: j > i を -1e9 に
        # mask_base: (T, T) 上三角（対角より上）が 1
        mask_base = nn.triu(nn.ones((t, t), dtype=scores.dtype), diagonal=1)
        mask_base = op.reshape(mask_base, (1, t, t))
        # broadcast to (B*H, T, T)
        mask = op.multiply(nn.ones((b * h, t, t), dtype=scores.dtype), mask_base)
        neg_large = nn.full(
            (1,),
            nn.Tensor.from_scalar(-1e9, dtype=scores.dtype),
            dtype=scores.dtype,
        )
        scores = op.add(scores, op.multiply(mask, neg_large))

        # softmax over last dim
        attn = nn.softmax(scores, axis=-1)

        # out: (B*H, T, D)
        out = op.matmul(attn, v2)
        # (B, H, T, D)
        out = op.reshape(out, (b, h, t, d))
        return out

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # x: (B, T, C)
        b, t, c = x.shape

        # QKV
        q = self.c_q(x)  # (B, T, H*D)
        k = self.c_k(x)  # (B, T, n_kv*D)
        v = self.c_v(x)  # (B, T, n_kv*D)

        q = op.reshape(q, (b, t, self.n_head, self.head_dim))
        k = op.reshape(k, (b, t, self.n_kv_head, self.head_dim))
        v = op.reshape(v, (b, t, self.n_kv_head, self.head_dim))

        # RoPE + RMSNorm on Q,K
        # 形を (B, H, T, D) に並べ替えてから RoPE
        q = op.permute_dims(q, [0, 2, 1, 3])
        k = op.permute_dims(k, [0, 2, 1, 3])

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = rms_norm(q)
        k = rms_norm(k)

        # KV を MQA 用に複製
        # k/v: (B, n_kv, T, D)
        # → repeat_kv → (B, H, T, D)
        k = repeat_kv(k, self.n_head // self.n_kv_head)
        v = repeat_kv(
            op.permute_dims(v, [0, 2, 1, 3]),
            self.n_head // self.n_kv_head,
        )

        # 注意：上で v を (B, n_kv, T, D) に permute してから repeat_kv する

        # Attention
        y = self._causal_attention(q, k, v)  # (B, H, T, D)

        # (B, T, C)
        y = op.permute_dims(y, [0, 2, 1, 3])
        y = op.reshape(y, (b, t, c))

        # 出力 projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """ReLU^2 MLP"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = op.multiply(nn.relu(x), nn.relu(x))  # ReLU(x)^2
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # x + Attn(norm(x))
        x = op.add(x, self.attn(rms_norm(x), cos, sin))
        # x + MLP(norm(x))
        x = op.add(x, self.mlp(rms_norm(x)))
        return x


def _tvm_dtype_to_torch(dtype: str) -> torch.dtype:
    """Map TVM dtype strings to torch dtypes used for parameters."""
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype conversion: {dtype}")


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # PyTorch と同じ階層・名前
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # rotary embeddings を TVM 側で precompute
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # ここは PyTorch の register_buffer 相当だが，
        # TVM では単純に Tensor として持っておく
        self.cos = cos
        self.sin = sin

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: int = 10000
    ):
        # numpy で cos/sin を作り，Tensor.from_const で埋め込む
        channel_range = np.arange(0, head_dim, 2, dtype="float32")
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = np.arange(seq_len, dtype="float32")
        freqs = np.outer(t, inv_freq)
        cos = np.cos(freqs)
        sin = np.sin(freqs)
        # (1, 1, seq_len, head_dim/2)
        cos = cos[None, None, :, :].astype("float32")
        sin = sin[None, None, :, :].astype("float32")
        cos_tensor = Tensor.from_const(cos)
        sin_tensor = Tensor.from_const(sin)
        return cos_tensor, sin_tensor

    def forward(
        self,
        idx: Tensor,  # (B, T) int32
        targets: Tensor | None = None,
    ) -> Tensor:
        # idx: token ids
        b, t = idx.shape

        # RoPE の長さチェック
        # ここでは単純に先頭 T ステップ分のみを使う
        positions = op.arange(0, t, dtype="int32")
        cos = op.take(self.cos, positions, axis=2)
        sin = op.take(self.sin, positions, axis=2)

        # Embedding
        x = self.transformer["wte"](idx)  # (B, T, C)
        x = rms_norm(x)
        cos = op.astype(cos, x.dtype)
        sin = op.astype(sin, x.dtype)

        # Blocks
        for block in self.transformer["h"]:
            x = block(x, cos, sin)

        x = rms_norm(x)

        # logits
        logits = self.lm_head(x)  # (B, T, V)

        # soft cap も PyTorch に合わせて入れておく
        softcap = 15.0
        logits = op.multiply(
            nn.Tensor.from_scalar(softcap, dtype=logits.dtype),
            op.tanh(
                op.divide(
                    logits,
                    nn.Tensor.from_scalar(softcap, dtype=logits.dtype),
                )
            ),
        )

        # TVM 版では loss 計算は外でやる想定にして
        # ここでは logits だけ返す
        return logits

    # Relax IR への export 用 spec
    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "idx": nn.spec.Tensor(["batch", self.config.sequence_len], "int32"),
                    "$": {
                        "param_mode": "packed",
                        "effect_mode": "none",
                    },
                },
            },
            self,
        )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run the model on (cpu or cuda)"
    )
    args = parser.parse_args()

    repo_id = "sdobson/nanochat"
    model_file = "model_000650.pt"
    meta_file = "meta_000650.json"

    local_meta_path = hf_hub_download(repo_id=repo_id, filename=meta_file)
    with open(local_meta_path) as f:
        model_config = json.load(f)["model_config"]
    tvm_config = GPTConfig(**model_config)
    tvm_model = GPT(tvm_config)

    dev = tvm.device(args.device, 0)  # "cpu" / "cuda"
    print("Using device:", dev)
    # tvm_model.to("float16")  # bfloat16 -> float16 などに揃える場合

    mod, named_params = tvm_model.export_tvm(spec=tvm_model.get_default_spec())
    named_params = dict(named_params)  # name -> Parameter

    local_pt_path = hf_hub_download(repo_id=repo_id, filename=model_file)
    torch_state = torch.load(local_pt_path, map_location="cpu", weights_only=True)

    print("Using config:", tvm_config)
    print("HF checkpoint wte shape:", torch_state["transformer.wte.weight"].shape)
    print("TVM named_params wte shape:", named_params["transformer.wte.weight"].shape)
    print("PyTorch keys example:         ", list(torch_state.keys())[:5])
    print("TVM named_params keys example:", list(named_params.keys())[:5])

    # 2) PyTorch Tensor -> numpy -> TVM NDArray
    param_ndarrays = []
    for name, param in named_params.items():
        # Align dtypes with the exported TVM module (HF checkpoint stores some weights in bf16).
        torch_tensor = torch_state[name].detach()
        desired_dtype = _tvm_dtype_to_torch(param.dtype)
        if torch_tensor.dtype != desired_dtype:
            torch_tensor = torch_tensor.to(desired_dtype)
        param_ndarrays.append(tvm.runtime.from_dlpack(torch_tensor))

    target = tvm.target.Target.from_device(dev)
    print("Target:", target)
    pipeline = relax.get_pipeline("opt_llm")

    print("Compiling with pipeline:", pipeline)
    with target:
        ex = tvm.compile(
            mod,
            target=target,
            relax_pipeline=pipeline,
        )
    vm = relax.VirtualMachine(ex, dev)

    # idx: (B, T) の int32 TVM tensor を用意
    print("Running inference...")
    idx_np = np.random.randint(
        0, tvm_config.vocab_size, size=(1, tvm_config.sequence_len), dtype="int32"
    )
    idx_nd = tvm.runtime.tensor(idx_np, device=dev)

    logits = vm["forward"](idx_nd, param_ndarrays)


if __name__ == "__main__":
    main()
