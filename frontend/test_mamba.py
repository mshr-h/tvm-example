import torch
import torch.nn as nn
import torch.nn.functional as F

from tvm.relax.frontend.torch import from_exported_program

class MambaLikeBlock(nn.Module):
    """
    Mambaの雰囲気：
    - 入力依存の dt (selective)
    - 対角AのSSMをscanで更新（線形時間）
    - depthwise 1D convで局所混合
    - gatingで出力制御
    """

    def __init__(self, d_model: int, d_state: int = 16, conv_kernel: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # (1) in_proj: uとgate用vに分ける
        self.in_proj = nn.Linear(d_model, 2 * d_model)

        # (2) depthwise conv（局所情報の混合）
        self.dwconv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            groups=d_model,
            padding=conv_kernel - 1,  # causalっぽくするため後で切る
        )

        # (3) selectiveなパラメータ生成
        # dt: 入力に応じて変化（softplusで正）
        self.dt_proj = nn.Linear(d_model, d_model)

        # B, C: 入力から生成（超簡略版。実物はもっと工夫が多い）
        self.B_proj = nn.Linear(d_model, d_model * d_state)
        self.C_proj = nn.Linear(d_model, d_model * d_state)

        # (4) SSMの学習パラメータ
        # Aは安定化のため負にしたいので -exp(A_log) にする（対角A）
        self.A_log = nn.Parameter(torch.zeros(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))  # skip/残差係数

        # (5) out_proj
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        """
        B, L, Dm = x.shape

        # ---- 1) in_proj & gate ----
        u, v = self.in_proj(x).chunk(2, dim=-1)  # (B, L, d_model) each

        # ---- 2) depthwise conv（causalっぽく）----
        # Conv1d expects (B, C, L)
        u_conv = self.dwconv(u.transpose(1, 2))  # (B, d_model, L + k - 1)
        u_conv = u_conv[:, :, :L].transpose(1, 2)  # (B, L, d_model)

        # ---- 3) selective params (dt, B, C) ----
        dt = F.softplus(self.dt_proj(u_conv))  # (B, L, d_model), >0

        B_t = self.B_proj(u_conv).view(B, L, Dm, self.d_state)  # (B, L, d_model, d_state)
        C_t = self.C_proj(u_conv).view(B, L, Dm, self.d_state)  # (B, L, d_model, d_state)

        # ---- 4) SSM scan ----
        # A: (d_model, d_state)
        A = -torch.exp(self.A_log)

        # 状態 s: (B, d_model, d_state)
        s = torch.zeros(B, Dm, self.d_state, device=x.device, dtype=x.dtype)

        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_model, 1)
            u_t = u_conv[:, t, :].unsqueeze(-1)  # (B, d_model, 1)
            B_tt = B_t[:, t, :, :]  # (B, d_model, d_state)
            C_tt = C_t[:, t, :, :]  # (B, d_model, d_state)

            # 離散化の超簡略： s = s * exp(dt*A) + dt * (B * u)
            # exp(dt*A): (B, d_model, d_state)
            exp_Adt = torch.exp(dt_t * A.unsqueeze(0))  # broadcast
            s = s * exp_Adt + dt_t * (B_tt * u_t)  # u_t broadcast to d_state

            # 出力 y = sum(C * s) + D*u
            y = (C_tt * s).sum(dim=-1) + self.D.unsqueeze(0) * u_conv[:, t, :]
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, d_model)

        # ---- 5) gating & out ----
        y = y * torch.sigmoid(v)
        return self.out_proj(y)


if __name__ == "__main__":
    B, L, d_model = 2, 128, 256
    x = torch.randn(B, L, d_model)

    block = MambaLikeBlock(d_model=d_model, d_state=16, conv_kernel=4)
    block_compiled = torch.compile(block)
    y = block(x)

    # would like to export to ExportedProgram
    ep = torch.export.export(block, (x,))

    y_ep = ep.module()(x)

    assert torch.allclose(y, y_ep, atol=1e-6)

    mod = from_exported_program(ep)
    print(mod)
