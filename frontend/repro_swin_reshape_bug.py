"""Minimal repro for the swin_t dynamic reshape bug.

The issue: `_reshape` in TVM's ExportedProgram importer crashes with
    TypeError: 'NoneType' object is not iterable
when `self.shape_of(x)` returns None for a tensor whose struct_info.shape
is None.

Root cause:
  In swin_t's shifted window attention, the "unpad features" step
      x = x[:, :H, :W, :].contiguous()
  generates an identity slice on the dynamic batch dim:
      aten.slice.Tensor(x, dim=0, start=0, end=INT_MAX)
  TVM represents the result shape as `[T.min(INT_MAX, s77), ...]` instead of
  `[s77, ...]`.  A subsequent residual `add` between the original tensor
  (shape `[s77, ...]`) and the sliced tensor produces a result with
  `struct_info.shape = None` because the shapes don't unify.
  Any `view`/`reshape` on that result then calls `list(None)` and crashes.

Fix: guard the identity-reshape check in `_reshape` with
    `if current_shape is not None`.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.export import Dim, export

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


class PadRollCropAdd(torch.nn.Module):
    """Mimics swin_t's shifted window attention pattern:
    pad → cyclic shift (roll) → crop (unpad) → residual add → reshape.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        # Pad spatial dims (as in swin_t before window partitioning).
        padded = F.pad(x, (0, 0, 0, 1, 0, 1))
        # Cyclic shift (torch.roll on spatial dims, same as swin_t).
        rolled = torch.roll(padded, shifts=(-1, -1), dims=(1, 2))
        # Unpad / crop back to original spatial size.
        # The [:] on dim 0 generates aten.slice.Tensor(rolled, 0, 0, INT_MAX),
        # an identity slice on the dynamic batch dim.
        cropped = rolled[:, :H, :W, :]
        # Residual add — TVM can't unify shapes:
        #   x has [s, H, W, C] but cropped has [T.min(INT_MAX, s), H, W, C]
        #   → result struct_info.shape = None.
        out = x + cropped
        # View on the shape=None tensor triggers the crash.
        return out.view(B, H * W * C)


def main():
    model = PadRollCropAdd().eval()
    x = torch.randn(2, 4, 4, 2)

    batch = Dim("batch")
    exported_program = export(model, (x,), dynamic_shapes={"x": {0: batch}})

    mod = from_exported_program(exported_program)

    mod = relax.transform.DecomposeOpsForInference()(mod)
    target = tvm.target.Target("llvm")
    exe = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(exe, tvm.cpu())

    tvm_input = tvm.runtime.from_dlpack(x.contiguous())
    tvm_output = vm["main"](tvm_input)
    tvm_output_np = tvm_output.numpy() if hasattr(tvm_output, "numpy") else tvm_output[0].numpy()

    with torch.no_grad():
        torch_output = model(x).numpy()

    np.testing.assert_allclose(tvm_output_np, torch_output, rtol=1e-5, atol=1e-5)
    print("Numerical check passed")


if __name__ == "__main__":
    main()
