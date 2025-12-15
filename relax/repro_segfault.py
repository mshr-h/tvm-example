import torch
import torchvision
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

import tvm

tvm.support.describe()

torch_model = torchvision.models.resnet18(weights=None).eval()
example_args = (torch.randn(1, 3, 224, 224),)
exported_program = export(
    torch_model,
    args=example_args,
)

# Relax
target = tvm.target.Target(
    "llvm -keys=arm_cpu,cpu -mcpu=apple-m4 -mtriple=arm64-apple-darwin25.2.0"
)  # Apple M4 Pro target
print(f"vector_width: {tvm.get_global_func('target.llvm_get_vector_width')(target)}")

mod = from_exported_program(exported_program)
exe = tvm.compile(mod, target=target)
