import torch
from model import GPT
from tvm.relax.frontend.torch import from_exported_program
import tvm
from tvm import relax
from torch.nn.attention import SDPBackend


def test_nanpgpt():
    model = GPT.from_pretrained("gpt2")
    example_args = (
        torch.randint(0, 100, (1, model.config.block_size), dtype=torch.long),
    )
    dynamic_shape = (
        {1: torch.export.Dim("token_dim", max=model.config.block_size)},
    )

    # PyTorch
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        exported_program = torch.export.export(model, example_args, dynamic_shapes=dynamic_shape)
    expected: torch.Tensor = exported_program.module()(*example_args)

    # Relax
    mod = from_exported_program(exported_program)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    dev = tvm.cpu()
    target = tvm.target.Target.from_device(dev)
    exe = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    tvm_args = [tvm.nd.from_dlpack(x.contiguous()) for x in example_args]
    tvm_outputs = vm["main"](*tvm_args)

    # check if the outputs match
    if isinstance(expected, dict):
        for i, key in enumerate(expected.keys()):
            actual = torch.from_numpy(tvm_outputs[i].numpy())
            torch.testing.assert_close(
                actual, expected[key], rtol=1e-4, atol=1e-4, equal_nan=True
            )
    else:
        actuals = torch.from_numpy(tvm_outputs[0].numpy())
        torch.testing.assert_close(
            actuals, expected, rtol=1e-4, atol=1e-4, equal_nan=True
        )

if __name__ == "__main__":
    test_nanpgpt()
