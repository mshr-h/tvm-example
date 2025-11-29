import torch
from torch.export import export, Dim
import pytest
from typing import Optional
import copy

import tvm
from tvm import relax
import tvm.testing
from tvm.relax.frontend.torch import from_exported_program


def verify_model(
    torch_model,
    example_args,
    example_kwargs={},
    dynamic_shapes=None,
    target: Optional[str] = None,
    dev=tvm.cpu(),
    rtol=None,
    atol=None,
    equal_nan=True,
    verbose=False,
):
    if target is None:
        target = tvm.target.Target.from_device(dev)

    original_example_args = copy.deepcopy(example_args)

    # PyTorch
    exported_program = export(
        torch_model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    if verbose:
        print("Exported Program:")
        print(exported_program)
    expected: torch.Tensor = exported_program.module()(*example_args)

    # Relax
    mod = from_exported_program(exported_program)
    if verbose:
        print("Relax Module:")
        print(mod)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    exe = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    tvm_args = [tvm.runtime.from_dlpack(x.contiguous()) for x in original_example_args]
    tvm_outputs = vm["main"](*tvm_args)

    # check if the outputs match
    if isinstance(expected, dict):
        for i, key in enumerate(expected.keys()):
            actual = torch.from_numpy(tvm_outputs[i].numpy())
            torch.testing.assert_close(
                actual, expected[key], rtol=rtol, atol=atol, equal_nan=equal_nan
            )
    else:
        actuals = torch.from_numpy(tvm_outputs[0].numpy())
        torch.testing.assert_close(
            actuals, expected, rtol=rtol, atol=atol, equal_nan=equal_nan
        )


operator_basic_unary = [
    (torch.ops.aten.abs),
    # (torch.ops.aten.acos), # AssertionError: Tensor-likes are not close!
    (torch.ops.aten.acosh),
    # (torch.ops.aten.asin), # AssertionError: Tensor-likes are not close!
    (torch.ops.aten.asinh),
    # (torch.ops.aten.atan), # AssertionError: Tensor-likes are not close!
    (torch.ops.aten.atanh),
    (torch.ops.aten.ceil),
    (torch.ops.aten.cos),
    (torch.ops.aten.cosh),
    (torch.ops.aten.erf),
    (torch.ops.aten.exp),
    (torch.ops.aten.floor),
    (torch.ops.aten.gelu),
    (torch.ops.aten.hardsigmoid),
    (torch.ops.aten.hardswish),
    (torch.ops.aten.log),
    (torch.ops.aten.neg),
    (torch.ops.aten.relu),
    (torch.ops.aten.round),
    (torch.ops.aten.rsqrt),
    (torch.ops.aten.sigmoid),
    (torch.ops.aten.sin),
    (torch.ops.aten.sinh),
    (torch.ops.aten.sign),
    (torch.ops.aten.sqrt),
    (torch.ops.aten.tan),
    (torch.ops.aten.tanh),
    (torch.ops.aten.trunc),
]


@pytest.mark.parametrize("pytorch_op", operator_basic_unary)
def test_basic_unary_ops(pytorch_op):
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    class UnaryOp(torch.nn.Module):
        def forward(self, input):
            return pytorch_op(input)

    verify_model(UnaryOp().eval(), example_args)


def test_dropout():
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    class DropoutModel(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.dropout(x, p=0.5, train=False)

    verify_model(DropoutModel().eval(), example_args)


operator_binary = [
    (torch.ops.aten.add),
    (torch.ops.aten.sub),
    (torch.ops.aten.mul),
    (torch.ops.aten.divide),
    (torch.ops.aten.fmod),
    (torch.ops.aten.pow),
]


@pytest.mark.parametrize("pytorch_op", operator_binary)
def test_binary_op(pytorch_op):
    example_args = (
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
    )

    class BinaryOp(torch.nn.Module):
        def forward(self, input1, input2):
            return pytorch_op(input1, input2)

    verify_model(BinaryOp().eval(), example_args)


operator_binary_inplace = [
    (torch.ops.aten.add_),
    (torch.ops.aten.mul_),
]


@pytest.mark.parametrize("pytorch_op", operator_binary_inplace)
def test_binary_op_inplace(pytorch_op):
    example_args = (
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
    )
    example_args = (
        torch.tensor([1, 2, 3], dtype=torch.float32),
        torch.tensor([4, 5, 6], dtype=torch.float32),
    )

    class BinaryOp(torch.nn.Module):
        def forward(self, input1, input2):
            return pytorch_op(input1, input2)

    verify_model(BinaryOp().eval(), example_args)


def test_conv2d():
    example_args = (
        torch.randn(1, 3, 32, 32, dtype=torch.float32),
        torch.randn(16, 3, 3, 3, dtype=torch.float32),
    )

    class Conv2dAten(torch.nn.Module):
        def forward(self, x, weight):
            return torch.ops.aten.conv2d(
                x,
                weight,
                bias=None,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
            )

    class Conv2DModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv2DGrouped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 4, 3, groups=2, bias=False)

        def forward(self, input):
            return self.conv(input)

    class Conv2DStrided(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 3, stride=2, bias=True)

        def forward(self, input):
            return self.conv(input)

    verify_model(Conv2dAten().eval(), example_args)
    verify_model(
        Conv2DModule().eval(), (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    )
    verify_model(
        Conv2DGrouped().eval(), (torch.randn(1, 4, 10, 10, dtype=torch.float32),)
    )
    verify_model(
        Conv2DStrided().eval(), (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    )


def test_batch_norm():
    example_args = (
        torch.randn(1, 3, 32, 32, dtype=torch.float32),
        torch.randn(3, dtype=torch.float32),
        torch.randn(3, dtype=torch.float32),
        torch.randn(3, dtype=torch.float32),
        torch.randn(3, dtype=torch.float32),
    )

    class BatchNormAten(torch.nn.Module):
        def forward(self, x, weight, bias, running_mean, running_var):
            return torch.ops.aten.batch_norm(
                x,
                running_mean,
                running_var,
                weight,
                bias,
                training=False,
                momentum=0.1,
                eps=1e-5,
                cudnn_enabled=False,
            )

    class BatchNormFunctional(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, input):
            return torch.nn.functional.batch_norm(
                input,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
            )

    verify_model(BatchNormAten().eval(), example_args)
    verify_model(
        BatchNormFunctional().eval(), (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    )


def test_adaptive_avg_pool2d():
    example_args = (torch.randn(1, 3, 32, 32, dtype=torch.float32),)

    class AdaptiveAvgPool2dAten(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.adaptive_avg_pool2d(x, output_size=(1, 1))

    verify_model(AdaptiveAvgPool2dAten().eval(), example_args)


def test_linear():
    example_args = (
        torch.randn(1, 10, dtype=torch.float32),
        torch.randn(5, 10, dtype=torch.float32),
        torch.randn(5, dtype=torch.float32),
    )

    class LinearAten(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.ops.aten.linear(x, weight, bias)

    verify_model(LinearAten().eval(), example_args)


def test_flatten():
    example_args = (torch.randn(2, 3, 4, 5, dtype=torch.float32),)

    class FlattenAten(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.flatten(x, 1)

    verify_model(FlattenAten().eval(), example_args)


def test_dynamic_output():
    class DynamicReshape(torch.nn.Module):
        def forward(self, x):
            # Get the batch size dynamically - this creates sym_size.int calls
            batch_size = x.shape[0]
            return x.reshape(batch_size, -1)

    example_args = (torch.randn(3, 4, 5),)
    dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}
    verify_model(
        DynamicReshape().eval(),
        example_args,
        dynamic_shapes=dynamic_shapes,
    )

    class Flatten(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.flatten()

    example_args = (torch.randn(3, 4, 5),)
    dynamic_shapes = {
        "x": {
            0: torch.export.Dim("batch"),
            1: torch.export.Dim("height"),
            2: torch.export.Dim("width"),
        }
    }
    verify_model(Flatten().eval(), example_args, dynamic_shapes=dynamic_shapes)

def test_register_buffer():
    class ModelWithBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("my_buffer", torch.randn(3, 4), persistent=False)

        def forward(self, x):
            return x + self.my_buffer

    example_args = (torch.randn(2, 3, 4),)
    verify_model(ModelWithBuffer().eval(), example_args, verbose=True)


if __name__ == "__main__":
    tvm.testing.main()
