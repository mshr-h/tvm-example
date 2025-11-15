import torch
from torch.export import export, Dim
import pytest
from typing import Optional

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
    rtol=1e-4,
    atol=1e-4,
    equal_nan=True,
):
    if target is None:
        target = tvm.target.Target.from_device(dev)

    # PyTorch
    exported_program = export(
        torch_model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    expected: torch.Tensor = exported_program.module()(*example_args)

    # Relax
    mod = from_exported_program(exported_program)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    exe = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    tvm_args = [tvm.runtime.from_dlpack(x.contiguous()) for x in example_args]
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


def verify_torchvision_model(model_name: str, is_dynamic: bool = False):
    from tvm.contrib.download import download_testdata
    from torchvision.models import get_model, get_model_weights
    from torchvision.io import read_image

    # prepare sample image
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_name = "cat.png"
    img_path = download_testdata(img_url, img_name, module="data")
    image_tensor = read_image(img_path)

    model = get_model(model_name, weights="DEFAULT").eval()
    weights = get_model_weights(model_name).DEFAULT
    transforms = weights.transforms()
    example = transforms(image_tensor).unsqueeze(0)

    if is_dynamic:
        example = example.expand(2, -1, -1, -1)
        example_args = (example,)
        batch = Dim("batch")
        dynamic_shapes = {"x": {0: batch}}
    else:
        example_args = (example,)
        dynamic_shapes = None

    verify_model(model, example_args, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "is_dynamic",
    [True, False],
    ids=["dynamic", "static"],
)
@pytest.mark.parametrize(
    "torch_model_name",
    [
        # classification models
        "alexnet",
        "convnext_tiny",
        "densenet121",
        "efficientnet_b0",
        "efficientnet_v2_s",
        "inception_v3",
        "maxvit_t",
        "mnasnet0_5",
        "mobilenet_v2",
        "mobilenet_v3_small",
        "regnet_x_400mf",
        "resnet18",
        "resnext50_32x4d",
        "shufflenet_v2_x0_5",
        "squeezenet1_0",
        "swin_t",
        "swin_v2_t",
        "vgg11",
        "vgg11_bn",
        "vit_b_32",
        "wide_resnet50_2",
        # quantized models
        "quantized_googlenet",
        "quantized_inception_v3",
        "quantized_mobilenet_v2",
        "quantized_mobilenet_v3_large",
        "quantized_resnet18",
        "quantized_resnext101_32x8d",
        "quantized_shufflenet_v2_x0_5",
        # segmentation models
        "deeplabv3_mobilenet_v3_large",
        "deeplabv3_resnet50",
        "fcn_resnet50",
        "lraspp_mobilenet_v3_large",
    ],
)
def test_e2e(torch_model_name: str, is_dynamic: bool):
    verify_torchvision_model(torch_model_name, is_dynamic)


if __name__ == "__main__":
    tvm.testing.main()
