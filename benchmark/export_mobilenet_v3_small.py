import argparse

import torch
import torchvision
import tvm
from tvm.relax.frontend.torch import from_exported_program


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="mobilenet_v3_small.so",
        help="Path to save the compiled Relax module",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to compile for",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        tvm_device = tvm.cpu()
    else:
        tvm_device = tvm.cuda()
    tvm_target = tvm.target.Target.from_device(tvm_device)

    model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1").eval()
    epxample_args = (torch.randn(1, 3, 224, 224),)
    exported_program = torch.export.export(model, epxample_args)

    mod = from_exported_program(exported_program)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    exe = tvm.compile(mod, target=tvm_target)
    exe.export_library(args.output)


if __name__ == "__main__":
    main()
