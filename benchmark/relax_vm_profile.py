import argparse

import numpy as np
import tvm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the VM on",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="mobilenet_v3_small.so",
        help="Path to the compiled Relax module",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        tvm_device = tvm.cpu()
    else:
        tvm_device = tvm.cuda()

    example_args = (np.random.randn(1, 3, 224, 224).astype("float32"),)
    mod = tvm.runtime.load_module(args.module)
    vm = tvm.relax.VirtualMachine(mod, tvm_device, profile=True)
    report = vm.profile("main", *example_args)
    print(report)


if __name__ == "__main__":
    main()
