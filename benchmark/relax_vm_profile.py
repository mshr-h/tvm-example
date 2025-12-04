import numpy as np
import tvm


def main():
    tvm_device = tvm.cpu()
    example_args = (np.random.randn(1, 3, 224, 224).astype("float32"),)

    mod = tvm.runtime.load_module("mobilenet_v3_small.so")
    vm = tvm.relax.VirtualMachine(mod, tvm_device, profile=True)
    report = vm.profile("main", *example_args)
    print(report)


if __name__ == "__main__":
    main()
