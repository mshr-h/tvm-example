import numpy as np
import tvm


def main():
    tvm_device = tvm.cpu()
    example_args = (np.random.randn(1, 3, 224, 224).astype("float32"),)

    mod = tvm.runtime.load_module("mobilenet_v3_small.so")
    vm = tvm.relax.VirtualMachine(mod, tvm_device)
    number = 10
    repeat = 3
    result = vm.time_evaluator("main", tvm_device, number=number, repeat=repeat)(
        *example_args
    )
    print(result)


if __name__ == "__main__":
    main()
