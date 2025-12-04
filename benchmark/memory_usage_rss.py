import os

import numpy as np
import psutil
import tvm

proc = psutil.Process(os.getpid())


def main():
    tvm_device = tvm.cpu()
    example_args = (np.random.randn(1, 3, 224, 224).astype("float32"),)

    mem = proc.memory_info().rss
    mod = tvm.runtime.load_module("mobilenet_v3_small.so")
    vm = tvm.relax.VirtualMachine(mod, tvm_device)
    outputs = vm["main"](*example_args)
    diff = proc.memory_info().rss - mem
    print("Output shape:", outputs[0].shape)
    print(f"CPU memory usage: {diff / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
