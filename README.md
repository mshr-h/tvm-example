# tvm-example

Collections of Apache TVM example.

## Prerequisite

- [uv](https://docs.astral.sh/uv/)
- llvm

## prepare venv and install tvm, pytorch

```bash
uv venv
source .venv/bin/activate
uv pip install cmake ninja setuptools cython pytest
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
cd 3rdparty
./build-tvm.sh --clean --llvm llvm-config # change if you want to use different llvm version
```
