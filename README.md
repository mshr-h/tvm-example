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
./build-tvm.sh --clean --llvm llvm-config # change if you want to use different llvm version
```

## tvm dev

```basah
uv venv
source .venv/bin/activate
cd tvm

# build docker image
docker/build.sh ci_cpu

# run test on docker image
docker/bash.sh tlcpack/ci-cpu:20260214-152058-2a448ce4 ./tests/scripts/task_config_build_cpu.sh build
docker/bash.sh tlcpack/ci-cpu:20260214-152058-2a448ce4 python3 ./tests/scripts/task_build.py --build-dir build
docker/bash.sh tlcpack/ci-cpu:20260214-152058-2a448ce4 python3 ./tests/scripts/task_build.py --cmake-target cpptest --build-dir build
docker/bash.sh tlcpack/ci-cpu:20260214-152058-2a448ce4 ./tests/scripts/task_python_unittest.sh

# use local ci script
python tests/scripts/ci.py cpu -d tvm.ci_cpu

# lint
docker/bash.sh tlcpack/ci-lint:20260214-152058-2a448ce4 ./tests/scripts/task_lint.sh
```
