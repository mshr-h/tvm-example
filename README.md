# exportedprogram-to-tvm-relax

Collections of PyTorch ExportedProgram to TVM Relax translation example.

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

## torchvision

```bash
python test_e2e_torchvision.py -v
```

## (wip) sam2

```bash
cd test_sam2
git clone https://github.com/facebookresearch/sam2.git sam2_repo
cd sam2_repo
uv pip install -e .
cd ../checkpoints/
bash download_ckpts.sh
cd ../../
uv run test_sam2.py
```

- `AssertionError: Unsupported function types ['sym_size.int', 'upsample_bicubic2d.vec', 'mul']`
