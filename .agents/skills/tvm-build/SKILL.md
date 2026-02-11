---
name: tvm-build
description: Rebuild Apache TVM from source by invoking the provided build script.
compatibility:
  os:
    - linux
  shell:
    - bash
---

This skill rebuilds Apache TVM from source by calling the existing build script. It does not reimplement build steps.

Script
- Single source of truth: [.agents/skills/tvm-build/build-tvm.sh](.agents/skills/tvm-build/build-tvm.sh)

Usage
```bash
.agents/skills/tvm-build/build-tvm.sh [--clean] [--cuda] [--msc] [--papi] [--llvm ON|OFF]
```

Defaults and options
- LLVM is expected to be enabled (`--llvm ON`) unless you explicitly disable it. You can also pass an llvm-config path or name (for example, `--llvm path/to/llvm-config` or `--llvm llvm-config-20`).
- Pass `--clean` to remove the build directory before rebuilding.
- Use `--cuda`, `--msc`, and `--papi` to enable optional features supported by the script.

Requirements
- git, cmake, ninja, python3, and either `uv` or `pip` available on PATH
- Network access to clone/update TVM submodules
- Optional LLVM packages if `--llvm ON` is used (the script references the apt.llvm.org installer)

Notes
- The script will clone Apache TVM if the source directory does not exist.
- For detailed build behavior, refer to the script itself.
