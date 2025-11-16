import os
import sys
import json
import torch
from huggingface_hub import hf_hub_download
from tvm.relax.frontend.torch import from_exported_program
import tvm
from tvm import relax


def test_nanochat():
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "nanochat"))
    )

    from nanochat.gpt import GPT, GPTConfig

    repo_id = "karpathy/nanochat-d32"
    model_file = "model_000650.pt"
    meta_file = "meta_000650.json"

    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_pt_path = hf_hub_download(repo_id=repo_id, filename=model_file)
    local_meta_path = hf_hub_download(repo_id=repo_id, filename=meta_file)

    with open(local_meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    cfg = GPTConfig(**meta_data["model_config"])
    model = GPT(cfg).to(device)

    state_dict = torch.load(local_pt_path, map_location=device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    class GPTForExport(torch.nn.Module):
        def __init__(self, gpt):
            super().__init__()
            self.gpt = gpt

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            return self.gpt(idx)

    export_model = GPTForExport(model).to(device)
    export_model.eval()

    B = 1
    T = cfg.sequence_len
    example_args = (torch.zeros((B, T), dtype=torch.long, device=device),)

    exported_program = torch.export.export(
        export_model,
        example_args,
    )

    # PyTorch
    expected: torch.Tensor = exported_program.module()(*example_args)

    # Relax
    mod = from_exported_program(exported_program, run_ep_decomposition=True)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    dev = tvm.cpu()
    target = tvm.target.Target.from_device(dev)
    exe = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    tvm_args = [tvm.runtime.from_dlpack(x.contiguous()) for x in example_args]
    tvm_outputs = vm["main"](*tvm_args)

    # check if the outputs match
    if isinstance(expected, dict):
        for i, key in enumerate(expected.keys()):
            actual = torch.from_numpy(tvm_outputs[i].numpy())
            torch.testing.assert_close(
                actual, expected[key], rtol=1e-4, atol=1e-4, equal_nan=True
            )
    else:
        actuals = torch.from_numpy(tvm_outputs[0].numpy())
        torch.testing.assert_close(
            actuals, expected, rtol=1e-4, atol=1e-4, equal_nan=True
        )


if __name__ == "__main__":
    test_nanochat()

"""
$ python test_nanochat.py 
Traceback (most recent call last):
  File "/home/ubuntu/data/project/exportedprogram-to-tvm-relax/test_nanochat/test_nanochat.py", line 88, in <module>
    test_nanochat()
  File "/home/ubuntu/data/project/exportedprogram-to-tvm-relax/test_nanochat/test_nanochat.py", line 64, in test_nanochat
    mod = from_exported_program(exported_program, run_ep_decomposition=True)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/data/project/exportedprogram-to-tvm-relax/3rdparty/tvm/python/tvm/relax/frontend/torch/exported_program_translator.py", line 1341, in from_exported_program
    return ExportedProgramImporter().from_exported_program(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/data/project/exportedprogram-to-tvm-relax/3rdparty/tvm/python/tvm/relax/frontend/torch/exported_program_translator.py", line 1173, in from_exported_program
    ) = self.create_input_vars(exported_program)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/data/project/exportedprogram-to-tvm-relax/3rdparty/tvm/python/tvm/relax/frontend/torch/exported_program_translator.py", line 1139, in create_input_vars
    torch_shape = exported_program.state_dict[spec.target].shape
                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
KeyError: 'gpt.cos'
"""
