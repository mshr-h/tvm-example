import pytest
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.integrations.executorch import convert_and_export_with_cache
from tvm.relax.frontend.torch import from_exported_program


def test_nanochat():
    generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation="static",
        cache_config={
            "batch_size": 1,
            "max_cache_len": 20,
        },
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "llm-jp/llm-jp-3-150m",
        device_map="cpu",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        generation_config=generation_config,
    )
    print("Exporting model to ExportedProgram...")
    exported_program = convert_and_export_with_cache(model)

    print("Converting ExportedProgram to TVM Relax module...")
    mod = from_exported_program(exported_program, run_ep_decomposition=True)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
