import torchbenchmark
import torch
from torch.export import export
import pytest

import tvm
from tvm import relax
import tvm.testing
from tvm.relax.frontend.torch import from_exported_program


def verify_model(
    torch_model, example_args, example_kwargs={}, target: str = "llvm", dev=tvm.cpu()
):
    # PyTorch
    exported_program = export(torch_model, args=example_args, kwargs=example_kwargs)
    expected: torch.Tensor = exported_program.module()(*example_args)

    # Relax
    mod = from_exported_program(exported_program)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    exe = relax.build(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    tvm_args = [tvm.nd.from_dlpack(x.contiguous()) for x in example_args]
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


def verify_torchbench_model(model_name):
    torch_model, example_args = torchbenchmark.load_model_by_name(model_name)(
        test="eval", device="cpu", batch_size=1
    ).get_module()
    verify_model(torch_model, example_args)


@pytest.mark.parametrize(
    "torchbench_model_name",
    [
        "BERT_pytorch",
        # "Background_Matting", # skip due to taking too long
        "LearningToPaint",
        "Super_SloMo",
        "alexnet",
        "basic_gnn_edgecnn",
        "basic_gnn_gcn",
        "basic_gnn_gin",
        "basic_gnn_sage",
        "dcgan",
        "demucs",
        "densenet121",
        "dlrm",
        "functorch_dp_cifar10",
        "functorch_maml_omniglot",
        "hf_Albert",
        "hf_Bart",
        "hf_Bert",
        "hf_Bert_large",
        "hf_BigBird",
        "hf_DistilBert",
        "hf_GPT2",
        "hf_GPT2_large",
        "hf_Roberta_base",
        "hf_T5",
        "hf_T5_base",
        "hf_T5_large",
        "hf_Whisper",
        # "hf_distil_whisper", # taking too long
        "lennard_jones",
        # "llama_v2_7b_16h", # need to set `HUGGING_FACE_HUB_TOKEN` to download weights
        "llava",
        "maml",
        "maml_omniglot",
        "microbench_unbacked_tolist_sum",
        "mnasnet1_0",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "moco",
        "moondream",
        # "nanogpt", # segmentation fault
        "nvidia_deeprecommender",
        "phlippe_densenet",
        "phlippe_resnet",
        "pyhpc_equation_of_state",
        "pyhpc_isoneutral_mixing",
        "pyhpc_turbulent_kinetic_energy",
        "pytorch_CycleGAN_and_pix2pix",
        "pytorch_stargan",
        "pytorch_unet",
        "resnet152",
        "resnet18",
        "resnet50",
        "resnext50_32x4d",
        "sam",
        "shufflenet_v2_x1_0",
        "squeezenet1_1",
        # "stable_diffusion_text_encoder", # need to set `HUGGING_FACE_HUB_TOKEN` to download weights
        # "stable_diffusion_unet", # need to set `HUGGING_FACE_HUB_TOKEN` to download weights
        "timm_efficientnet",
        "timm_nfnet",
        "timm_regnet",
        "timm_resnest",
        "timm_vision_transformer",
        "timm_vision_transformer_large",
        "timm_vovnet",
        "torch_multimodal_clip",
        "tts_angular",
        "vgg16",
        "yolov3",
    ],
)
def test_e2e(torchbench_model_name: str):
    verify_torchbench_model(torchbench_model_name)


if __name__ == "__main__":
    tvm.testing.main()
