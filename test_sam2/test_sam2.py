import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.export import Dim

import tvm

from tvm import relax

from tvm.relax.frontend.torch import from_exported_program

def test_sam2():
    sam2_checkpoint = "./sam2_repo/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cpu"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)
    # torch_model = predictor.model.image_encoder.eval()
    torch_model = predictor.model.image_encoder.bfloat16().eval()
    img_size=1024

    example_args = (
        torch.randn(5, 3, img_size, img_size).to(device).type(torch.bfloat16)
    )

    # dynamic shapes
    batch_size = Dim("batch", min=2, max=20)
    # height = Dim("height", min=2, max=2048)
    # width = Dim("width", min=2, max=2048)
    dynamic_shapes = {
        "sample": {0: batch_size},
    }

    # PyTorch
    with torch.no_grad():
        exported_program = torch.export.export(
            torch_model,
            (example_args,),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )

    mod = from_exported_program(exported_program)
    expected: torch.Tensor = exported_program.module()(*example_args)

    # Relax
    dev = tvm.cpu()
    target = tvm.target.Target.from_device(dev)

    mod = from_exported_program(exported_program)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    exe = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    tvm_args = [tvm.runtime.from_dlpack(x.contiguous()) for x in example_args]
    tvm_outputs = vm["main"](*tvm_args)



if __name__ == "__main__":
    test_sam2()
