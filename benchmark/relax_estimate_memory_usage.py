import torch
import torchvision
import tvm
from tvm.relax.frontend.torch import from_exported_program


def main():
    tvm_device = tvm.cpu()
    tvm_target = tvm.target.Target.from_device(tvm_device)

    model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1").eval()
    epxample_args = (torch.randn(1, 3, 224, 224),)
    exported_program = torch.export.export(model, epxample_args)

    mod = from_exported_program(exported_program)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    pipeline = tvm.transform.Sequential(
        [
            tvm.relax.transform.LegalizeOps(),
            tvm.relax.transform.RewriteDataflowReshape(),
            tvm.relax.transform.ToNonDataflow(),
            tvm.relax.transform.RemovePurityChecking(),
            tvm.relax.transform.CallTIRRewrite(),
            tvm.relax.transform.StaticPlanBlockMemory(),
        ]
    )
    with tvm_target:
        mod = pipeline(mod)
    print(mod.show())
    print(tvm.relax.analysis.estimate_memory_usage(mod))


if __name__ == "__main__":
    main()
