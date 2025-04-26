import timm
import torch
from torch.export import export
import pytest
from PIL import Image

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


def verify_timm_model(model_name):
    from tvm.contrib.download import download_testdata

    # prepare sample image
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_name = "cat.png"
    img_path = download_testdata(img_url, img_name, module="data")
    image = Image.open(img_path)

    torch_model = timm.create_model(model_name, pretrained=True).eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_data_config(torch_model.pretrained_cfg)
    )

    batch = transform(image).unsqueeze(0)
    example_args = (batch,)
    verify_model(torch_model, example_args)


@pytest.mark.parametrize(
    "timm_model_name",
    [
        "adv_inception_v3",
        "beit_base_patch16_224",
        "botnet26t_256",
        "cait_m36_384",
        "coat_lite_mini",
        "convit_base",
        "convmixer_768_32",
        "convnext_base",
        "crossvit_9_240",
        "cspdarknet53",
        "deit_base_distilled_patch16_224",
        "dla102",
        "dm_nfnet_f0",
        "dpn107",
        "eca_botnext26ts_256",
        "eca_halonext26ts",
        "ese_vovnet19b_dw",
        "fbnetc_100",
        "fbnetv3_b",
        "gernet_l",
        "ghostnet_100",
        "gluon_inception_v3",
        "gmixer_24_224",
        "gmlp_s16_224",
        "hrnet_w18",
        "inception_v3",
        "jx_nest_base",
        "lcnet_050",
        "levit_128",
        "mixer_b16_224",
        "mixnet_l",
        "mnasnet_100",
        "mobilenetv2_100",
        "mobilenetv3_large_100",
        "mobilevit_s",
        "nfnet_l0",
        "pit_b_224",
        "pnasnet5large",
        "poolformer_m36",
        "regnety_002",
        "repvgg_a2",
        "res2net101_26w_4s",
        "res2net50_14w_8s",
        "res2next50",
        "resmlp_12_224",
        "resnest101e",
        "rexnet_100",
        "sebotnet33ts_256",
        "selecsls42b",
        "spnasnet_100",
        "swin_base_patch4_window7_224",
        "swsl_resnext101_32x16d",
        "tf_efficientnet_b0",
        "tf_mixnet_l",
        "tinynet_a",
        "tnt_s_patch16_224",
        "twins_pcpvt_base",
        "visformer_small",
        "vit_base_patch16_224",
        "volo_d1_224",
        "xcit_large_24_p8_224",
    ],
)
def test_e2e(timm_model_name: str):
    verify_timm_model(timm_model_name)


if __name__ == "__main__":
    tvm.testing.main()
