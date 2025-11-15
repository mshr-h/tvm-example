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
pytest test_torchvision.py -v
```

```
FAILED test_torchvision.py::test_e2e[inception_v3-dynamic] - AssertionError: Tensor-likes are not close!
FAILED test_torchvision.py::test_e2e[inception_v3-static] - AssertionError: Tensor-likes are not close!
FAILED test_torchvision.py::test_e2e[maxvit_t-dynamic] - AssertionError: Unsupported function types ['sym_size.int', 'swapaxes.default']
FAILED test_torchvision.py::test_e2e[maxvit_t-static] - AssertionError: Unsupported function types ['swapaxes.default']
FAILED test_torchvision.py::test_e2e[shufflenet_v2_x0_5-dynamic] - AssertionError: Unsupported function types ['sym_size.int']
FAILED test_torchvision.py::test_e2e[swin_t-dynamic] - AssertionError: Unsupported function types ['sym_size.int', 'fill_.Tensor', 'mul']
FAILED test_torchvision.py::test_e2e[swin_t-static] - AssertionError: Unsupported function types ['alias.default', 'fill_.Tensor']
FAILED test_torchvision.py::test_e2e[swin_v2_t-dynamic] - AssertionError: Unsupported function types ['sym_size.int', 'fill_.Tensor', 'mul']
FAILED test_torchvision.py::test_e2e[swin_v2_t-static] - AssertionError: Unsupported function types ['alias.default', 'fill_.Tensor']
FAILED test_torchvision.py::test_e2e[vit_b_32-dynamic] - AssertionError: Unsupported function types ['sym_size.int', 'mul']
FAILED test_torchvision.py::test_e2e[quantized_inception_v3-dynamic] - AssertionError: Tensor-likes are not close!
FAILED test_torchvision.py::test_e2e[quantized_inception_v3-static] - AssertionError: Tensor-likes are not close!
FAILED test_torchvision.py::test_e2e[quantized_shufflenet_v2_x0_5-dynamic] - AssertionError: Unsupported function types ['sym_size.int']
FAILED test_torchvision.py::test_e2e[lraspp_mobilenet_v3_large-dynamic] - torch._dynamo.exc.UserError: When `dynamic_shapes` is specified as a dict, its top-level keys must be the arg names ['input'] of `inputs`, but here they are ['x']. Alternatively,...
```

## torchbench

```bash
cd test_torchbench
git clone https://github.com/pytorch/benchmark --recursive
cd benchmark
uv pip install -e .
cd ..
pytest test_torchbench.py -v
```

```
FAILED test_torchbench.py::test_e2e[BERT_pytorch] - AssertionError: Mutating module attribute mask during export.
FAILED test_torchbench.py::test_e2e[LearningToPaint] - AssertionError: Tensor-likes are not close!
FAILED test_torchbench.py::test_e2e[Super_SloMo] - AssertionError: Unsupported function types ['l1_loss.default', 'mse_loss.default', 'broadcast_tensors.default', 'grid_sampler.default', 'mean.default']
FAILED test_torchbench.py::test_e2e[basic_gnn_edgecnn] - AssertionError: Unsupported function types ['new_zeros.default', 'scatter_reduce_.two']
FAILED test_torchbench.py::test_e2e[basic_gnn_gcn] - AssertionError: Unsupported function types ['add', 'masked_fill_.Scalar', 'le', 'pow_.Scalar', 'scatter_add_.default', 'new_zeros.default', 'sym_size.int', 'sym_constrain_range_for_size.default', '_assert_scalar.default'...
FAILED test_torchbench.py::test_e2e[basic_gnn_gin] - AssertionError: Unsupported function types ['new_zeros.default', 'scatter_add_.default']
FAILED test_torchbench.py::test_e2e[basic_gnn_sage] - AssertionError: Unsupported function types ['new_zeros.default', 'scatter_add_.default']
FAILED test_torchbench.py::test_e2e[dcgan] - AssertionError: Tensor-likes are not close!
FAILED test_torchbench.py::test_e2e[demucs] - AssertionError: Unsupported function types ['lstm.input', 'glu.default']
FAILED test_torchbench.py::test_e2e[dlrm] - AssertionError: Unsupported function types ['embedding_bag.padding_idx']
FAILED test_torchbench.py::test_e2e[functorch_maml_omniglot] - _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
FAILED test_torchbench.py::test_e2e[hf_Albert] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_Bart] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_Bert] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_Bert_large] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_BigBird] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_DistilBert] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_GPT2] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_GPT2_large] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_Roberta_base] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_T5] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_T5_base] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[hf_T5_large] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[llava] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[maml] - torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
FAILED test_torchbench.py::test_e2e[maml_omniglot] - NotImplementedError: Model doesn't support customizing batch size, but eval test is providing a batch size other than DEFAULT_EVAL_BSIZE
FAILED test_torchbench.py::test_e2e[microbench_unbacked_tolist_sum] - AssertionError: Unsupported function types ['add', 'le', 'sym_constrain_range_for_size.default', 'lt', '_assert_scalar.default', 'ge']
FAILED test_torchbench.py::test_e2e[moco] - NotImplementedError: DistributedDataParallel/allgather requires cuda
FAILED test_torchbench.py::test_e2e[moondream] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'dict'>
FAILED test_torchbench.py::test_e2e[phlippe_densenet] - AssertionError: Tensor-likes are not close!
FAILED test_torchbench.py::test_e2e[phlippe_resnet] - AssertionError: Tensor-likes are not close!
FAILED test_torchbench.py::test_e2e[pyhpc_equation_of_state] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'list'>
FAILED test_torchbench.py::test_e2e[pyhpc_isoneutral_mixing] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'list'>
FAILED test_torchbench.py::test_e2e[pyhpc_turbulent_kinetic_energy] - NotImplementedError: Model doesn't support customizing batch size, but eval test is providing a batch size other than DEFAULT_EVAL_BSIZE
FAILED test_torchbench.py::test_e2e[pytorch_CycleGAN_and_pix2pix] - AssertionError: Unsupported function types ['instance_norm.default']
FAILED test_torchbench.py::test_e2e[pytorch_stargan] - NotImplementedError: Model doesn't support customizing batch size, but eval test is providing a batch size other than DEFAULT_EVAL_BSIZE
FAILED test_torchbench.py::test_e2e[sam] - KeyError: 'pixel_mean'
FAILED test_torchbench.py::test_e2e[timm_nfnet] - AssertionError: Tensor-likes are not close!
FAILED test_torchbench.py::test_e2e[timm_resnest] - AssertionError: Tensor-likes are not close!
FAILED test_torchbench.py::test_e2e[torch_multimodal_clip] - IndexError: ShapeExpr index out of range
FAILED test_torchbench.py::test_e2e[tts_angular] - torch._dynamo.exc.UserError: Expecting `args` to be a tuple of example positional inputs, got <class 'list'>
FAILED test_torchbench.py::test_e2e[yolov3] - AssertionError: Unsupported function types ['sigmoid_.default', 'meshgrid.default']
```

## TIMM

```bash
pytest test_timm.py -v
```

```
FAILED test_timm.py::test_e2e[adv_inception_v3] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[beit_base_patch16_224] - KeyError: 'blocks.0.attn.k_bias'
FAILED test_timm.py::test_e2e[convit_base] - AssertionError: Mutating module attribute rel_indices during export.
FAILED test_timm.py::test_e2e[convmixer_768_32] - AssertionError: Unsupported function types ['conv2d.padding']
FAILED test_timm.py::test_e2e[crossvit_9_240] - AssertionError: Unsupported function types ['upsample_bicubic2d.vec']
FAILED test_timm.py::test_e2e[dm_nfnet_f0] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[eca_halonext26ts] - AssertionError: Unsupported function types ['unfold.default']
FAILED test_timm.py::test_e2e[gluon_inception_v3] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[inception_v3] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[jx_nest_base] - tvm._ffi.base.TVMError: Traceback (most recent call last):
FAILED test_timm.py::test_e2e[levit_128] - KeyError: 'stages.0.blocks.0.attn.attention_bias_idxs'
FAILED test_timm.py::test_e2e[mixer_b16_224] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[nfnet_l0] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[poolformer_m36] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[res2net101_26w_4s] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[res2net50_14w_8s] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[res2next50] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[resmlp_12_224] - AssertionError: Unsupported function types ['addcmul.default']
FAILED test_timm.py::test_e2e[resnest101e] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[swin_base_patch4_window7_224] - KeyError: 'layers.0.blocks.0.attn.relative_position_index'
FAILED test_timm.py::test_e2e[tnt_s_patch16_224] - AssertionError: Unsupported function types ['im2col.default']
FAILED test_timm.py::test_e2e[volo_d1_224] - AssertionError: Unsupported function types ['max.dim', 'col2im.default', 'im2col.default']
FAILED test_timm.py::test_e2e[xcit_large_24_p8_224] - AssertionError: Unsupported function types ['div.Tensor_mode']
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

```
AssertionError: Unsupported function types ['sym_size.int', 'upsample_bicubic2d.vec', 'mul']
```

## nanoGPT

```bash
cd test_nanogpt
pytest test_nanogpt.py -v
```

```
FAILED test_nanogpt.py::test_nanpgpt - AssertionError: Unsupported function types ['sym_size.int']
```

## Ideas

- TorchInductor Performance DashBoard (aot_inductor)
  - torchbench
    - BERT_pytorch
    - Background_Matting
    - LearningToPaint
    - Super_SloMo
    - alexnet
    - basic_gnn_edgecnn
    - basic_gnn_gcn
    - basic_gnn_gin
    - basic_gnn_sage
    - dcgan
    - demucs
    - densenet121
    - dlrm
    - functorch_dp_cifar10
    - functorch_maml_omniglot
    - hf_Albert
    - hf_Bart
    - hf_Bert
    - hf_Bert_large
    - hf_BigBird
    - hf_DistilBert
    - hf_GPT2
    - hf_GPT2_large
    - hf_Roberta_base
    - hf_T5
    - hf_T5_base
    - hf_T5_large
    - hf_Whisper
    - hf_distil_whisper
    - lennard_jones
    - llama_v2_7b_16h: HUGGING_FACE_HUB_TOKENが必要
    - llava
    - maml
    - maml_omniglot
    - microbench_unbacked_tolist_sum
    - mnasnet1_0
    - mobilenet_v2
    - mobilenet_v3_large
    - moco
    - moondream
    - nanogpt
    - nvidia_deeprecommender
    - phlippe_densenet
    - phlippe_resnet
    - pyhpc_equation_of_state
    - pyhpc_isoneutral_mixing
    - pyhpc_turbulent_kinetic_energy
    - pytorch_CycleGAN_and_pix2pix
    - pytorch_stargan
    - pytorch_unet
    - resnet152
    - resnet18
    - resnet50
    - resnext50_32x4d
    - sam
    - shufflenet_v2_x1_0
    - squeezenet1_1
    - stable_diffusion_text_encoder: HUGGING_FACE_HUB_TOKENが必要
    - stable_diffusion_unet: HUGGING_FACE_HUB_TOKENが必要
    - timm_efficientnet
    - timm_nfnet
    - timm_regnet
    - timm_resnest
    - timm_vision_transformer
    - timm_vision_transformer_large
    - timm_vovnet
    - torch_multimodal_clip
    - tts_angular
    - vgg16
    - yolov3
  - Huggingface
    - AlbertForMaskedLM
    - AlbertForQuestionAnswering
    - BartForCausalLM
    - BartForConditionalGeneration
    - BertForMaskedLM
    - BertForQuestionAnswering
    - BlenderbotForCausalLM
    - BlenderbotSmallForCausalLM
    - BlenderbotSmallForConditionalGeneration
    - CamemBert
    - DebertaForMaskedLM
    - DebertaForQuestionAnswering
    - DebertaV2ForMaskedLM
    - DebertaV2ForQuestionAnswering
    - DistilBertForMaskedLM
    - DistilBertForQuestionAnswering
    - DistillGPT2
    - ElectraForCausalLM
    - ElectraForQuestionAnswering
    - GPT2ForSequenceClassification
    - GoogleFnet
    - LayoutLMForMaskedLM
    - LayoutLMForSequenceClassification
    - M2M100ForConditionalGeneration
    - MBartForCausalLM
    - MBartForConditionalGeneration
    - MT5ForConditionalGeneration
    - MegatronBertForCausalLM
    - MegatronBertForQuestionAnswering
    - MobileBertForMaskedLM
    - MobileBertForQuestionAnswering
    - OPTForCausalLM
    - PLBartForCausalLM
    - PLBartForConditionalGeneration
    - PegasusForCausalLM
    - PegasusForConditionalGeneration
    - RobertaForCausalLM
    - RobertaForQuestionAnswering
    - Speech2Text2ForCausalLM
    - T5ForConditionalGeneration
    - T5Small
    - TrOCRForCausalLM
    - XGLMForCausalLM
    - XLNetLMHeadModel
    - YituTechConvBert
  - TIMM
    - adv_inception_v3
    - beit_base_patch16_224
    - botnet26t_256
    - cait_m36_384
    - coat_lite_mini
    - convit_base
    - convmixer_768_32
    - convnext_base
    - crossvit_9_240
    - cspdarknet53
    - deit_base_distilled_patch16_224
    - dla102
    - dm_nfnet_f0
    - dpn107
    - eca_botnext26ts_256
    - eca_halonext26ts
    - ese_vovnet19b_dw
    - fbnetc_100
    - fbnetv3_b
    - gernet_l
    - ghostnet_100
    - gluon_inception_v3
    - gmixer_24_224
    - gmlp_s16_224
    - hrnet_w18
    - inception_v3
    - jx_nest_base
    - lcnet_050
    - levit_128
    - mixer_b16_224
    - mixnet_l
    - mnasnet_100
    - mobilenetv2_100
    - mobilenetv3_large_100
    - mobilevit_s
    - nfnet_l0
    - pit_b_224
    - pnasnet5large
    - poolformer_m36
    - regnety_002
    - repvgg_a2
    - res2net101_26w_4s
    - res2net50_14w_8s
    - res2next50
    - resmlp_12_224
    - resnest101e
    - rexnet_100
    - sebotnet33ts_256
    - selecsls42b
    - spnasnet_100
    - swin_base_patch4_window7_224
    - swsl_resnext101_32x16d
    - tf_efficientnet_b0
    - tf_mixnet_l
    - tinynet_a
    - tnt_s_patch16_224
    - twins_pcpvt_base
    - visformer_small
    - vit_base_patch16_224
    - volo_d1_224
    - xcit_large_24_p8_224
  - Dynamic
    - BERT_pytorch
    - basic_gnn_edgecnn
    - basic_gnn_gcn
    - basic_gnn_gin
    - basic_gnn_sage
    - cm3leon_generate
    - detectron2_fcos_r_50_fpn
    - dlrm
    - hf_T5
    - hf_T5_generate
    - llama
    - nanogpt
    - vision_maskrcnn
