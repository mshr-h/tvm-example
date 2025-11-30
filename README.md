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
uv run pytest test_torchvision.py -v
```

```
FAILED test_torchvision.py::test_e2e[swin_t-dynamic] - TypeError: 'NoneType' object is not iterable
FAILED test_torchvision.py::test_e2e[swin_v2_t-dynamic] - TypeError: 'NoneType' object is not iterable
FAILED test_torchvision.py::test_e2e[lraspp_mobilenet_v3_large-dynamic] - torch._dynamo.exc.UserError: When `dynamic_shapes` is specified as a dict, its top-level keys must be the arg names ['input'] of `inputs`, but here they are ['x']. Alternatively, you could also ignore arg na...
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
FAILED test_torchbench.py::test_e2e[BERT_pytorch] - AssertionError: Tensor-likes are not close!
FAILED test_torchbench.py::test_e2e[LearningToPaint] - AssertionError: The values for attribute 'shape' do not match: torch.Size([1, 64, 64, 64]) != torch.Size([1, 65]).
FAILED test_torchbench.py::test_e2e[Super_SloMo] - AssertionError: Unsupported function types ['grid_sampler_2d.default', 'mean.default']
FAILED test_torchbench.py::test_e2e[basic_gnn_edgecnn] - KeyError: 'torch_geometric.compile'
FAILED test_torchbench.py::test_e2e[basic_gnn_gcn] - KeyError: 'torch_geometric.compile'
FAILED test_torchbench.py::test_e2e[basic_gnn_gin] - KeyError: 'torch_geometric.compile'
FAILED test_torchbench.py::test_e2e[basic_gnn_sage] - KeyError: 'torch_geometric.compile'
FAILED test_torchbench.py::test_e2e[dcgan] - AssertionError: The values for attribute 'shape' do not match: torch.Size([1, 128, 16, 16]) != torch.Size([1, 1, 1, 1]).
FAILED test_torchbench.py::test_e2e[demucs] - AssertionError: The values for attribute 'shape' do not match: torch.Size([1, 4, 2, 371372]) != torch.Size([4, 2, 371372]).
FAILED test_torchbench.py::test_e2e[dlrm] - ModuleNotFoundError: No module named 'onnx'
FAILED test_torchbench.py::test_e2e[functorch_dp_cifar10] - AssertionError: Unsupported function types ['native_group_norm.default']
FAILED test_torchbench.py::test_e2e[functorch_maml_omniglot] - _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
FAILED test_torchbench.py::test_e2e[hf_Albert] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_Bart] - AttributeError: 'str' object has no attribute 'new_zeros'
FAILED test_torchbench.py::test_e2e[hf_Bert] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_Bert_large] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_BigBird] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_DistilBert] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_GPT2] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_GPT2_large] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_Roberta_base] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_T5] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_T5_base] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_T5_large] - AttributeError: 'str' object has no attribute 'size'
FAILED test_torchbench.py::test_e2e[hf_Whisper] - AssertionError: Unsupported function types ['native_layer_norm.default']
FAILED test_torchbench.py::test_e2e[llava] - TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not str
FAILED test_torchbench.py::test_e2e[maml] - torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: Could not extract specialized integer from data-dependent expression u192 + u197 + u202 + u207 + u212 + u217 + u222 + u227 + u232 + u237 + u242 + u247 + u252...
FAILED test_torchbench.py::test_e2e[maml_omniglot] - ModuleNotFoundError: No module named 'higher'
FAILED test_torchbench.py::test_e2e[microbench_unbacked_tolist_sum] - AssertionError: Unsupported function types ['_assert_scalar.default', 'ge', 'add', 'le', 'lt', 'sym_constrain_range_for_size.default']
FAILED test_torchbench.py::test_e2e[moco] - NotImplementedError: DistributedDataParallel/allgather requires cuda
FAILED test_torchbench.py::test_e2e[moondream] - TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not str
FAILED test_torchbench.py::test_e2e[nvidia_deeprecommender] - AssertionError: Tensor-likes are not close!
FAILED test_torchbench.py::test_e2e[phlippe_densenet] - AssertionError: The values for attribute 'shape' do not match: torch.Size([1, 32, 32, 32]) != torch.Size([1, 10]).
FAILED test_torchbench.py::test_e2e[phlippe_resnet] - AssertionError: The values for attribute 'shape' do not match: torch.Size([1, 16, 32, 32]) != torch.Size([1, 10]).
FAILED test_torchbench.py::test_e2e[pyhpc_equation_of_state] - NotImplementedError: input_type torch.float64 is not handled yet
FAILED test_torchbench.py::test_e2e[pyhpc_isoneutral_mixing] - NotImplementedError: input_type torch.float64 is not handled yet
FAILED test_torchbench.py::test_e2e[pyhpc_turbulent_kinetic_energy] - NotImplementedError: Model doesn't support customizing batch size, but eval test is providing a batch size other than DEFAULT_EVAL_BSIZE
FAILED test_torchbench.py::test_e2e[pytorch_CycleGAN_and_pix2pix] - ModuleNotFoundError: No module named 'dominate'
FAILED test_torchbench.py::test_e2e[pytorch_stargan] - NotImplementedError: Model doesn't support customizing batch size, but eval test is providing a batch size other than DEFAULT_EVAL_BSIZE
FAILED test_torchbench.py::test_e2e[pytorch_unet] - AssertionError: Unsupported function types ['constant_pad_nd.default']
FAILED test_torchbench.py::test_e2e[sam] - KeyError: 'pixel_mean'
FAILED test_torchbench.py::test_e2e[timm_nfnet] - AssertionError: Unsupported function types ['constant_pad_nd.default', '_native_batch_norm_legit.no_stats']
FAILED test_torchbench.py::test_e2e[timm_resnest] - AssertionError: Tensor-likes are not close!
FAILED test_torchbench.py::test_e2e[timm_vision_transformer] - AssertionError: Unsupported function types ['native_layer_norm.default']
FAILED test_torchbench.py::test_e2e[timm_vision_transformer_large] - AssertionError: Unsupported function types ['native_layer_norm.default']
FAILED test_torchbench.py::test_e2e[torch_multimodal_clip] - AssertionError: Unsupported function types ['native_layer_norm.default']
FAILED test_torchbench.py::test_e2e[tts_angular] - AssertionError: Tensor-likes are not close!
FAILED test_torchbench.py::test_e2e[yolov3] - AssertionError: Unsupported function types ['copy.default']
```

## TIMM

```bash
pytest test_timm.py -v -k "static" # only run static shape tests because dynamic shape tests take too long
```

```
FAILED test_timm.py::test_e2e[adv_inception_v3-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[beit_base_patch16_224-static] - KeyError: 'blocks.0.attn.k_bias'
FAILED test_timm.py::test_e2e[convit_base-static] - AssertionError: Unsupported function types ['div_.Tensor', 'repeat_interleave.self_int', 'to.device']
FAILED test_timm.py::test_e2e[convmixer_768_32-static] - AssertionError: Unsupported function types ['conv2d.padding']
FAILED test_timm.py::test_e2e[crossvit_9_240-static] - AssertionError: Unsupported function types ['alias.default']
FAILED test_timm.py::test_e2e[dm_nfnet_f0-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[eca_halonext26ts-static] - AssertionError: Unsupported function types ['unfold.default']
FAILED test_timm.py::test_e2e[ghostnet_100-static] - AssertionError: Unsupported function types ['alias.default']
FAILED test_timm.py::test_e2e[gluon_inception_v3-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[inception_v3-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[jx_nest_base-static] - tvm.error.InternalError: PermuteDims expects the number of input axes to equal the ndim of the input tensor. However, the tensor ndim is 5 while the given number of axes is 4
FAILED test_timm.py::test_e2e[levit_128-static] - KeyError: 'stages.0.blocks.0.attn.attention_bias_idxs'
FAILED test_timm.py::test_e2e[mixer_b16_224-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[nfnet_l0-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[poolformer_m36-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[res2net101_26w_4s-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[res2net50_14w_8s-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[res2next50-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[resmlp_12_224-static] - AssertionError: Unsupported function types ['addcmul.default']
FAILED test_timm.py::test_e2e[resnest101e-static] - AssertionError: Tensor-likes are not close!
FAILED test_timm.py::test_e2e[swin_base_patch4_window7_224-static] - KeyError: 'layers.0.blocks.0.attn.relative_position_index'
FAILED test_timm.py::test_e2e[tnt_s_patch16_224-static] - AssertionError: Unsupported function types ['im2col.default']
FAILED test_timm.py::test_e2e[volo_d1_224-static] - AssertionError: Unsupported function types ['im2col.default', 'max.dim', 'col2im.default']
FAILED test_timm.py::test_e2e[xcit_large_24_p8_224-static] - TypeError: Binary operators must have the same datatype for both operands.  However, R.floor_divide(lv29, R.const(2, "int32")) uses datatype float32 on the LHS (StructInfo of R.Tensor((32,), dtype="float32")), and datatype i...
```

## (wip) sam2

```bash
cd test_sam2
git clone https://github.com/facebookresearch/sam2.git sam2_repo
cd sam2_repo
uv pip install -e .
cd checkpoints
bash download_ckpts.sh
cd ../../
uv run test_sam2.py
```

```
AssertionError: Unsupported function types ['mul']
```

## nanoGPT

```bash
cd test_nanogpt
uv run pytest test_nanogpt.py -v
```

```
FAILED test_nanogpt.py::test_nanpgpt - tvm.error.InternalError: Cannot decide min_value for typebool
```

## nanochat

```bash
uv run test_nanochat.py
```

```
Traceback (most recent call last):
  File "/home/ubuntu/data/project/exportedprogram-to-tvm-relax/test_nanochat.py", line 283, in <module>
    test_nanochat()
  File "/home/ubuntu/data/project/exportedprogram-to-tvm-relax/test_nanochat.py", line 255, in test_nanochat
    mod = from_exported_program(exported_program, run_ep_decomposition=True)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/data/project/exportedprogram-to-tvm-relax/3rdparty/tvm/python/tvm/relax/frontend/torch/exported_program_translator.py", line 1784, in from_exported_program
    return ExportedProgramImporter().from_exported_program(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/data/project/exportedprogram-to-tvm-relax/3rdparty/tvm/python/tvm/relax/frontend/torch/exported_program_translator.py", line 1643, in from_exported_program
    self._check_unsupported_func_type(nodes)
  File "/home/ubuntu/data/project/exportedprogram-to-tvm-relax/3rdparty/tvm/python/tvm/relax/frontend/torch/base_fx_graph_translator.py", line 182, in _check_unsupported_func_type
    assert not missing_func_types, f"Unsupported function types {missing_func_types}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Unsupported function types ['add.Scalar']
```

## llm-jp-3

```bash
uv run pytest test_llm-jp-3.py -v -s
```

```
3rdparty/tvm/python/tvm/relax/frontend/torch/exported_program_translator.py:1786: in from_exported_program
    return ExportedProgramImporter().from_exported_program(
3rdparty/tvm/python/tvm/relax/frontend/torch/exported_program_translator.py:1678: in from_exported_program
    self.env[node] = self.convert_map[func_name](node)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
3rdparty/tvm/python/tvm/relax/frontend/torch/base_fx_graph_translator.py:1819: in _index_put
    return self.block_builder.emit(relax.op.index_put(tensor, indices, values, accumulate))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
3rdparty/tvm/python/tvm/relax/block_builder.py:328: in emit
    return _ffi_api.BlockBuilderEmit(self, expr, name_hint)  # type: ignore
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
python/tvm_ffi/cython/function.pxi:904: in tvm_ffi.core.Function.__call__
    ???
<unknown>:0: in tvm::relax::BlockBuilderImpl::Emit(tvm::RelaxExpr, tvm::ffi::String)
    ???
<unknown>:0: in tvm::relax::BlockBuilderImpl::Emit(tvm::RelaxExpr, bool, tvm::ffi::String)
    ???
<unknown>:0: in tvm::relax::Normalizer::Normalize(tvm::RelaxExpr const&)
    ???
<unknown>:0: in tvm::relax::Normalizer::VisitExpr(tvm::RelaxExpr const&)
    ???
<unknown>:0: in tvm::relax::ExprFunctor<tvm::RelaxExpr (tvm::RelaxExpr const&)>::VisitExpr(tvm::RelaxExpr const&)
    ???
<unknown>:0: in tvm::NodeFunctor<tvm::RelaxExpr (tvm::ffi::ObjectRef const&, tvm::relax::ExprFunctor<tvm::RelaxExpr (tvm::RelaxExpr const&)>*)>::operator()(tvm::ffi::ObjectRef const&, tvm::relax::ExprFunctor<tvm::RelaxExpr (tvm::RelaxExpr const&)>*) const
    ???
<unknown>:0: in tvm::relax::ExprFunctor<tvm::RelaxExpr (tvm::RelaxExpr const&)>::InitVTable()::{lambda(tvm::ffi::ObjectRef const&, tvm::relax::ExprFunctor<tvm::RelaxExpr (tvm::RelaxExpr const&)>*)#9}::__invoke(tvm::ffi::ObjectRef const&, tvm::relax::ExprFunctor<tvm::RelaxExpr (tvm::RelaxExpr const&)>*)
    ???
<unknown>:0: in non-virtual thunk to tvm::relax::Normalizer::VisitExpr_(tvm::relax::CallNode const*)
    ???
<unknown>:0: in tvm::relax::Normalizer::VisitExpr_(tvm::relax::CallNode const*)
    ???
<unknown>:0: in tvm::relax::Normalizer::InferStructInfo(tvm::relax::Call const&)
    ???
<unknown>:0: in tvm::relax::InferStructInfoIndexPut(tvm::relax::Call const&, tvm::relax::BlockBuilder const&)
    ???
<unknown>:0: in tvm::relax::BlockBuilderImpl::ReportFatal(tvm::Diagnostic const&)
    ???
<unknown>:0: in tvm::runtime::detail::LogFatal::~LogFatal()
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

>   ???
E   tvm.error.InternalError: IndexPut requires the number of index tensors (3) to match the data tensor dimensions (4)

<unknown>:0: InternalError
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
