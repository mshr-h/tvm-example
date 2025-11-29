import torch
import torch.nn.functional as F
from model import GPT
from tvm.relax.frontend.torch import from_exported_program
import tvm
from tvm import relax
from torch.nn.attention import SDPBackend


def test_nanpgpt():
    model = GPT.from_pretrained("gpt2")

    # Monkey-patch forward to avoid negative indexing (x[:, [-1], :]) which
    # generates out-of-bounds gathers in TVM lowering.
    def _forward_no_neg(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, t - 1 : t, :])
        loss = None
        if targets is not None:
            logits_full = self.lm_head(x)
            loss = F.cross_entropy(
                logits_full.view(-1, logits_full.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            logits = logits_full[:, t - 1 : t, :]
        return logits, loss

    model.forward = _forward_no_neg.__get__(model, GPT)

    seq_len = 32
    example_args = (torch.randint(0, 100, (1, seq_len), dtype=torch.long),)
    dynamic_shape = ({1: torch.export.Dim("token_dim", max=model.config.block_size)},)

    # PyTorch
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        exported_program = torch.export.export(
            model, example_args, dynamic_shapes=dynamic_shape
        )
    expected: torch.Tensor = exported_program.module()(*example_args)

    # Relax
    mod = from_exported_program(exported_program, run_ep_decomposition=True)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    dev = tvm.cpu()
    target = tvm.target.Target.from_device(dev)
    exe = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    # The Relax graph keeps lm_head.weight as a runtime parameter (p_lm_head_weight).
    # Pass it explicitly to match the PyTorch execution.
    tvm_args = [tvm.runtime.from_dlpack(x.contiguous()) for x in example_args] + [
        tvm.runtime.from_dlpack(model.lm_head.weight.detach().contiguous())
    ]
    tvm_outputs = vm["main"](*tvm_args)

    # check if the outputs match
    if isinstance(expected, dict):
        for i, key in enumerate(expected.keys()):
            actual = torch.from_numpy(tvm_outputs[i].numpy())
            torch.testing.assert_close(
                actual, expected[key], rtol=1e-4, atol=1e-4, equal_nan=True
            )
    elif isinstance(expected, (tuple, list)):
        assert len(tvm_outputs) == len(expected)
        for actual, exp in zip(tvm_outputs, expected):
            if exp is None:
                assert actual is None
            else:
                torch.testing.assert_close(
                    torch.from_numpy(actual.numpy()),
                    exp,
                    rtol=1e-4,
                    atol=1e-4,
                    equal_nan=True,
                )
    else:
        actuals = torch.from_numpy(tvm_outputs[0].numpy())
        torch.testing.assert_close(
            actuals, expected, rtol=1e-4, atol=1e-4, equal_nan=True
        )


if __name__ == "__main__":
    test_nanpgpt()
