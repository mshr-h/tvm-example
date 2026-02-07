import torch
from tvm.relax.frontend.torch import relax_dynamo

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor(1.0))
        self.y = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self):
        return self.x + self.y


def main() -> None:
    model = TestModel().cuda()
    eager = model()

    compiled = torch.compile(model, backend=relax_dynamo("zero"))
    compiled_out = compiled()

    print("eager", eager)
    print("compiled", compiled_out)
    print("eager.item", eager.item())
    print("compiled.item", compiled_out.item())


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
