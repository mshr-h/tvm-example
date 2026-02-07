import torch
from tvm.relax.frontend.torch import relax_dynamo

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


def main() -> None:
    x = torch.tensor([1.0, 2.0]).cuda()
    y = torch.tensor([2.0, 3.0]).cuda()
    model = TestModel().cuda()
    eager = model(x, y)

    compiled = torch.compile(model, backend=relax_dynamo("zero"))
    compiled_out = compiled(x, y)

    print("eager", eager)
    print("compiled", compiled_out)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
