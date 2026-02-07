import torch
from tvm.relax.frontend.torch import from_fx


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor(1.0))
        self.y = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self):
        return self.x + self.y


def main():
    model = torch.fx.symbolic_trace(TestModel())
    print(model)
    print(model.graph)
    eager = model()
    mod = from_fx(
        model,
        [],
    )
    print(mod)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
