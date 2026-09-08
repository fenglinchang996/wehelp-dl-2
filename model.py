from torch import Tensor, nn

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        class_num: int,
        activation: str = "relu",
    ):
        super().__init__()
        act = ACTIVATION_MAP.get(activation.lower(), nn.ReLU)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, class_num))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
