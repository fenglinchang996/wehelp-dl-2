import csv
from pathlib import Path

import numpy as np
import torch
from gensim.models.doc2vec import KeyedVectors
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}


class CustomDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, X_data: np.ndarray, e_data: list[int]):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.e = torch.tensor(e_data, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.X[idx], self.e[idx]


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


default_device = torch.device("cpu")


def classify(
    corpus_file: Path,
    vecs: KeyedVectors,
    board_names: list[str],
    hidden_layers: list[int],
    activation: str = "relu",
    optimizer_type: str = "sgd",
    learning_rate: float = 0.001,
    epochs: int = 30,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    device: torch.device = default_device,
):
    board_to_id = {board: i for i, board in enumerate(board_names)}
    X_data = []
    e_data = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row or row[0] not in board_to_id:
                continue
            vec = vecs[i]
            board = row[0]
            X_data.append(vec)
            e_data.append(board_to_id[board])

    full_dataset = CustomDataset(np.array(X_data), e_data)
    train_data_count = int(len(full_dataset) * train_ratio)
    test_data_count = len(full_dataset) - train_data_count
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_data_count, test_data_count],
        generator=torch.Generator().manual_seed(random_seed),
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = len(train_dataset[0][0])
    class_num = len(board_to_id)
    model = NeuralNetwork(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        class_num=class_num,
        activation=activation,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    avg_loss = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_e in train_dataloader:
            batch_X = batch_X.to(device)
            batch_e = batch_e.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_e)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_dataloader)
        print(
            f"\rEpoch {epoch + 1} finished, avg loss: {avg_loss:.4f}",
            end="",
            flush=True,
        )
    print(
        f"\r{epochs} epochs training finished, final avg loss: {avg_loss:.4f}",
        flush=True,
    )
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for batch_X, batch_e in test_dataloader:
            batch_X = batch_X.to(device)
            batch_e = batch_e.to(device)
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, 1)
            total += batch_e.size(0)
            correct += (predicted == batch_e).sum().item()
        accuracy = correct / total
        print(f"Classify accuracy: {accuracy * 100:.2f}%")
        return {
            "testing_accuracy": accuracy,
            "train_count": train_data_count,
            "test_count": test_data_count,
        }
