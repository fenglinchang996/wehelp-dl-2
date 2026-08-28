import csv
from pathlib import Path

import numpy as np
import torch
from gensim.models.doc2vec import KeyedVectors
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

BOARD_TO_ID = {
    "Baseball": 0,
    "Boy-Girl": 1,
    "C_Chat": 2,
    "HatePolitics": 3,
    "Lifeismoney": 4,
    "Military": 5,
    "PC_Shopping": 6,
    "Stock": 7,
    "Tech_Job": 8,
}

TRAINING_EPOCHS = 10


class CustomDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, X_data: np.ndarray, e_data: list[tuple[float]]):
        self.X = torch.tensor(X_data)
        self.e = torch.tensor(e_data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.X[idx], self.e[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, class_num: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, class_num),
        )

    def forward(self, x):
        return self.net(x)


def classify(corpus_file: Path, vecs: KeyedVectors, device: torch.device):
    X_data = []
    e_data = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            vec = vecs[i]
            board = row[0]
            X_data.append(vec)
            e_data.append(BOARD_TO_ID[board])

    full_dataset = CustomDataset(np.array(X_data), e_data)
    train_data_count = int(len(full_dataset) * 0.8)
    test_data_count = len(full_dataset) - train_data_count
    train_dataset, test_dataset = random_split(
        full_dataset, [train_data_count, test_data_count]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = NeuralNetwork(len(train_dataset[0][0]), len(BOARD_TO_ID)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(TRAINING_EPOCHS):
        running_loss = 0
        for batch_X, batch_e in train_dataloader:
            batch_X = batch_X.to(device)
            batch_e = batch_e.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_e)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Model training epoch {epoch + 1} finished, avg loss: {running_loss / len(train_dataloader)}"
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
        print(f"Accuracy: {accuracy * 100:.2f}%")
