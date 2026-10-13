"""Small CPU baselines and explicitly specified sequence architectures."""
from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        encoded, _ = self.lstm(x)
        return self.head(encoded[:, -1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, lookback):
        super().__init__()
        self.input_projection = nn.Linear(1, 16)
        position = torch.arange(lookback, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, 16, 2, dtype=torch.float32) * (-np.log(10000) / 16))
        encoding = torch.zeros(lookback, 16)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        self.register_buffer("position", encoding[None])
        layer = nn.TransformerEncoderLayer(16, 2, dim_feedforward=32, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        # All tokens precede the target; no future target token enters attention.
        x = self.input_projection(x) + self.position
        return self.head(self.encoder(x)[:, -1]).squeeze(-1)


def predict(model, inputs, batch_size=256):
    model.eval()
    with torch.no_grad():
        batches = [model(torch.from_numpy(inputs[i:i + batch_size])).numpy()
                   for i in range(0, len(inputs), batch_size)]
    return np.concatenate(batches)


def fit_network(kind, history, target, masks, scaler, epochs=20, seed=42):
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    lookback = 168 if kind == "lstm168" else 24
    inputs = ((history[:, -lookback:] - scaler.mean_[0]) / scaler.scale_[0]).astype(np.float32)[..., None]
    y = scaler.transform(target[:, None]).ravel().astype(np.float32)
    model = TransformerRegressor(lookback) if kind == "transformer24" else LSTMRegressor()
    dataset = TensorDataset(torch.from_numpy(inputs[masks["train"]]), torch.from_numpy(y[masks["train"]]))
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    best, best_state, stale, history_rows = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item() * len(x_batch)
        validation_predictions = predict(model, inputs[masks["validation"]])
        validation_loss = float(np.mean((validation_predictions - y[masks["validation"]]) ** 2))
        if not np.isfinite(validation_loss):
            raise ValueError("Non-finite validation loss.")
        history_rows.append({"epoch": epoch, "train_mse_scaled": loss_total / len(dataset),
                             "validation_mse_scaled": validation_loss})
        if validation_loss < best:
            best, best_state, stale = validation_loss, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(best_state)
    output = {partition: scaler.inverse_transform(predict(model, inputs[mask])[:, None]).ravel()
              for partition, mask in masks.items() if partition != "train"}
    return output, history_rows
