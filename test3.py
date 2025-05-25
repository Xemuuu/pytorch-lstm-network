

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === Parametry ===
input_file = 'data/AAPL.csv'
window_size = 60
test_split = 0.2
num_epochs = 30

# === Wczytanie danych ===
df = pd.read_csv(input_file)
all_features = ['open', 'close', 'low', 'high', 'volume']
target_col = 'close'

# === Model LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# === Funkcja do tworzenia sekwencji ===
def create_sequences(features, target, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(target[i+window_size])
    return np.array(X), np.array(y)

# === Funkcja do trenowania modelu ===
def train_model(X_train, y_train, X_test, y_test):
    model = LSTMModel(input_size=X_train.shape[2])
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy()
        test_loss = criterion(model(torch.tensor(X_test)), torch.tensor(y_test)).item()
    return preds, test_loss

# === Wyniki ===
predictions_dict = {}

# === Przetestuj każdy przypadek bez jednej cechy ===
for removed_feature in all_features:
    print(f"\n=== Trenowanie bez cechy: {removed_feature.upper()} ===")

    used_features = [f for f in all_features if f != removed_feature]

    features = df[used_features].values.astype(np.float32)
    target = df[[target_col]].values.astype(np.float32)

    # Skalowanie
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target)

    # Podział na trening i test
    total_len = len(features_scaled)
    test_size = int(total_len * test_split)

    features_train = features_scaled[:-test_size]
    target_train = target_scaled[:-test_size]
    features_test = features_scaled[-test_size:]
    target_test = target_scaled[-test_size:]

    # Sekwencje
    X_train, y_train = create_sequences(features_train, target_train, window_size)
    X_test, y_test = create_sequences(features_test, target_test, window_size)

    if len(X_train) == 0:
        print("Zbyt mało danych do utworzenia sekwencji – pomijanie.")
        continue

    # Trening
    preds_scaled, loss = train_model(
        X_train.astype(np.float32), y_train.astype(np.float32),
        X_test.astype(np.float32), y_test.astype(np.float32)
    )

    # Denormalizacja
    preds = target_scaler.inverse_transform(preds_scaled)
    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    predictions_dict[f'Bez {removed_feature}'] = preds
    if 'actual' not in locals():
        actual = actual  # przypisanie tylko raz
    print(f"Loss bez {removed_feature.upper()}: {loss:.6f}")

# === Wizualizacja ===
plt.figure(figsize=(14, 7))
plt.plot(actual, label='Rzeczywiste ceny CLOSE', linewidth=2)
for label, pred in predictions_dict.items():
    plt.plot(pred, label=label)
plt.title('Wpływ cech wejściowych na predykcję ceny akcji (CLOSE)')
plt.xlabel('Dzień')
plt.ylabel('Cena (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
