

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === Wczytanie danych ===
input_file = 'data/AAPL.csv'  # upewnij się, że ten plik istnieje
df = pd.read_csv(input_file)
features = df[['open', 'close', 'low', 'high', 'volume']].values.astype(np.float32)
target = df[['close']].values.astype(np.float32)

# === Skalowanie danych ===
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
features_scaled = feature_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target)

# === Parametry ===
window_size = 60
total_len = len(features_scaled)
test_size = int(total_len * 0.2)
features_test = features_scaled[-test_size:]
target_test = target_scaled[-test_size:]

# === Tworzenie sekwencji ===
def create_sequences(features, target, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(target[i+window_size])
    return np.array(X), np.array(y)

# === Model LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# === Funkcja do trenowania modelu ===
def train_model(X_train, y_train, X_test_t, y_test_t, num_epochs=30):
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
        predicted = model(X_test_t).numpy()
        test_output = model(X_test_t)
        test_loss = criterion(test_output, y_test_t).item()

    return predicted, test_loss

# === Przygotowanie danych testowych ===
X_test, y_test = create_sequences(features_test, target_test, window_size)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
actual_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# === Trening i testowanie dla różnych rozmiarów danych ===
train_sizes = [0.2, 0.4, 0.6, 0.8]
predicted_all = {}

print("\n=== Wyniki trenowania modeli ===")
for size in train_sizes:
    train_len = int(total_len * size)
    features_train = features_scaled[:train_len]
    target_train = target_scaled[:train_len]

    X_train, y_train = create_sequences(features_train, target_train, window_size)

    if len(X_train) == 0:
        print(f"[{int(size*100)}%] Za mało danych do utworzenia sekwencji – pominięto.")
        continue

    predicted, test_loss = train_model(
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        X_test_t,
        y_test_t
    )

    predicted_prices = target_scaler.inverse_transform(predicted)
    predicted_all[f'{int(size*100)}% danych'] = predicted_prices
    print(f"[{int(size*100)}% danych treningowych] Loss na zbiorze testowym: {test_loss:.6f}")

# === Wizualizacja wyników ===
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, label='Rzeczywiste ceny CLOSE', linewidth=2)

for label, predicted in predicted_all.items():
    plt.plot(predicted, label=f'Predykcja - {label}')

plt.xlabel('Dzień')
plt.ylabel('Cena (USD)')
plt.title('Porównanie predykcji modeli LSTM uczonych na różnych ilościach danych')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
