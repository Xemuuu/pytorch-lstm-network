# Plik działa na podstawie danych z pliku np. data/AAPL.csv które zawierają kolumny 'open', 'close', 'low', 'high', 'volume'
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


input_file = 'data/AAPL.csv'
train_dir = 'train_data'
test_dir = 'test_data'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


df = pd.read_csv(input_file)
features = df[['open', 'close', 'low', 'high', 'volume']].values.astype(np.float32)
target = df[['close']].values.astype(np.float32)


feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_scaled = feature_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target)


train_size = int(len(features_scaled) * 0.8)
features_train, features_test = features_scaled[:train_size], features_scaled[train_size:]
target_train, target_test = target_scaled[:train_size], target_scaled[train_size:]

np.savetxt(f"{train_dir}/train_AAPL.csv", features_train, delimiter=",")
np.savetxt(f"{test_dir}/test_AAPL.csv", features_test, delimiter=",")


def create_sequences(features, target, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(target[i+window_size])
    return np.array(X), np.array(y)

window_size = 30
X_train, y_train = create_sequences(features_train, target_train, window_size)
X_test, y_test = create_sequences(features_test, target_test, window_size)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

input_size = X_train.shape[2]
model = LSTMModel(input_size=input_size)


X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")


model.eval()
with torch.no_grad():
    predicted = model(X_test_t).numpy()


predicted_prices = target_scaler.inverse_transform(predicted)
actual_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1))


# Wizualizacja 
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Rzeczywiste ceny CLOSE')
plt.plot(predicted_prices, label='Prognozowane ceny CLOSE')
plt.xlabel('Dzień')
plt.ylabel('Cena (USD)')
plt.title('Prognoza cen akcji')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

