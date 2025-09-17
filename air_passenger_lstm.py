from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Veri yükleme
df = pd.read_csv("AirPassengers.csv")
data = df['#Passengers'].values.reshape(-1,1)

# Normalizasyon
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

# Sliding window fonksiyonu
def create_sequences(dataset, window_size):
    X = []
    y = []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i+window_size, 0])  # geçmiş değerler
        y.append(dataset[i+window_size, 0])    # tahmin edilecek değer
    return np.array(X), np.array(y)

# Kullanım
window_size = 12
X, y = create_sequences(data_scaled, window_size)

# LSTM input formatına çevirme
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

def build_lstm_model(): 
    model= Sequential()
    model.add(LSTM(units=100, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # regression olduğu için tek değer tahmini
    
    model.compile(loss= "mean_squared_error", optimizer= "adam")
    
    return model

model= build_lstm_model()
model.summary()

early_stopping= EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history= model.fit(X_train, y_train,
                   batch_size=32 ,
                   epochs=100,
                   callbacks=[early_stopping])

y_pred = model.predict(X_test)
y_test_orig = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_orig = scaler.inverse_transform(y_pred)

mse = mean_squared_error(y_test_orig, y_pred_orig)
mae = mean_absolute_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)

plt.figure(figsize=(10,6))
plt.plot(y_test_orig, color='blue', label='Gerçek Yolcu Sayısı')
plt.plot(y_pred_orig, color='red', label='Tahmin Edilen Yolcu Sayısı')
plt.title('LSTM ile Airline Passengers Tahmini')
plt.xlabel('Zaman')
plt.ylabel('Yolcu Sayısı')
plt.legend()
plt.show()