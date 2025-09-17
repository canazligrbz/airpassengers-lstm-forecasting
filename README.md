# Airline Passengers Forecasting with LSTM

Bu proje, **AirPassengers** zaman serisi veri seti üzerinde **LSTM (Long Short-Term Memory)** kullanarak gelecek yolcu sayısını tahmin etmeyi amaçlamaktadır.  

---

## Veri Seti

- **AirPassengers.csv**: 1949–1960 yılları arasındaki aylık yolcu sayısı verilerini içerir.  
- Veri setini [Kaggle’den buradan](https://www.kaggle.com/datasets/rakannimer/air-passengers) indirebilirsiniz.  
- Klasik bir zaman serisi veri setidir ve zaman serisi tahmin çalışmalarında sıkça kullanılır.

---

## Kullanılan Teknolojiler
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Matplotlib  
- Pandas & NumPy  

---

## Adımlar

### 1. Veri Yükleme ve Normalizasyon
- `MinMaxScaler` ile [0,1] aralığında ölçeklendirme yapıldı.  

### 2. Veri Hazırlama (Sliding Window)
- Geçmiş **12 ay** kullanılarak bir sonraki ayın yolcu sayısı tahmin edildi.  

### 3. Model
- LSTM katmanı (100 nöron)  
- Dropout (0.2)  
- Dense çıkış katmanı (1 nöron, regresyon için)  
- Loss: **Mean Squared Error (MSE)**  
- Optimizer: **Adam**  

### 4. Eğitim
- Train-test split (80%-20%)  
- `EarlyStopping` kullanıldı (patience=2).  

### 5. Değerlendirme
- MSE, MAE ve RMSE metrikleri hesaplandı.  
- Gerçek vs Tahmin grafiği çizildi.  

---

## 📊 Sonuçlar

Modelin test verisi üzerindeki hata metrikleri:  

- **MSE**: 503.4399466702405
- **MAE**: 18.34698260271991
- **RMSE**: 22.437467474522176
