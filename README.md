# 🥇 Prediksi Harga Emas Menggunakan LSTM

---

## 📁 Struktur File

```
gold_prediction/
├── emas_lstm.ipynb          ← Notebook analisis lengkap (WAJIB jalankan ini dulu)
├── app.py                   ← Dashboard Streamlit
├── requirements.txt         ← Dependensi Python
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependensi
```bash
pip install -r requirements.txt
```

### 2. Siapkan Data
Buat file Excel **`data_emas.xlsx`** dengan kolom:

| Kolom         | Keterangan                  | Satuan      |
|---------------|-----------------------------|-------------|
| Tanggal       | Format: YYYY-MM-DD          | -           |
| Harga_Emas    | Harga emas dunia            | USD/troy oz |
| Inflasi       | Inflasi Indonesia           | %           |
| Kurs_Rupiah   | Kurs IDR/USD (JISDOR BI)    | IDR         |
| BI_Rate       | Suku bunga BI               | %           |
| Harga_Minyak  | Brent Crude Oil             | USD/barrel  |

**Sumber data:**
- Harga Emas: [investing.com](https://investing.com) → Gold Futures
- Inflasi & BI Rate: [bps.go.id](https://bps.go.id) / [bi.go.id](https://bi.go.id)
- Kurs Rupiah: [bi.go.id](https://bi.go.id) → JISDOR
- Harga Minyak: [investing.com](https://investing.com) → Brent Oil

### 3. Jalankan Notebook
```bash
jupyter notebook emas_lstm.ipynb
```
Jalankan semua cell dari atas ke bawah. File model akan tersimpan otomatis.

### 4. Jalankan Dashboard Streamlit
```bash
streamlit run app.py
```

---

## 🧠 Arsitektur Model LSTM

```
Input (look_back=12, n_features=5)
       ↓
LSTM Layer 1 (64 units, return_sequences=True)
       ↓
Dropout (0.2)
       ↓
LSTM Layer 2 (32 units)
       ↓
Dropout (0.2)
       ↓
Dense (16 units, ReLU)
       ↓
Output Dense (1 unit) → Harga Emas
```

**Konfigurasi Training:**
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error
- Callback: EarlyStopping + ReduceLROnPlateau
- Max Epoch: 200
- Batch Size: 16
- Normalisasi: MinMaxScaler [0, 1]

---

## 📊 Metrik Evaluasi

| Metrik | Keterangan |
|--------|-----------|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **MAPE** | Mean Absolute Percentage Error |
| **R²** | Koefisien Determinasi |

---

## 📋 Output yang Dihasilkan

| File | Keterangan |
|------|-----------|
| `model_lstm_emas.h5` | Model LSTM yang telah dilatih |
| `scaler.pkl` | Scaler untuk semua fitur |
| `scaler_target.pkl` | Scaler khusus target |
| `hasil_prediksi_test.csv` | Tabel aktual vs prediksi |
| `hasil_forecast_future.csv` | Forecast 6 bulan ke depan |
| `plot_timeseries.png` | Plot time series |
| `plot_korelasi.png` | Matriks korelasi |
| `plot_scatter.png` | Scatter plot |
| `plot_learning_curve.png` | Kurva pembelajaran |
| `plot_prediksi.png` | Aktual vs prediksi |
| `plot_residual.png` | Analisis residual |
| `plot_forecast.png` | Forecast ke depan |

---

## ⚙️ Cara Mengganti Data Sampel ke Data Asli

Di **`emas_lstm.ipynb`**, pada Cell **"2. Load & Persiapan Data"**:

```python
# HAPUS bagian "OPSI B: Data Simulasi" dan aktifkan:
df = pd.read_excel('data_emas.xlsx', parse_dates=['Tanggal'])
# atau
df = pd.read_csv('data_emas.csv', parse_dates=['Tanggal'])
df = df.set_index('Tanggal')
```

Di **`app.py`**, upload file melalui sidebar → "Upload File Saya".
