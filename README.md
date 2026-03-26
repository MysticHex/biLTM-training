# 🏢 AttnRetrofit

# Attention-Based Energy Forecasting dengan SHAP Diagnostics untuk Smart Building Retrofit Prioritization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **SDG 13 - Climate Action** | Machine Learning Competition Project

---

## 📋 Deskripsi Project

**AttnRetrofit** adalah sistem prediksi konsumsi energi gedung berbasis Deep Learning yang menggunakan **Bidirectional LSTM + Multi-Head Attention** untuk membantu pengambilan keputusan retrofit gedung dalam rangka efisiensi energi dan mitigasi perubahan iklim.

### 🎯 Problem Statement

Gedung-gedung menyumbang sekitar **40% konsumsi energi global** dan **30% emisi CO2**. Retrofit gedung yang tepat sasaran dapat mengurangi konsumsi energi hingga 20-30%. Namun, dengan ribuan gedung yang perlu dianalisis, diperlukan sistem cerdas untuk memprioritaskan gedung mana yang paling membutuhkan retrofit.

### 💡 Solusi

AttnRetrofit menyediakan:
1. **Energy Forecasting** - Prediksi konsumsi energi 24 jam ke depan
2. **Anomaly Detection** - Deteksi gedung dengan pola konsumsi abnormal
3. **Explainable AI** - Penjelasan faktor-faktor penyebab anomali
4. **Retrofit Prioritization** - Ranking gedung berdasarkan prioritas retrofit

---

## 🏗️ Arsitektur Model

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│  Sequence (168h x 14 features) + Building ID Embedding       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              BIDIRECTIONAL LSTM (2 layers)                   │
│  • Hidden dim: 128 per direction (256 total)                 │
│  • Residual connections + Layer Normalization                │
│  • Orthogonal weight initialization                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              MULTI-HEAD SELF-ATTENTION (4 heads)             │
│  • Each head: 64 dimensions                                  │
│  • Pattern capture: recent, seasonal, daily, distant         │
│  • Pre-LN Transformer style                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              FEED-FORWARD NETWORK                            │
│  • 2-layer MLP with ReLU                                     │
│  • Residual connection + LayerNorm                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT LAYER                                    │
│  • Global average pooling                                    │
│  • FC layers: 256 → 128 → 64 → 24                           │
│  • Output: 24-hour energy prediction                         │
└─────────────────────────────────────────────────────────────┘
```

### Key Features:
- **~1.2M parameters** (efficient yet powerful)
- **Multi-head attention** untuk capture pattern temporal berbeda
- **Building embedding** untuk encode karakteristik gedung
- **Gradient clipping** (max_norm=0.5) untuk stable training

---

## 📊 Dataset

Menggunakan **ASHRAE Great Energy Predictor III** dataset:

| Metric | Value |
|--------|-------|
| Total Records | 53.6M+ |
| Buildings | 1,636 |
| Time Range | 2016-2017 (2 tahun) |
| Frequency | Hourly |
| Meter Types | Electricity, Chilled Water, Hot Water, Steam |

### Data Structure:
```
data/
├── train.csv              # Training meter readings (~600MB)
├── weather_train.csv      # Weather data (~200MB)
├── building_metadata.csv  # Building information
├── test.csv               # Test meter readings
└── weather_test.csv       # Test weather data
```

---

## 🗂️ Project Structure

```
AttnRetrofit/
│
├── 📓 AttnRetrofit_EndToEnd.ipynb  # Jupyter notebook (EDA & experiments)
│
├── 🔧 preprocessing.py              # Data cleaning & feature engineering
│   ├── load_and_clean_data()       # Load dan filter data
│   ├── merge_data()                # Merge train + weather + metadata
│   ├── feature_engineering()       # Cyclical encoding, building features
│   ├── create_lag_features()       # Lag & rolling statistics
│   ├── time_based_split()          # Train/Val/Test split
│   ├── create_sequences()          # Sequence creation untuk LSTM
│   └── ASHRAEDataset               # PyTorch Dataset class
│
├── 🧠 model.py                      # Model architecture
│   ├── MultiHeadSelfAttention      # Multi-head attention module
│   └── AttnRetrofitModelV2         # Main model (BiLSTM + Attention)
│
├── 🚀 train.py                      # Training & hyperparameter tuning
│   ├── ASHRAELoss                  # Custom loss (MSE + RMSLE)
│   ├── EarlyStopping               # Early stopping callback
│   ├── train_model()               # Training loop
│   └── create_optuna_study()       # Optuna hyperparameter tuning
│
├── 🔍 evaluate.py                   # Evaluation & anomaly detection
│   ├── get_predictions()           # Batch prediction
│   ├── calculate_metrics()         # RMSE, MAE, RMSLE
│   ├── train_anomaly_classifier()  # Random Forest classifier
│   └── calculate_combined_score()  # Combined anomaly scoring
│
├── 🎨 xai.py                        # Explainable AI
│   ├── calculate_shap_values()     # SHAP DeepExplainer
│   ├── visualize_global_shap()     # Global feature importance
│   ├── visualize_local_shap()      # Local explanations
│   └── visualize_multihead_attention()  # Attention visualization
│
├── 📊 dashboard.py                  # Retrofit priority dashboard (matplotlib)
│   └── create_retrofit_dashboard() # 4-panel visualization
│
├── 🌐 api_server.py                 # FastAPI backend for web dashboard
│   └── REST API endpoints          # /api/buildings, /api/metrics, etc.
│
├── 🌐 dashboard.html                # React + Three.js web dashboard
│   ├── 3D Building Visualization   # Three.js 3D scene
│   ├── Charts                      # Chart.js visualizations
│   └── Building Table              # Searchable building list
│
├── 🎬 run_training.py               # Training pipeline (preprocessing + tuning)
├── 🎬 run_final.py                  # Final model training
├── 🎬 run_xai_dashboard.py          # XAI analysis & dashboard generator
├── 🚀 start_dashboard.bat           # Windows launcher for web dashboard
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 CLAUDE.md                     # Project context (untuk AI assistant)
└── 📄 README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/AttnRetrofit.git
cd AttnRetrofit
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Download ASHRAE dataset dari [Kaggle](https://www.kaggle.com/c/ashrae-energy-prediction/data) dan extract ke folder `data/`:
```
data/
├── train.csv
├── weather_train.csv
├── building_metadata.csv
├── test.csv
└── weather_test.csv
```

### 5. Run Training Pipeline
```bash
# Step 1: Preprocessing + Hyperparameter Tuning
python run_training.py

# Step 2: Final Model Training
python run_final.py
```

---

## 📈 Training Pipeline

### Phase 1: Preprocessing
```python
from preprocessing import run_preprocessing

data = run_preprocessing(data_path='data')
# Output: artifacts/processed/processed_*.npy
```

**Pipeline:**
1. Filter electricity meter only (meter=0)
2. Remove outlier buildings (1099)
3. Merge train + weather + building_metadata
4. Feature engineering:
   - Cyclical time encoding (hour, day_of_week, month)
   - Log transform square_feet
   - Building age calculation
   - Lag features (24h)
   - Rolling statistics (24h mean, std)
5. Time-based split:
   - Train: 2016
   - Validation: 2017 H1
   - Test: 2017 H2
6. Sequence creation: 168h → 24h prediction

### Phase 2: Hyperparameter Tuning (Optuna)
```python
from train import create_optuna_study

best_params = create_optuna_study(train_loader, val_loader, base_config, device)
```

**Search Space:**
| Hyperparameter | Range |
|----------------|-------|
| hidden_dim | [64, 128, 256] |
| num_layers | [1, 2, 3] |
| num_attention_heads | [2, 4, 8] |
| dropout | [0.1, 0.5] |
| learning_rate | [1e-4, 1e-2] |
| weight_decay | [1e-6, 1e-3] |
| batch_size | [32, 64, 128] |
| loss_alpha | [0.3, 0.7] |

### Phase 3: Final Training
```python
from train import train_model
from model import create_model

model = create_model(final_config, device)
history = train_model(model, train_loader, val_loader, final_config, device)
```

**Training Config:**
- Optimizer: AdamW
- Scheduler: CosineAnnealingWarmRestarts
- Gradient Clipping: max_norm=0.5
- Early Stopping: patience=10
- Max Epochs: 100

---

## 🎯 Evaluation Metrics

### Target Performance
| Metric | Target | Description |
|--------|--------|-------------|
| Test RMSLE | < 1.5 | Root Mean Squared Logarithmic Error |
| Test RMSE | < 500 kWh | Root Mean Squared Error |
| Anomaly F1-Score | > 0.80 | F1 untuk anomaly detection |
| Training Time | < 2 hours | With GPU |
| Inference Time | < 100ms | Per building |

### Anomaly Detection
Combined scoring dengan weighted formula:
```
Score = 0.4 × Residual + 0.4 × ML_Probability + 0.2 × SHAP_Variance
```

---

## 🎨 Explainable AI (XAI)

### 1. Global SHAP Analysis
```python
from xai import calculate_shap_values, visualize_global_shap

shap_values, test_sample, indices = calculate_shap_values(
    model, background_data, test_data, test_buildings, feature_names, device
)
visualize_global_shap(shap_values, test_sample, feature_names)
```

### 2. Local SHAP (Per-Building)
```python
from xai import visualize_local_shap

visualize_local_shap(shap_values, test_sample, indices, residual_errors, feature_names)
```

### 3. Multi-Head Attention Visualization
```python
from xai import visualize_multihead_attention

visualize_multihead_attention(model, sample_seq, sample_bid, device)
```

Head Analysis Output:
- **Head 1:** Recent focus (last 24h)
- **Head 2:** Seasonal patterns
- **Head 3:** Daily patterns
- **Head 4:** Long-term trends

---

## 📊 Dashboard Output

### Retrofit Priority Report
```csv
rank,building_id,primary_use,square_feet,anomaly_score,avg_consumption_kwh,potential_savings_kwh,priority
1,1234,Office,50000,2.45,1500,300,HIGH
2,5678,Education,75000,2.12,2000,400,HIGH
...
```

### 4-Panel Dashboard
Generated at: `artifacts/plots/retrofit_dashboard.png`

1. **Priority by Building Type** - Anomaly score per use type
2. **Potential Savings** - Scatter plot (size vs savings)
3. **Top Anomalies** - Horizontal bar chart
4. **Summary Metrics** - Key statistics

---

## 🖥️ Hardware Requirements

### Minimum
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 16GB
- Storage: 10GB free space
- GPU: Not required (CPU training possible but slow)

### Recommended
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 32GB
- Storage: 20GB SSD
- GPU: NVIDIA RTX 3060+ (6GB VRAM)
- CUDA: 11.8 or 12.1

---

## 🌐 Web Dashboard

AttnRetrofit dilengkapi dengan **Web Dashboard** interaktif menggunakan **FastAPI + React + Three.js**:

### Features
- 🏢 **3D Building Visualization** - Visualisasi gedung dengan warna berdasarkan anomaly score
- 📊 **Interactive Charts** - Distribusi anomaly, top buildings, trend analysis
- 📋 **Building Table** - Searchable, sortable daftar gedung
- 🎯 **Priority Ranking** - Retrofit recommendations dengan scoring
- 📈 **Real-time Metrics** - Model performance dashboard

### Quick Start

#### Option 1: Using Batch Script (Windows)
```batch
start_dashboard.bat
```

#### Option 2: Manual
```bash
# Activate conda environment
conda activate attnretrofit

# Install FastAPI dependencies (jika belum)
pip install fastapi uvicorn

# Start server
python api_server.py
```

#### Then Open Browser
- **Dashboard:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/health` | GET | Health check |
| `/api/metrics` | GET | Model performance metrics |
| `/api/summary` | GET | Dashboard summary statistics |
| `/api/buildings` | GET | Building list with anomaly scores |
| `/api/building/{id}` | GET | Single building details |
| `/api/chart/anomaly-distribution` | GET | Anomaly score histogram |
| `/api/chart/top-anomalies` | GET | Top anomalous buildings |

### Dashboard Preview
Generated at: `artifacts/plots/retrofit_dashboard.png`

**Tech Stack:**
- **Backend:** FastAPI + Python
- **Frontend:** React 18 (CDN) + Three.js + Chart.js
- **UI:** Modern dark theme dengan glassmorphism effects

---

## 📚 Features List

### Input Features (14)
1. `meter_reading` - Current energy reading
2. `air_temperature` - Outdoor temperature
3. `dew_temperature` - Dew point
4. `hour_sin/cos` - Cyclical hour encoding
5. `dow_sin/cos` - Cyclical day of week
6. `month_sin/cos` - Cyclical month
7. `square_feet_log` - Log-transformed building size
8. `building_age` - Years since construction
9. `temp_diff` - Air temp - Dew temp
10. `meter_reading_lag_24h` - 24-hour lag
11. `meter_reading_roll_mean_24h` - 24-hour rolling mean

---

## 🔮 Future Improvements

1. **Web Dashboard** - Streamlit/Dash interactive dashboard
2. **Real-time Inference** - FastAPI deployment
3. **Multi-meter Support** - Extend to all meter types
4. **Transfer Learning** - Pre-trained model untuk gedung baru
5. **Uncertainty Quantification** - Confidence intervals untuk predictions

---

## 📝 Citation

Jika menggunakan project ini, mohon cite:
```bibtex
@misc{attnretrofit2026,
  author = {Your Name},
  title = {AttnRetrofit: Attention-Based Energy Forecasting untuk Smart Building Retrofit Prioritization},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/AttnRetrofit}
}
```

---

## 📄 License

MIT License - lihat file [LICENSE](LICENSE) untuk detail.

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📞 Contact

- **Author:** [Your Name]
- **Email:** your.email@example.com
- **LinkedIn:** [Your LinkedIn]

---

<p align="center">
  Made with ❤️ for <strong>SDG 13 - Climate Action</strong>
</p>
