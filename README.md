# Smart Portfolio
## 📈 Stock Price Prediction using LSTM (TensorFlow / Keras)

A modular, production-ready implementation of a **stock price prediction system** using **LSTM (Long Short-Term Memory)** neural networks. This project demonstrates clean architecture, separation of concerns, and best practices for building ML pipelines in Python.
In this implementation, the model is trained using **seven input features**, comprising the target stock price along with six global market indicators. This setup forms a **multivariate time-series model**, where the prediction is influenced not only by the stock’s historical prices but also by broader macroeconomic signals.

---

## 🚀 Features

* 📊 Fetch stock and index data using **Yahoo Finance**
* 🧹 Data preprocessing & normalization
* 🔄 Time-series dataset creation for LSTM
* 🧠 Deep Learning model using **Stacked LSTM**
* 💾 Model caching & reuse
* 📉 Prediction visualization (Actual vs Predicted)
* 🧱 Clean, modular architecture (production-ready)

---

## 🏗️ Project Structure

```
.
├── config/
│   └── model_config.py        # Configuration (hyperparameters, paths)
│
├── data/
│   ├── data_provider.py      # Fetch stock & index data
│   └── data_processor.py     # Data cleaning, splitting, normalization
│
├── models/
│   └── lstm_model.py         # LSTM model definition & loading
│
├── training/
│   └── model_trainer.py      # Training, prediction, plotting
│
├── cache/                    # Cached datasets
├── models/                   # Saved models
│
├── univariant_test.py        # Entry point, build, train and test the LSTM model
├── run_predict.py            # Future price prediction
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/stock-lstm-predictor.git
cd stock-lstm-predictor
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 📦 Required Libraries

* tensorflow / keras
* numpy
* pandas
* matplotlib
* yfinance

---

## ▶️ Usage

Run the main program:

```bash
python multivariant_test.py
```

---

## ⚡ Configuration

All parameters are managed via:

```bash
config/model_config.py
```

### Example:

```bash
ModelConfig(
    stock_name="HDFCBANK.NS",
    look_back=60,
    epochs=20,
    batch_size=1
)
```

### Key Parameters

| Parameter     | Description              |
| ------------- | ------------------------ |
| `stock_name`  | Target stock ticker      |
| `index_list`  | Market indicators        |
| `look_back`   | Time steps for LSTM      |
| `features`    | Number of input features |
| `epochs`      | Training epochs          |
| `batch_size`  | Batch size               |
| `cache_data`  | Enable dataset caching   |
| `cache_model` | Enable model reuse       |

---

## 🔄 Workflow

1. **Data Fetching**

   * Stock + index data downloaded via Yahoo Finance

2. **Preprocessing**

   * Merge datasets
   * Handle missing values
   * Normalize data

3. **Dataset Creation**

   * Convert into supervised learning format (X, y)

4. **Model Training**

   * Stacked LSTM network
   * Mean Squared Error loss

5. **Prediction**

   * Train & test predictions generated

6. **Visualization**

   * Actual vs predicted stock prices plotted

---

## 🧠 Model Architecture

```
Input → LSTM(200, return_sequences=True)
      → LSTM(200)
      → Dense(1)
      → Output
```

---

## 📊 Output Example

* 📈 Actual vs Predicted price graph
* Model learns temporal dependencies in stock movement

---

## ⚠️ Notes & Best Practices

* LSTM works best with **scaled data** ✔
* Avoid `reset_states()` unless using **stateful LSTM**
* Use **shuffle=False** for time series ✔
* Ensure enough data for meaningful learning

---

## 🔧 Future Improvements

* ✅ Add EarlyStopping & Learning Rate Scheduler
* ✅ Hyperparameter tuning (Optuna / Grid Search)
* ✅ Multi-stock training pipeline
* ✅ Model evaluation metrics (RMSE, MAPE)
* ✅ Deploy as REST API (FastAPI)
* ✅ Add Transformer-based models

---

## 📌 Example Stocks

* HDFCBANK.NS
* RELIANCE.NS
* TCS.NS
* INFY.NS

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Brijesh S**

---

## ⭐ If you found this useful

Give this repo a ⭐ and share it!

---

## 📌 Notes
This is not production-grade code. It’s a simple demonstration of concepts. For deep dives into SmartData design and architecture, feel free to reach out: 📧 brij_joe@yahoo.com