# 📦 Sakto: AI-Driven Stock Management System

**Sakto** is an AI-powered stock management and demand trend prediction system designed to help retail businesses make data-driven inventory decisions. It combines **machine learning**, **numerical analysis**, and **NLP-based feature extraction** to predict demand trends and generate actionable inventory recommendations.

---

## 🚀 Key Features

- 📊 Demand trend prediction using a trained neural network  
- 🧠 Machine learning model for inventory forecasting  
- 🔢 Feature engineering from retail inventory data  
- 📈 Automated inventory decision rules  
- 💾 Model persistence and CSV-based prediction outputs  
- 🧪 Notebook-based experimentation + script-based training  

---

## 🛠️ Technologies Used

- Python 3  
- PyTorch  
- scikit-learn  
- Pandas / NumPy  
- Jupyter Notebook  
- Joblib  

---

## 📂 Project Structure

```bash
sakto-stock-management/
│
├── data/
│   └── retail_store_inventory.csv
│
├── models/
│   ├── label_map.joblib
│   └── trend_model.pt
│
├── notebooks/
│   └── sakto_stock_management.ipynb
│
├── predictions/
│   └── inventory_predictions.csv
│
├── src/
│   ├── data/
│   ├── models/
│   ├── templates/
│   ├── app.py
│   ├── data_generator.py
│   ├── krr_engine.py
│   └── train_model.py
│
├── data.txt
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📘 Workflow Overview

### 1️⃣ Data Preparation
- Inventory data stored in `data/retail_store_inventory.csv`
- Optional synthetic data generation via `data_generator.py`

### 2️⃣ Feature Engineering & Modeling
- Numerical features are processed
- Demand trends learned via neural network
- Training logic implemented in `train_model.py`

### 3️⃣ Model Training
- Model saved to `models/trend_model.pt`
- Label mappings saved to `models/label_map.joblib`

### 4️⃣ Prediction & Decision Making
- Predictions saved to `predictions/inventory_predictions.csv`
- Inventory actions derived from demand trends

---

## 📊 Demand Trend Classes

| Class | Description |
|------|------------|
| 0 | Decreasing Demand |
| 1 | Stable Demand |
| 2 | Increasing Demand |

---

## 🧮 Inventory Decision Rules

| Demand Trend | Action |
|-------------|--------|
| Increasing | Restock |
| Stable | Maintain |
| Decreasing | Reduce |

---

## ▶️ How to Run

### 1. Clone Repository
```bash
git clone https://github.com/your-username/sakto-stock-management.git
cd sakto-stock-management
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook notebooks/sakto_stock_management.ipynb
```

### 4. Train the Model
```bash
python src/train_model.py
```

### 5. Run the app
```bash
python src/app.py
```

---

## 🎓 Use Cases

- Retail inventory optimization  
- Demand forecasting  
- Academic & capstone projects  
- ML portfolio demonstrations  

---

## 🔮 Future Improvements

- Web dashboard
- Real-time inventory sync
- Alert & notification system
- Model explainability
- Flask/Django deployment

---

## 👨‍💻 Author

**Nigel Agojo, Lance Vincent Gallardo, Ross Cedric Nazareno**  
Computer Science Student  
Laguna State Polytechnic University  

---

## 📄 License

This project is for **educational and research purposes**.
