# phishing-detection-ml
# 🔐 Phishing Website Detection System

A machine learning–based phishing detection system that classifies URLs as **legitimate or malicious** using engineered security features and supervised learning models.

This project demonstrates the practical application of cybersecurity and machine learning techniques to identify real-world phishing threats.

---

## 🚀 Project Overview

Phishing attacks remain one of the most common cyber threats. This system analyzes URL structure and domain-based indicators to detect suspicious websites.

The pipeline:

Input URL → Feature Extraction → ML Model → Prediction → Result

---

## 🧠 Key Features

- URL feature engineering (length, special characters, structure)
- SSL and domain-based checks
- Supervised ML classification
- Performance evaluation (Precision, Recall, F1-score, ROC)
- Real-time prediction capability
- Modular Python implementation

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib  
- **Models:** Random Forest / Logistic Regression  
- **Environment:** Python 3.x  

---

## 📊 Dataset

The model was trained on publicly available datasets of phishing and legitimate URLs.

Example extracted features:

- URL length  
- Presence of special characters  
- HTTPS usage  
- Domain characteristics  
- Suspicious keyword patterns  

---

## 📁 Project Structure
phishing-detection-ml/
│
├── src/ # Source code
├── data/ # Dataset files
├── notebooks/ # Experiments
├── models/ # Saved models
├── images/ # Screenshots
├── requirements.txt
└── main.py


---

## ▶️ Installation & Usage

### 1️⃣ Clone the repository

bash
git clone https://github.com/YOUR-USERNAME/phishing-detection-ml.git
cd phishing-detection-ml
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the project
python main.py "http://example.com"
📈 Model Performance

⚠️ Replace these with your real results.

Metric	Score
Accuracy	--
Precision	--
Recall	--
F1 Score	--

🔮 Future Improvements

Deep learning–based detection

Browser extension integration

Email header analysis

Live phishing feed integration

REST API deployment

👤 Author

Devansh Jamdar
BSc Computer Science (Cybersecurity)
University of Hertfordshire

📫 LinkedIn: https://linkedin.com/in/devansh-jamdar-34b6b620b

