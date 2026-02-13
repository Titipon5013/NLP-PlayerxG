# ⚽ PlayerxG - AI Scouting Engine

**PlayerxG** is an NLP-powered analytics dashboard that predicts 
**Expected Goals (xG)** based on match commentary text. Unlike traditional xG models that rely on coordinates, this project uses 
**Natural Language Processing (NLP)** to analyze the semantic context of a shot (e.g., "powerful volley", "header", "tight angle") to quantify scoring probabilities.

## 🚀 Features
* **Text-Based xG:** Predicts goal probability using TF-IDF and Logistic Regression on match commentary.
* **Technique Analysis:** Extracts shot techniques (Left Foot, Right Foot, Header) using Regex.
* **Zone Analysis:** Identifies shooting zones (Penalty Area, Outside Box, Six Yard Box) using Regex.
* **Explainable AI:** Highlights specific keywords (e.g., "penalty", "open goal") that influence the xG score.
* **Interactive Dashboard:** Built with Streamlit & Altair for responsive data visualization.

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Titipon5013/NLP-PlayerxG.git
cd NLP-PlayerxG
```

### 2. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn altair joblib matplotlib seaborn
```

### 3. Train the Model (ETL Pipeline)
```bash
python train_model.py
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```

