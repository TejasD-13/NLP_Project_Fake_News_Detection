# TruthLens – Fake News Detection System

A simple, attractive, and fully functional **Streamlit** frontend for Fake News Detection using Machine Learning and Natural Language Processing, trained on the **WELFake dataset**.

---

## 📌 Project Overview

TruthLens is an end-to-end NLP and Machine Learning solution designed to classify news articles as either **Fake News** or **Real News**. It integrates pre-trained models trained on 72,134 news records from the benchmark WELFake dataset.

### Benchmark Highlights:
- **Original Dataset:** 72,134 records
- **Cleaned Dataset:** 63,289 records (Deduplicated & validated)
- **Fake News Records:** 34,519 (54.54%)
- **Real News Records:** 28,770 (45.46%)
- **Default Best Model:** **Linear SVM** with **96.94% Accuracy** and **96.92% Macro F1-Score**

---

## 🛠️ Technology Stack

- **Language:** Python 3.11
- **Frontend Framework:** Streamlit
- **Machine Learning:** Scikit-learn, Joblib
- **Natural Language Processing:** NLTK (Stopwords, WordNetLemmatizer)
- **Data Manipulation:** Pandas, NumPy
- **Interactive Visualizations:** Plotly Express & Graph Objects

---

## 🏆 Model Evaluation Results

All four models were trained and evaluated using stratified 80/20 train-test splits on 12,658 test articles. Metrics are macro-averaged:

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| **Linear SVM** *(Default)* | **96.94%** | **96.92%** | **96.91%** | **96.92%** | **99.58%** | 2.02s |
| **Logistic Regression** | 96.11% | 96.08% | 96.08% | 96.08% | 99.35% | 1.24s |
| **Random Forest** | 93.62% | 93.55% | 93.58% | 93.57% | 98.49% | 14.11s |
| **Multinomial Naive Bayes** | 88.36% | 88.22% | 88.45% | 88.30% | 95.28% | 0.27s |

---

## 📁 Project Structure

```
Fake news detection/
│
├── app.py                    # Streamlit web application with 3-page navigation
├── utils.py                  # Text preprocessing, artifact loading & prediction logic
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation and run guide
├── code.ipynb                # Original Jupyter notebook with ML pipeline
├── WELFake_Dataset.csv       # Benchmark WELFake dataset
│
└── saved_model/
    ├── best_fake_news_model.pkl       # Best trained model (Linear SVM)
    ├── tfidf_vectorizer.pkl           # TF-IDF Vectorizer (30,000 max features)
    ├── model_evaluation_results.pkl   # Actual evaluation metrics dataframe
    └── all_trained_models.pkl         # Dictionary containing all four trained models
```

---

## ⚙️ NLP Preprocessing Pipeline

The system ensures 100% preprocessing consistency with the Jupyter Notebook:
1. **Case Normalization:** Converts all input text to lowercase.
2. **Noise Removal:** Strips URLs (`http...`, `www...`), HTML tags, and email addresses.
3. **Character Cleansing:** Retains only alphabetic letters (`a-z`) and spaces.
4. **Tokenization:** Extracts word tokens using regular expressions.
5. **Stopword Removal:** Eliminates standard English stopwords while strictly preserving negation keywords (`no`, `not`, `nor`, `never`).
6. **Lemmatization:** Applies WordNet lemmatization on tokens with length > 1.
7. **Vectorization:** Transforms cleaned text into 30,000-dimensional TF-IDF vectors using unigrams and bigrams (`ngram_range=(1, 2)`).

---

## 🚀 Installation & Running the Application

### 1. Set Up Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Frontend

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🧭 Application Pages

1. **Home:** Professional dashboard summarizing project objectives, dataset statistics, class distribution donut chart, model descriptions, and visual 6-stage NLP workflow.
2. **Fake News Detection:** Interactive classifier with headline and article inputs, sample news selector, model dropdown (Linear SVM default), prediction indicators (Red for Fake, Green for Real), token count inspection, and preprocessed text viewer.
3. **Model Performance:** Comparative evaluation table, accuracy bar chart, ROC-AUC comparison, macro precision/recall/F1 grouped bar chart, and test confusion matrices for all 4 models.

---

## ⚠️ Academic Disclaimer

*This system identifies linguistic patterns learned from the WELFake dataset. It does not independently verify the factual accuracy of real-world events.*
