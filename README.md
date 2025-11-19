# IMDB-Sentiment-Analysis-with-Multiple-Classifiers
Analyzes IMDB movie reviews using TF-IDF features and compares multiple classifiers—Logistic Regression, Naive Bayes, SVM, Random Forest, and KNN—on accuracy, precision, recall, and F1-score with visual performance comparisons.

## 📋 Project Overview

This project implements a binary text classifier to determine sentiment in IMDB movie reviews. It uses natural language processing techniques including HTML tag removal, stopword removal, TF-IDF vectorization, and multiple machine learning classifiers for performance comparison.

## 🎯 Problem Statement

**Can we automatically detect the sentiment of movie reviews based on their text content?**

The classifier predicts:

* **Negative (0)**: Negative sentiment review
* **Positive (1)**: Positive sentiment review

## 🔑 Key Features

* ✅ Text preprocessing with NLTK
* ✅ HTML tag removal
* ✅ Special character filtering
* ✅ Lowercase conversion
* ✅ Stopword removal (English)
* ✅ TF-IDF vectorization (5000 features)
* ✅ Multiple classifiers: Logistic Regression, Naive Bayes, SVM, Random Forest, KNN
* ✅ Comprehensive performance metrics: Accuracy, Precision, Recall, F1-Score
* ✅ Visualization of model comparison

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/notfakh/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download the dataset:

   * `IMDB Dataset.csv` from [Google Drive Dataset](https://drive.google.com/file/d/1reS9mUR7Hopn8qPOkOFeKvZzhg8P1lGQ/view?usp=sharing)
   * Place it in the project root directory

### Usage

Run the script:

```bash
python imdb_sentiment_analysis.py
```

**First Run:**

* Downloads NLTK stopwords
* Preprocesses all text reviews
* Trains multiple classifiers
* Displays metrics and plots for accuracy, precision, recall, and F1-score

## 📊 Text Preprocessing Pipeline

### Step-by-Step Process:

1. **HTML Tag Removal**

```python
"<p>Great movie!</p>" → "Great movie"
```

2. **Special Character Filtering**

```python
"Amazing!!! 10/10" → "Amazing"
```

3. **Lowercase Conversion**

```python
"BAD MOVIE" → "bad movie"
```

4. **Stopword Removal**

```python
"this movie is not good" → "movie good"
```

## 📈 Model Architecture

### TF-IDF Vectorization

```python
TfidfVectorizer(max_features=5000, stop_words='english')
```

* Converts text to numerical features based on word importance
* Top 5000 words considered

### Classifiers Used

* **Logistic Regression**
* **Naive Bayes**
* **Support Vector Machine (SVM)**
* **Random Forest**
* **K-Nearest Neighbors (KNN)**

## 📊 Expected Results

### Sample Performance Metrics:

| Metric   | Logistic Regression | Naive Bayes | SVM  | Random Forest | KNN  |
| -------- | ------------------- | ----------- | ---- | ------------- | ---- |
| Accuracy | ~88%                | ~86%        | ~87% | ~85%          | ~81% |
| F1-Score | ~88%                | ~86%        | ~87% | ~85%          | ~81% |

**Classification Report** will show per-class precision, recall, and F1-score.

## 🔍 Dataset Information

**IMDB Movie Reviews Dataset:**

* **Total Reviews:** 50,000
* **Positive Reviews:** 50%
* **Negative Reviews:** 50%
* **Format:** CSV with `review` and `sentiment` columns
* **Source:** Kaggle

## 🛠️ Customization

### Adjust TF-IDF Features

```python
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
```

### Try Different Classifiers

```python
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
```

### Modify Preprocessing

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def preprocess_text(text):
    # ... existing preprocessing ...
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)
```

### Change Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.3, random_state=42
)
```

## 💡 Understanding the Metrics

### Precision

* Of all predicted positives, how many were correct?

### Recall

* Of all actual positives, how many did we catch?

### F1-Score

* Harmonic mean of precision and recall

### Accuracy

* Overall percentage of correct predictions

## 🔬 Extending the Project

* Hyperparameter tuning
* Cross-validation
* Confusion matrix visualization with Seaborn
* Word cloud for most frequent words
* Save and load trained models with Pickle
* Real-time sentiment prediction function

## 🤝 Contributing

Contributions welcome! Ideas include:

* Add deep learning models (LSTM, BERT)
* Implement n-gram analysis
* Create web interface with Flask/Streamlit

## 👤 Author

**Fakhrul Sufian**

* GitHub: [@notfakh](https://github.com/notfakh)
* LinkedIn: [Fakhrul Sufian](https://www.linkedin.com/in/fakhrul-sufian-b51454363/)
* Email: [fkhrlnasry@gmail.com](mailto:fkhrlnasry@gmail.com)

## 🙏 Acknowledgments

* Kaggle for the IMDB dataset
* NLTK library for NLP
* Scikit-learn for machine learning

## 📚 References

* [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* [Scikit-learn Documentation](https://scikit-learn.org/)
* [NLTK Documentation](https://www.nltk.org/)

## 🐛 Troubleshooting

* **NLTK stopwords not found**

```python
import nltk
nltk.download('stopwords')
```

* **Encoding error**

```python
pd.read_csv('IMDB Dataset.csv', encoding='ISO-8859-1')
```

* **Low accuracy**

  * Increase `max_features`
  * Try different classifiers (SVM, Random Forest)
  * Add n-grams: `ngram_range=(1,2)`

## 📧 Contact

* Email: [fkhrlnasry@gmail.com](mailto:fkhrlnasry@gmail.com)
* Open an issue on GitHub

---

⭐ If this project helped you understand text classification and sentiment analysis, please give it a star!
