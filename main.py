import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the IMDB Movie Reviews Dataset
df = pd.read_csv('IMDB Dataset.csv')

# Text Preprocessing Function
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove non-letter characters (only keep letters)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Convert text to lowercase
    text = text.lower()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing to the review column
df['review'] = df['review'].apply(preprocess_text)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['review'],
df['sentiment'], test_size=0.2, random_state=42)

# Convert labels to binary: positive -> 1, negative -> 0
y_train = y_train.apply(lambda x: 1 if x == 'positive' else 0)
y_test = y_test.apply(lambda x: 1 if x == 'positive' else 0)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# List of classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'SVM (Support Vector Machine)': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5)
}

# Store metrics for each classifier
metrics = {
    'Classifier': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

# Train, predict, and collect metrics for each classifier
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train_vec, y_train)
    # Make predictions
    y_pred = clf.predict(X_test_vec)
    # Collect metrics
    metrics['Classifier'].append(name)
    metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['Precision'].append(precision_score(y_test, y_pred))
    metrics['Recall'].append(recall_score(y_test, y_pred))
    metrics['F1-Score'].append(f1_score(y_test, y_pred))

# Convert metrics dictionary to a DataFrame for easy plotting
metrics_df = pd.DataFrame(metrics)

# Plot the results for each metric
plt.figure(figsize=(12, 8))

# Plot Accuracy
plt.subplot(2, 2, 1)
sns.barplot(x='Accuracy', y='Classifier', data=metrics_df, palette='Blues_d')
plt.title('Accuracy Comparison')

# Plot Precision
plt.subplot(2, 2, 2)
sns.barplot(x='Precision', y='Classifier', data=metrics_df, palette='Greens_d')
plt.title('Precision Comparison')

# Plot Recall
plt.subplot(2, 2, 3)
sns.barplot(x='Recall', y='Classifier', data=metrics_df, palette='Oranges_d')
plt.title('Recall Comparison')

# Plot F1-Score
plt.subplot(2, 2, 4)
sns.barplot(x='F1-Score', y='Classifier', data=metrics_df, palette='Purples_d')

plt.title('F1-Score Comparison')
plt.tight_layout()

plt.show()
