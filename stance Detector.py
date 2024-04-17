import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV
data = pd.read_csv("stance_dataset.csv")

def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove punctuation and lowercase
    tokens = [word.lower() for word in tokens if word.isalnum()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Preprocess the data
data['text'] = data['Sentence'].apply(preprocess_text)

# Split preprocessed data into features and labels
X = data['text']
y = data['Stance']

# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Apply dimensionality reduction
lsa = TruncatedSVD(n_components=10)
X_lsa = lsa.fit_transform(X_tfidf)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_lsa, y, test_size=0.33, random_state=42, stratify=y)

# Train SVM classifier with grid search for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=2)  # Decrease the number of folds
grid_search.fit(X_train, y_train)

# Predict on test set
y_pred = grid_search.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Best parameters:", grid_search.best_params_)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualization: Confusion Matrix Heatmap
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Visualization: Pie Chart for Stance Distribution
plt.figure(figsize=(6, 6))
data['Stance'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'coral'])
plt.title('Stance Distribution')
plt.ylabel('')
plt.show()

# Visualization: Bar Chart for Hyperparameter Tuning Results
param_results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
sns.barplot(x='param_C', y='mean_test_score', hue='param_gamma', data=param_results)
plt.title('Hyperparameter Tuning Results')
plt.xlabel('C')
plt.ylabel('Mean Test Score')
plt.legend(title='Gamma')
plt.show()

# Visualization: Line Graph for Dimensionality Reduction Explained Variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, lsa.n_components + 1), lsa.explained_variance_ratio_, marker='o', linestyle='-')
plt.title('Explained Variance Ratio by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

# Example usage of the model
user_input = input("Enter a debate statement: ")
preprocessed_text = preprocess_text(user_input)
tfidf_vector = vectorizer.transform([preprocessed_text])
lsa_vector = lsa.transform(tfidf_vector)
prediction = grid_search.predict(lsa_vector)[0]
print("Predicted stance:", prediction)
