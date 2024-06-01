import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv('spam.csv', encoding='latin1')

# Data preprocessing
# Assume 'text' is the column containing SMS messages
data['v2'] = data['v2'].str.lower()  # Convert text to lowercase

# TF-IDF feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
tfidf_features = tfidf_vectorizer.fit_transform(data['v2'])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, data['v1'], test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
