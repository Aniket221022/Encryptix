import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to load and process data
def load_data(filepath, is_train=True):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if is_train:
                if len(parts) == 4:
                    _, _, genre, description = parts
                    data.append((description, genre))
            else:
                if len(parts) == 3:
                    _, _, description = parts
                    data.append((description,))
    return pd.DataFrame(data, columns=['description', 'genre'] if is_train else ['description'])

# Load the data
train_df = load_data('train_data.txt', is_train=True)
test_df = load_data('test_data.txt', is_train=False)

# Show a sample of the data
print(train_df.head())
print(test_df.head())

# Split the training data
X_train = train_df['description']
y_train = train_df['genre']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Split the training data to include a validation set
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train_split, y_train_split)

# Predict on the validation split
y_val_pred = model.predict(X_val_split)

# Calculate the accuracy
accuracy = accuracy_score(y_val_split, y_val_pred)

# Print the accuracy
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Transform the test data
X_test = test_df['description']
X_test_tfidf = tfidf.transform(X_test)

# Predict on the test data
y_test_pred = model.predict(X_test_tfidf)

# Print the predictions for the test data
print(f'Test Predictions: {y_test_pred}')
