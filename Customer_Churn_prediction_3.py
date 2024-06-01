import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_csv('churn_Modelling.csv')

# Preprocess the data
X = data.drop('Exited', axis=1)
y = data['Exited']
X = pd.get_dummies(X)  # One-hot encode categorical variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

classification_rep = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", classification_rep)

classification_rep = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", classification_rep)



print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# Feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("Feature Importance:\n", feature_importance)
