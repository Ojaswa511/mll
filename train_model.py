from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

print("Loading data...")
digits = load_digits()

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Making predictions...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("Saving model...")
with open('digit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Done! Model saved as 'digit_model.pkl'")