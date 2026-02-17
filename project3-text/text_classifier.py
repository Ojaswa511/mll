from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

print("Loading text data...")
# Using only 2 categories for simplicity
categories = ['alt.atheism', 'talk.religion.misc']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, 
                                 remove=('headers', 'footers', 'quotes'))

print(f"Total documents: {len(newsgroups.data)}")
print(f"Categories: {newsgroups.target_names}")

print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.2, random_state=42
)

print("\nConverting text to numbers (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model 1: Naive Bayes
print("\n--- Naive Bayes ---")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"Accuracy: {nb_acc:.4f}")

# Model 2: Logistic Regression
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Accuracy: {lr_acc:.4f}")

# Save the better model AND the vectorizer
if lr_acc > nb_acc:
    print("\nLogistic Regression performed better! Saving it...")
    with open('text_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
else:
    print("\nNaive Bayes performed better! Saving it...")
    with open('text_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

print("Model and vectorizer saved!")
print("\nDone!")