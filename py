# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load dataset
# You can download dataset from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv("/content/spam.csv", encoding="latin-1")

# Keep only needed columns
df = df[['v1','v2']]
df.columns = ['label','message']

print(df.head())

# 3. Convert labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham':0, 'spam':1})

# 4. Split data
x_train, x_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 5. Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# 6. Train Naive Bayes model
model = MultinomialNB()
model.fit(x_train_vec, y_train)

# 7. Predictions
y_pred = model.predict(x_test_vec)

# 8. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Test with custom input
def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]
    return "Spam 🚨" if prediction == 1 else "Ham ✅"

print(predict_message("Congratulations! You won a free lottery ticket."))
print(predict_message("Hi, are we meeting tomorrow?"))
