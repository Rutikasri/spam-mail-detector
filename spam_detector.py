import pandas as pd
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load dataset
data = pd.read_csv("SMSSpamCollection", sep='\t', names=["label", "message"])

# Clean text
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data["cleaned"] = data["message"].apply(clean_text)

# Encode labels
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(data["cleaned"], data["label_num"], test_size=0.2, random_state=42)

# Vectorize
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(cv, "vectorizer.pkl")

print("✅ Model trained and saved successfully!")
