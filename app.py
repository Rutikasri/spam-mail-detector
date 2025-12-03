from flask import Flask, request, render_template
import joblib
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

app = Flask(__name__)

model = joblib.load("spam_model.pkl")
cv = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    cleaned = clean_text(message)
    data = cv.transform([cleaned])
    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "🚨 Spam Message"
    else:
        result = "✅ Not Spam"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
