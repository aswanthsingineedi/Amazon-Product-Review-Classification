import joblib
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

# Load the pre-trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define preprocessing functions
stop_words = set(stopwords.words('english'))

def clean_text(text):
    #text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    words = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

# Predict function
def predict_sentiment(review):
    processed_review = lemmatize_text(clean_text(review))
    vectorized_review = vectorizer.transform([processed_review])
    prediction = model.predict(vectorized_review)
    return "Positive" if prediction[0] == 2 else "Negative"
