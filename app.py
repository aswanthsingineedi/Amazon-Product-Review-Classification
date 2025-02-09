from flask import Flask, render_template, request, jsonify
from model import predict_sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review = request.form.get('review')
        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400

        # Predict sentiment
        sentiment = predict_sentiment(review)

        # Return the result
        return jsonify({'review': review, 'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
