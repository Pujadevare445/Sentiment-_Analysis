# app.py
from flask import Flask, render_request, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model & Vectorizer
# Ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    model = None
    vectorizer = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Dashboard</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 600px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 8px;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 30px;
            font-size: 0.95rem;
        }

        form {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        textarea {
            width: 100%;
            height: 140px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 16px;
            color: #f8fafc;
            font-size: 1rem;
            resize: none;
            outline: none;
            transition: all 0.3s ease;
        }

        textarea:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        button {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            padding: 14px;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
        }

        .result-card {
            margin-top: 30px;
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            animation: fadeIn 0.4s ease-in-out;
        }

        .positive {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: #4ade80;
        }

        .negative {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
        }

        .sentiment-title {
            font-size: 1.25rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }

        .confidence {
            font-size: 0.9rem;
            opacity: 0.85;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis</h1>
        <p class="subtitle">Enter text below to detect its underlying tone.</p>
        
        <form action="/" method="POST">
            <textarea name="text" placeholder="Type or paste text here..." required>{{ text }}</textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>

        {% if sentiment %}
        <div class="result-card {{ sentiment.lower() }}">
            <div class="sentiment-title">{{ sentiment }} Sentiment</div>
            {% if confidence %}
            <div class="confidence">Confidence score: {{ confidence }}%</div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    confidence = None
    text = ""
    
    if request.method == 'POST':
        text = request.form.get('text', '')
        if text and model and vectorizer:
            transformed_text = vectorizer.transform([text])
            prediction = model.predict(transformed_text)[0]
            
            # Map output label
            sentiment = str(prediction).capitalize()

            # Calculate confidence score if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(transformed_text)[0]
                confidence = round(np.max(probs) * 100, 2)

    return render_template_string(HTML_TEMPLATE, sentiment=sentiment, confidence=confidence, text=text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
