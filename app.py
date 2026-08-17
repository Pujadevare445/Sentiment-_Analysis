import os
import pickle
from flask import Flask, request, render_template_string

app = Flask(__name__)

# --- LOAD MODEL ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# --- EMBEDDED GUI (HTML + modern CSS) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --pos-color: #22c55e;
            --neg-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: var(--card-bg);
            width: 100%;
            max-width: 600px;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        textarea {
            width: 100%;
            height: 140px;
            background-color: rgba(15, 23, 42, 0.6);
            border: 2px solid #334155;
            border-radius: 12px;
            padding: 16px;
            color: var(--text-main);
            font-size: 1rem;
            resize: none;
            outline: none;
            transition: border-color 0.2s ease;
        }

        textarea:focus {
            border-color: var(--accent);
        }

        .btn {
            width: 100%;
            background: var(--accent);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
            transition: background 0.2s ease;
        }

        .btn:hover {
            background: var(--accent-hover);
        }

        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            font-size: 1.2rem;
            animation: fadeIn 0.3s ease-in-out;
        }

        .result-positive {
            background-color: rgba(34, 197, 94, 0.1);
            color: var(--pos-color);
            border: 1px solid var(--pos-color);
        }

        .result-negative {
            background-color: rgba(239, 68, 68, 0.1);
            color: var(--neg-color);
            border: 1px solid var(--neg-color);
        }

        .error-box {
            background-color: rgba(239, 68, 68, 0.1);
            color: var(--neg-color);
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 0.9rem;
            text-align: center;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sentiment Analysis AI</h1>
            <p>Analyze the emotion behind your text in real time</p>
        </div>

        {% if error %}
            <div class="error-box">{{ error }}</div>
        {% endif %}

        <form method="POST" action="/predict">
            <textarea name="text" placeholder="Type or paste your text here..." required>{{ user_text }}</textarea>
            <button type="submit" class="btn">Analyze Sentiment</button>
        </form>

        {% if prediction %}
            <div class="result-box {% if prediction == 'positive' %}result-positive{% else %}result-negative{% endif %}">
                Result: {{ prediction|capitalize }} Sentiment
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, prediction=None, user_text="")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template_string(
            HTML_TEMPLATE, 
            error="Model file (model.pkl) not loaded properly.", 
            prediction=None, 
            user_text=""
        )

    user_text = request.form.get('text', '')
    
    if user_text.strip():
        try:
            # Predict using your MultinomialNB model
            prediction_array = model.predict([user_text])
            prediction = prediction_array[0]
        except Exception as e:
            return render_template_string(
                HTML_TEMPLATE, 
                error=f"Prediction failed: {str(e)}", 
                prediction=None, 
                user_text=user_text
            )
    else:
        prediction = None

    return render_template_string(HTML_TEMPLATE, prediction=prediction, user_text=user_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
