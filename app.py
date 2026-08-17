import os
import pickle
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# --- MODEL LOADING ---
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    # Load Model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    elif os.path.exists("model_file.pkl"): # Alternative fallback name
        with open("model_file.pkl", "rb") as f:
            model = pickle.load(f)

    # Load Vectorizer
    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    elif os.path.exists("tfidf.pkl"): # Alternative fallback name
        with open("tfidf.pkl", "rb") as f:
            vectorizer = pickle.load(f)

load_artifacts()

# --- UI TEMPLATE (HTML + Tailwind CSS + Modern Animations) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Sentiment Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .glass {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen flex items-center justify-center p-4 text-slate-800">

    <div class="glass w-full max-w-2xl rounded-3xl shadow-2xl p-8 transition-all duration-300">
        <!-- Header -->
        <div class="text-center mb-8">
            <div class="inline-flex items-center justify-center w-16 h-16 bg-indigo-600 text-white rounded-2xl shadow-lg mb-4">
                <i class="fa-solid fa-[#brain] fa-2xl"></i>
            </div>
            <h1 class="text-3xl font-extrabold text-slate-900 tracking-tight">Sentiment Detector</h1>
            <p class="text-slate-500 text-sm mt-1">Analyze textual context instantly using Machine Learning</p>
        </div>

        <!-- Text Input Form -->
        <form id="sentimentForm" class="space-y-4">
            <div>
                <label for="text_input" class="block text-sm font-semibold text-slate-700 mb-2">Enter your text below:</label>
                <textarea 
                    id="text_input" 
                    name="text" 
                    rows="4" 
                    required
                    placeholder="Type or paste customer review, tweet, or message here..."
                    class="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all shadow-sm text-slate-700 placeholder-slate-400"
                ></textarea>
            </div>

            <button 
                type="submit" 
                id="submitBtn"
                class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-xl shadow-lg hover:shadow-indigo-500/30 transition-all flex items-center justify-center gap-2"
            >
                <span>Analyze Sentiment</span>
                <i class="fa-solid fa-wand-magic-sparkles"></i>
            </button>
        </form>

        <!-- Loading Spinner -->
        <div id="loading" class="hidden text-center my-6">
            <i class="fa-solid fa-circle-notch fa-spin text-3xl text-indigo-600"></i>
            <p class="text-sm text-slate-500 mt-2 font-medium">Evaluating text structure...</p>
        </div>

        <!-- Output Result Section -->
        <div id="resultCard" class="hidden mt-6 p-6 rounded-2xl border transition-all">
            <div class="flex items-center justify-between">
                <div>
                    <span class="text-xs font-bold uppercase tracking-wider text-slate-400 block mb-1">Prediction</span>
                    <h3 id="sentimentResult" class="text-2xl font-black capitalize"></h3>
                </div>
                <div id="sentimentIcon" class="text-4xl"></div>
            </div>
            <div id="probabilityContainer" class="mt-4 hidden">
                <div class="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                    <div id="probabilityBar" class="h-2.5 rounded-full transition-all duration-500" style="width: 0%"></div>
                </div>
                <p id="probabilityText" class="text-right text-xs text-slate-500 mt-1 font-semibold"></p>
            </div>
        </div>
    </div>

    <!-- Client-Side Dynamic Interaction -->
    <script>
        document.getElementById('sentimentForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const textInput = document.getElementById('text_input').value;
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const resultCard = document.getElementById('resultCard');
            const sentimentResult = document.getElementById('sentimentResult');
            const sentimentIcon = document.getElementById('sentimentIcon');
            const probContainer = document.getElementById('probabilityContainer');
            const probBar = document.getElementById('probabilityBar');
            const probText = document.getElementById('probabilityText');

            // Set loading UI
            submitBtn.disabled = true;
            loading.classList.remove('hidden');
            resultCard.classList.add('hidden');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: textInput })
                });

                const data = await response.json();

                if (data.error) {
                    alert(data.error);
                } else {
                    sentimentResult.textContent = data.sentiment;
                    
                    // UI styling based on sentiment value
                    const isPositive = data.sentiment.toLowerCase().includes('pos');
                    
                    if (isPositive) {
                        resultCard.className = "mt-6 p-6 rounded-2xl border border-emerald-200 bg-emerald-50/50 text-emerald-900";
                        sentimentIcon.innerHTML = '<i class="fa-solid fa-face-smile text-emerald-500"></i>';
                        probBar.className = "h-2.5 rounded-full bg-emerald-500 transition-all duration-500";
                    } else {
                        resultCard.className = "mt-6 p-6 rounded-2xl border border-rose-200 bg-rose-50/50 text-rose-900";
                        sentimentIcon.innerHTML = '<i class="fa-solid fa-face-frown text-rose-500"></i>';
                        probBar.className = "h-2.5 rounded-full bg-rose-500 transition-all duration-500";
                    }

                    if (data.confidence !== null) {
                        probContainer.classList.remove('hidden');
                        probBar.style.width = `${data.confidence}%`;
                        probText.textContent = `Confidence: ${data.confidence}%`;
                    } else {
                        probContainer.classList.add('hidden');
                    }

                    resultCard.classList.remove('hidden');
                }
            } catch (err) {
                alert("Something went wrong with the connection!");
            } finally {
                submitBtn.disabled = false;
                loading.classList.add('hidden');
            }
        });
    </script>
</body>
</html>
"""

# --- ROUTES ---
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model or vectorizer pickles were not loaded correctly on server."}), 500

    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Empty text provided."}), 400

    # Vectorize and Predict
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    
    # Calculate Probability/Confidence if supported by model
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_text)[0]
        confidence = round(max(probabilities) * 100, 2)

    return jsonify({
        "sentiment": str(prediction),
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
