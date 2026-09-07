from flask import Flask, render_template, request, redirect, session, flash, g, url_for, jsonify
import mysql.connector
from werkzeug.utils import secure_filename
import requests
import os
import json
import pandas as pd

from translations import LANGUAGES, LANGUAGE_NAMES, DEFAULT_LANGUAGE, get_text
from disease_data import disease_database

# Optionally load a local .env file (e.g. GEMINI_API_KEY) if python-dotenv
# is installed. Entirely optional — the app works fine with real
# environment variables set instead, and does not fail if the package
# or the .env file is missing.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = Flask(__name__)
app.secret_key = 'smart_agriculture_secret'

# Upload Folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =====================================================
# AI / MACHINE LEARNING MODELS
# =====================================================
# Both models are optional at import time. If the trained artifacts are
# missing (not trained yet) or a dependency (scikit-learn / TensorFlow)
# is not installed / incompatible with the running Python version, the
# app falls back to the original rule-based logic instead of crashing.
# This mirrors real production ML systems, where a model outage should
# degrade gracefully rather than take the whole app down.

ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml", "models")

# ---- Crop recommendation: RandomForestClassifier ----
CROP_MODEL_AVAILABLE = False
crop_model = None
crop_label_encoder = None
crop_scaler = None

try:
    import joblib

    crop_model = joblib.load(os.path.join(ML_DIR, "crop_model.pkl"))
    crop_label_encoder = joblib.load(os.path.join(ML_DIR, "crop_label_encoder.pkl"))
    crop_scaler = joblib.load(os.path.join(ML_DIR, "crop_scaler.pkl"))
    CROP_MODEL_AVAILABLE = True
    print("[AI] Crop recommendation model loaded (RandomForest).")
except Exception as e:
    print(f"[AI] Crop ML model not loaded, using rule-based fallback. Reason: {e}")

CROP_FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def ml_predict_crop(n, p, k, temperature, humidity, ph, rainfall, top_n=5):
    """Return a list of (crop_name, confidence_percent) using the trained
    RandomForest model, sorted by confidence descending."""
    if not CROP_MODEL_AVAILABLE:
        return []

    features = [[n, p, k, temperature, humidity, ph, rainfall]]
    features_scaled = crop_scaler.transform(features)
    probabilities = crop_model.predict_proba(features_scaled)[0]

    ranked = sorted(
        zip(crop_label_encoder.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return [(name, round(float(prob) * 100, 2)) for name, prob in ranked]


def ml_predict_crop_confidence_map(n, p, k, temperature, humidity, ph, rainfall):
    """Return {crop_name: confidence_percent} for every crop the model
    knows about, so it can be combined with the agronomic rule score for
    every crop in crop_database (not just the model's top guess)."""
    if not CROP_MODEL_AVAILABLE:
        return {}

    features = [[n, p, k, temperature, humidity, ph, rainfall]]
    features_scaled = crop_scaler.transform(features)
    probabilities = crop_model.predict_proba(features_scaled)[0]

    return {
        name: round(float(prob) * 100, 2)
        for name, prob in zip(crop_label_encoder.classes_, probabilities)
    }


# ---- Disease detection: CNN (MobileNetV2 transfer learning) ----
DISEASE_MODEL_AVAILABLE = False
disease_model = None
disease_class_indices = {}
DISEASE_IMG_SIZE = (224, 224)

try:
    # TensorFlow can be heavy / version-sensitive, so it is imported lazily
    # and failure here never prevents the rest of the app from starting.
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image

    disease_model = load_model(os.path.join(ML_DIR, "disease_cnn_model.h5"))

    with open(os.path.join(ML_DIR, "disease_class_indices.json")) as f:
        disease_class_indices = json.load(f)

    DISEASE_MODEL_AVAILABLE = True
    print("[AI] Disease detection CNN model loaded (MobileNetV2).")
except Exception as e:
    print(f"[AI] Disease CNN model not loaded, using rule-based fallback. Reason: {e}")


def predict_disease_cnn(filepath):
    """Run the trained CNN on an uploaded leaf image.

    Returns a dict {"disease": str, "confidence": float} on success,
    or None if the model isn't available / prediction fails, so callers
    can fall back to the symptom-based diagnosis engine.
    """
    if not DISEASE_MODEL_AVAILABLE:
        return None

    try:
        img = keras_image.load_img(filepath, target_size=DISEASE_IMG_SIZE)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = disease_model.predict(img_array, verbose=0)[0]
        best_index = int(np.argmax(predictions))
        confidence = float(predictions[best_index]) * 100
        label = disease_class_indices.get(str(best_index), "Unknown")

        return {
            "disease": label.replace("_", " "),
            "confidence": round(confidence, 2),
        }
    except Exception as e:
        print(f"[AI] CNN prediction failed, falling back. Reason: {e}")
        return None


# ---- Disease detection: symptom-based AI (RandomForest) ----
# This is a *second*, independent AI model that predicts disease purely
# from the questionnaire fields (no photo required), trained on
# ml/disease_symptom_dataset.csv via ml/train_disease_symptom_model.py.
# It runs whenever no image is uploaded / the CNN isn't available, so
# the app always has a real trained model behind its diagnosis instead
# of only the original if/else rule scoring.
SYMPTOM_MODEL_AVAILABLE = False
disease_symptom_model = None
disease_symptom_encoders = None
disease_symptom_label_encoder = None
DISEASE_SYMPTOM_CATEGORICAL_FIELDS = [
    "crop", "leaf_color", "spot_color", "spot_size",
    "leaf_curl", "powder", "stem",
]
DISEASE_SYMPTOM_NUMERIC_FIELDS = ["temperature", "humidity"]

try:
    import joblib as _joblib_symptom  # already imported above for crop model, but safe to re-import

    disease_symptom_model = _joblib_symptom.load(
        os.path.join(ML_DIR, "disease_symptom_model.pkl")
    )
    disease_symptom_encoders = _joblib_symptom.load(
        os.path.join(ML_DIR, "disease_symptom_encoders.pkl")
    )
    disease_symptom_label_encoder = _joblib_symptom.load(
        os.path.join(ML_DIR, "disease_symptom_label_encoder.pkl")
    )
    SYMPTOM_MODEL_AVAILABLE = True
    print("[AI] Disease symptom model loaded (RandomForest).")
except Exception as e:
    print(f"[AI] Disease symptom model not loaded, using rule-based fallback. Reason: {e}")


def ml_predict_disease_symptom(crop, leaf_color, spot_color, spot_size,
                                leaf_curl, powder, stem, temperature, humidity,
                                top_n=3):
    """Return a list of (disease_name, confidence_percent) using the
    trained RandomForest symptom model, sorted by confidence descending.
    Unknown categorical values (not seen during training) safely fall
    back to the first known class for that column rather than crashing.
    """
    if not SYMPTOM_MODEL_AVAILABLE:
        return []

    raw = {
        "crop": crop, "leaf_color": leaf_color, "spot_color": spot_color,
        "spot_size": spot_size, "leaf_curl": leaf_curl, "powder": powder,
        "stem": stem,
    }

    encoded_row = []
    for col in DISEASE_SYMPTOM_CATEGORICAL_FIELDS:
        le = disease_symptom_encoders[col]
        value = raw[col]
        if value in le.classes_:
            encoded_row.append(int(le.transform([value])[0]))
        else:
            encoded_row.append(0)  # unseen value -> safest known default

    encoded_row.extend([float(temperature), float(humidity)])

    features = pd.DataFrame(
        [encoded_row],
        columns=DISEASE_SYMPTOM_CATEGORICAL_FIELDS + DISEASE_SYMPTOM_NUMERIC_FIELDS,
    )

    probabilities = disease_symptom_model.predict_proba(features)[0]
    class_labels = disease_symptom_label_encoder.inverse_transform(
        range(len(probabilities))
    )

    ranked = sorted(
        zip(class_labels, probabilities), key=lambda x: x[1], reverse=True
    )[:top_n]

    return [(name, round(float(prob) * 100, 2)) for name, prob in ranked]

# =====================================================
# MULTI-LANGUAGE SUPPORT (English, Hindi, Telugu +
# other Indian state languages)
# =====================================================
# The active language is stored in the session so it persists
# across pages/requests. `t(key)` is exposed to every Jinja
# template (see context_processor below) so templates can write
# {{ t('nav_home') }} instead of hard-coding English text.

SUPPORTED_LANGUAGES = set(LANGUAGES.keys())


@app.before_request
def set_current_language():
    lang = session.get("lang", DEFAULT_LANGUAGE)
    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE
    g.lang = lang


@app.context_processor
def inject_i18n_helpers():
    def t(key):
        return get_text(g.get("lang", DEFAULT_LANGUAGE), key)

    return {
        "t": t,
        "language_names": LANGUAGE_NAMES,
    }


@app.route("/set_language/<lang_code>")
def set_language(lang_code):
    if lang_code in SUPPORTED_LANGUAGES:
        session["lang"] = lang_code
    else:
        flash("Unsupported language selected.")

    next_page = request.args.get("next")
    # Only allow relative redirects (avoid open-redirect issues)
    if next_page and next_page.startswith("/"):
        return redirect(next_page)
    return redirect("/")


# =====================================================
# AI CHATBOT — Flask backend calling the Google Gemini API
# =====================================================
# The chat widget (static/chatbot.js + static/chatbot.css,
# included in base.html / index.html) POSTs each user message
# to /api/chatbot as JSON: {"message": "...", "lang": "hi"}.
#
# SETUP:
#   1. Get a free Gemini API key from https://aistudio.google.com/app/apikey
#   2. Set it as an environment variable before running the app:
#        Windows (PowerShell):  $env:GEMINI_API_KEY="your_key_here"
#        macOS/Linux:           export GEMINI_API_KEY="your_key_here"
#   3. (Optional) Override the model with GEMINI_MODEL, default
#      is "gemini-2.0-flash".
#
# If no key is configured, the endpoint responds with a friendly
# message instead of crashing, so the rest of the site keeps working.

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# Keep a short rolling chat history per browser session so the
# assistant has some conversational context (kept small to avoid
# session bloat / large request payloads to Gemini).
CHAT_HISTORY_LIMIT = 12  # messages (user + bot combined)

CHATBOT_SYSTEM_PROMPT = (
    "You are Krishi Mitra, a friendly and knowledgeable AI assistant "
    "built into the Smart Agriculture Advisor web app for Indian farmers. "
    "You help with crop selection, soil and fertilizer guidance, plant "
    "disease symptoms and treatment, weather-related farming advice, and "
    "general farming best practices. Keep answers practical, encouraging, "
    "and easy for a farmer to follow. Prefer short paragraphs or bullet "
    "points over long essays. If a question is unrelated to agriculture, "
    "politely steer the conversation back to farming topics."
)


def _gemini_language_instruction(lang_code):
    language_name = LANGUAGE_NAMES.get(lang_code, "English")
    if lang_code == "en":
        return "Respond in clear, simple English."
    return (
        f"Respond ONLY in {language_name} ({lang_code}), using natural, "
        f"everyday {language_name} that a farmer would easily understand. "
        "Do not respond in English unless the user explicitly writes in English."
    )


def call_gemini(user_message, lang_code, history):
    """Call the Gemini generateContent REST API.

    `history` is a list of {"role": "user"|"model", "text": str} dicts
    representing the recent conversation (oldest first).
    Returns the assistant's reply text, or raises an exception on failure.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    contents = []
    for turn in history:
        contents.append({
            "role": turn["role"],
            "parts": [{"text": turn["text"]}],
        })
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}],
    })

    payload = {
        "system_instruction": {
            "parts": [{
                "text": CHATBOT_SYSTEM_PROMPT + " " + _gemini_language_instruction(lang_code)
            }]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": 512,
        },
    }

    response = requests.post(
        GEMINI_ENDPOINT,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini API returned no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    reply_text = "".join(part.get("text", "") for part in parts).strip()

    if not reply_text:
        raise RuntimeError("Gemini API returned an empty response")

    return reply_text


@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    lang_code = data.get("lang") or g.get("lang", DEFAULT_LANGUAGE)

    if lang_code not in SUPPORTED_LANGUAGES:
        lang_code = DEFAULT_LANGUAGE

    if not user_message:
        return jsonify({"reply": get_text(lang_code, "chat_placeholder")}), 400

    history = session.get("chat_history", [])

    try:
        reply_text = call_gemini(user_message, lang_code, history)
    except Exception as e:
        print(f"[Chatbot] Gemini API call failed: {e}")
        return jsonify({"reply": get_text(lang_code, "chat_error")}), 200

    # Update rolling history (store in Gemini's role vocabulary)
    history.append({"role": "user", "text": user_message})
    history.append({"role": "model", "text": reply_text})
    session["chat_history"] = history[-CHAT_HISTORY_LIMIT:]

    return jsonify({"reply": reply_text}), 200


@app.route("/api/chatbot/reset", methods=["POST"])
def api_chatbot_reset():
    session.pop("chat_history", None)
    return jsonify({"status": "ok"}), 200


# MySQL Connection
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Sampath@09',   # CHANGE THIS TO YOUR MYSQL PASSWORD
    database='smart_agriculture'
)

cursor = conn.cursor(dictionary=True)

# =========================
# Home Page
# =========================
@app.route('/')
def home():
    return render_template('index.html')

# =========================
# About Page
# =========================
@app.route('/about')
def about():
    return render_template('about.html')

# =========================
# Register Page
# =========================
@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Check if email already exists
        check_query = "SELECT * FROM users WHERE email=%s"
        cursor.execute(check_query, (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Email already registered. Please Login.')
            return redirect('/register')

        # Insert new user
        query = "INSERT INTO users(name, email, password) VALUES(%s,%s,%s)"
        values = (name, email, password)

        cursor.execute(query, values)
        conn.commit()

        flash('Registration Successful')
        return redirect('/login')

    return render_template('register.html')

# =========================
# Login Page
# =========================
@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form['email']
        password = request.form['password']

        query = "SELECT * FROM users WHERE email=%s AND password=%s"
        values = (email, password)

        cursor.execute(query, values)
        user = cursor.fetchone()

        if user:
            session['user'] = user['name']
            return redirect('/dashboard')

        else:
            flash('Invalid Email or Password')

    return render_template('login.html')

# =========================
# Dashboard
# =========================
@app.route('/dashboard')
def dashboard():

    if 'user' not in session:
        return redirect('/login')

    return render_template('dashboard.html')


# =========================
# Advanced Crop Recommendation
# =========================
# =====================================================
# ADVANCED CROP DATABASE - PART 1A (CEREALS)
# =====================================================

# =====================================================
# CROP DATABASE (60 crops)
# =====================================================
# Previously this list was accidentally closed early with a stray
# "]" after just the first 8 cereal crops, so the other 52 crops
# (pulses, oilseeds, vegetables, fruits, plantation crops) were typed
# out but never actually used by the app. Fixed here by loading the
# full, corrected 60-crop dataset from ml/crop_database.json, which
# is also the single source of truth used to train the crop
# recommendation ML model (see ml/generate_crop_dataset.py).
_CROP_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ml", "crop_database.json"
)
with open(_CROP_DB_PATH, "r", encoding="utf-8") as _f:
    crop_database = json.load(_f)

# ======================================================
# SOIL HEALTH ANALYSIS
# ======================================================

def soil_analysis(n, p, k, ph):

    report = []

    score = 100

    # Nitrogen

    if n < 40:
        report.append("Nitrogen : Low")
        report.append("Recommendation : Apply Urea")
        score -= 20

    elif n > 120:
        report.append("Nitrogen : High")
        report.append("Recommendation : Reduce Nitrogen Fertilizer")
        score -= 10

    else:
        report.append("Nitrogen : Normal")

    # Phosphorus

    if p < 20:
        report.append("Phosphorus : Low")
        report.append("Recommendation : Apply DAP")
        score -= 20

    elif p > 80:
        report.append("Phosphorus : High")
        report.append("Recommendation : Reduce DAP")
        score -= 10

    else:
        report.append("Phosphorus : Normal")

    # Potassium

    if k < 20:
        report.append("Potassium : Low")
        report.append("Recommendation : Apply Potash")
        score -= 20

    elif k > 80:
        report.append("Potassium : High")
        report.append("Recommendation : Reduce Potash")
        score -= 10

    else:
        report.append("Potassium : Normal")

    # pH

    if ph < 5.5:
        report.append("Soil is Acidic")
        report.append("Recommendation : Apply Lime")
        score -= 15

    elif ph > 8.0:
        report.append("Soil is Alkaline")
        report.append("Recommendation : Apply Gypsum")
        score -= 15

    else:
        report.append("Soil pH is Normal")

    return report, score

# ==========================================================
## ==========================================================
# ADVANCED CROP RECOMMENDATION ENGINE
# ==========================================================

@app.route('/crop', methods=['GET', 'POST'])
def crop():

    recommendations = []

    if request.method == "POST":

        state = request.form["state"]
        soil = request.form["soil"]
        season = request.form["season"]

        nitrogen = int(request.form["nitrogen"])
        phosphorus = int(request.form["phosphorus"])
        potassium = int(request.form["potassium"])

        ph = float(request.form["ph"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        rainfall = float(request.form["rainfall"])

        # ===============================
        # AI PREDICTION (RandomForest)
        # ===============================
        # Confidence per crop from the trained model, keyed by crop name.
        # Empty dict if the model isn't trained/loaded yet, in which case
        # every crop's ml_confidence defaults to 0 and hybrid_score falls
        # back to the original pure rule-based score below.
        ml_confidence_map = ml_predict_crop_confidence_map(
            nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall
        )

        for crop in crop_database:

            score = 0
            reasons = []

            # ===============================
            # STATE
            # ===============================
            if state in crop["states"]:
                score += 20
                reasons.append("✔ Suitable for selected state")

            # ===============================
            # SOIL
            # ===============================
            if soil in crop["soil"]:
                score += 15
                reasons.append("✔ Suitable soil")

            # ===============================
            # SEASON
            # ===============================
            if "All" in crop["season"] or season in crop["season"]:
                score += 10
                reasons.append("✔ Suitable season")

            # ===============================
            # NITROGEN
            # ===============================
            if crop["N"][0] <= nitrogen <= crop["N"][1]:
                score += 10
                reasons.append("✔ Nitrogen level matched")

            # ===============================
            # PHOSPHORUS
            # ===============================
            if crop["P"][0] <= phosphorus <= crop["P"][1]:
                score += 10
                reasons.append("✔ Phosphorus level matched")

            # ===============================
            # POTASSIUM
            # ===============================
            if crop["K"][0] <= potassium <= crop["K"][1]:
                score += 10
                reasons.append("✔ Potassium level matched")

            # ===============================
            # PH
            # ===============================
            if crop["ph"][0] <= ph <= crop["ph"][1]:
                score += 8
                reasons.append("✔ Soil pH matched")

            # ===============================
            # TEMPERATURE
            # ===============================
            if crop["temp"][0] <= temperature <= crop["temp"][1]:
                score += 7
                reasons.append("✔ Temperature suitable")

            # ===============================
            # HUMIDITY
            # ===============================
            if crop["humidity"][0] <= humidity <= crop["humidity"][1]:
                score += 5
                reasons.append("✔ Humidity suitable")

            # ===============================
            # RAINFALL
            # ===============================
            if crop["rainfall"][0] <= rainfall <= crop["rainfall"][1]:
                score += 5
                reasons.append("✔ Rainfall suitable")

            agro_score = score

            # ===============================
            # HYBRID SCORE (AI + Agronomic Rules)
            # ===============================
            ml_confidence = ml_confidence_map.get(crop["crop"], 0)

            if CROP_MODEL_AVAILABLE:
                # 60% weight on the trained ML model's confidence for this
                # exact soil/climate reading, 40% on the agronomic rule
                # score (state/soil/season/NPK/pH/climate range matches).
                hybrid_score = round((0.6 * ml_confidence) + (0.4 * agro_score), 2)
                reasons.insert(0, f"🤖 AI model confidence: {ml_confidence}%")
            else:
                hybrid_score = agro_score

            # ===============================
            # STAR RATING (based on hybrid score)
            # ===============================
            if hybrid_score >= 90:

                stars = "★★★★★"
                status = "Excellent"

            elif hybrid_score >= 75:

                stars = "★★★★☆"
                status = "Very Good"

            elif hybrid_score >= 60:

                stars = "★★★☆☆"
                status = "Good"

            elif hybrid_score >= 45:

                stars = "★★☆☆☆"
                status = "Average"

            else:

                stars = "★☆☆☆☆"
                status = "Not Recommended"

            if hybrid_score >= 30:

                recommendations.append({

                    "crop": crop["crop"],

                    "score": hybrid_score,

                    "ml_confidence": ml_confidence,

                    "agro_score": agro_score,

                    "status": status,

                    "stars": stars,

                    "water": crop["water"],

                    "yield": crop["yield"],

                    "duration": crop["duration"],

                    "fertilizer": crop["fertilizer"],

                    "profit": crop["profit"],

                    "diseases": crop["diseases"],

                    "reasons": reasons

                })

        recommendations.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        recommendations = recommendations[:20]

    return render_template(
        "crop.html",
        recommendations=recommendations,
        ai_active=CROP_MODEL_AVAILABLE
    )

# =========================
# Dynamic Weekly Weather Prediction
# =========================
import requests

@app.route('/weather', methods=['GET', 'POST'])
def weather():

    weather_list = []
    city = ""
    error = ""

    if request.method == 'POST':

        city = request.form.get('city')

        API_KEY = "2c73ddd1a17cdfb68b530e439ba49796"

        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"

        try:

            response = requests.get(url)

            data = response.json()

            print(data)  # Check terminal output

            if str(data.get("cod")) == "200":

                forecasts = data["list"]

                added_dates = set()

                for item in forecasts:

                    date = item["dt_txt"].split(" ")[0]

                    if date not in added_dates:

                        weather = {

                            "day": date,

                            "temperature": f"{item['main']['temp']} °C",

                            "humidity": f"{item['main']['humidity']} %",

                            "condition": item['weather'][0]['description'].title()

                        }

                        weather_list.append(weather)

                        added_dates.add(date)

                    if len(weather_list) == 7:
                        break

            else:

                error = data.get("message", "Unable to fetch weather data.")

        except Exception as e:

            error = str(e)

    return render_template(
        'weather.html',
        weather_list=weather_list,
        city=city,
        error=error
    )


# =========================
# Fertilizer Suggestion System
# =========================
# =========================
# Fertilizer Suggestion System
# =========================
@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():

    fertilizer_result = {}
    crop_name = ""

    if request.method == 'POST':

        crop_name = request.form.get('crop_name', '').lower()

        fertilizer_data = {

            "rice": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Neem Cake", "Organic Compost", "Vermicompost"],
                "Average": ["Bio Fertilizer", "Zinc Sulphate", "Ammonium Sulphate"]
            },

            "wheat": {
                "Best": ["Urea", "DAP", "Gypsum"],
                "Good": ["Potash", "Organic Manure", "Bone Meal"],
                "Average": ["Neem Cake", "Bio Fertilizer", "Vermicompost"]
            },

            "maize": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Organic Compost", "Gypsum", "Neem Cake"],
                "Average": ["Bio Fertilizer", "Zinc Sulphate", "Vermicompost"]
            },

            "cotton": {
                "Best": ["Urea", "Super Phosphate", "Potash"],
                "Good": ["Magnesium Sulphate", "Organic Compost", "Neem Cake"],
                "Average": ["Bio Fertilizer", "Micronutrient Mix", "Farmyard Manure"]
            },

            "sugarcane": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Press Mud", "Organic Compost", "Vermicompost"],
                "Average": ["Gypsum", "Bio Fertilizer", "Neem Cake"]
            },

            "tomato": {
                "Best": ["Calcium Nitrate", "Potash", "Bone Meal"],
                "Good": ["Seaweed Fertilizer", "Organic Compost", "Cow Dung"],
                "Average": ["Neem Cake", "Micronutrient Mix", "Bio Fertilizer"]
            },

            "potato": {
                "Best": ["DAP", "Potash", "Gypsum"],
                "Good": ["Organic Compost", "Bone Meal", "Calcium Nitrate"],
                "Average": ["Farmyard Manure", "Bio Fertilizer", "Neem Cake"]
            },

            "banana": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Magnesium Sulphate", "Organic Compost", "Neem Cake"],
                "Average": ["Bio Compost", "Micronutrient Mix", "Vermicompost"]
            },

            "mango": {
                "Best": ["Organic Compost", "DAP", "Potash"],
                "Good": ["Bone Meal", "Neem Cake", "Gypsum"],
                "Average": ["Bio Fertilizer", "Zinc Sulphate", "Vermicompost"]
            },

            "apple": {
                "Best": ["Organic Compost", "DAP", "Potash"],
                "Good": ["Calcium Nitrate", "Gypsum", "Bone Meal"],
                "Average": ["Bio Fertilizer", "Neem Cake", "Vermicompost"]
            },

            "grapes": {
                "Best": ["Potash", "DAP", "Organic Compost"],
                "Good": ["Calcium Nitrate", "Bone Meal", "Seaweed Fertilizer"],
                "Average": ["Neem Cake", "Bio Fertilizer", "Vermicompost"]
            },

            "orange": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Organic Compost", "Micronutrient Mix", "Cow Dung"],
                "Average": ["Gypsum", "Neem Cake", "Bone Meal"]
            },

            "papaya": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Organic Compost", "Vermicompost", "Neem Cake"],
                "Average": ["Bio Compost", "Seaweed Fertilizer", "Bone Meal"]
            },

            "watermelon": {
                "Best": ["DAP", "Potash", "Organic Compost"],
                "Good": ["Gypsum", "Bone Meal", "Neem Cake"],
                "Average": ["Bio Fertilizer", "Vermicompost", "Seaweed Fertilizer"]
            },

            "muskmelon": {
                "Best": ["Urea", "Potash", "DAP"],
                "Good": ["Organic Compost", "Gypsum", "Neem Cake"],
                "Average": ["Seaweed Fertilizer", "Bone Meal", "Cow Dung"]
            },

            "carrot": {
                "Best": ["DAP", "Potash", "Organic Compost"],
                "Good": ["Gypsum", "Bone Meal", "Cow Dung"],
                "Average": ["Neem Cake", "Bio Fertilizer", "Vermicompost"]
            },

            "cabbage": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Organic Compost", "Gypsum", "Bone Meal"],
                "Average": ["Cow Dung", "Neem Cake", "Vermicompost"]
            },

            "cauliflower": {
                "Best": ["DAP", "Potash", "Organic Compost"],
                "Good": ["Gypsum", "Bone Meal", "Cow Dung"],
                "Average": ["Neem Cake", "Bio Fertilizer", "Vermicompost"]
            },

            "peas": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Organic Compost", "Gypsum", "Bone Meal"],
                "Average": ["Cow Dung", "Neem Cake", "Seaweed Fertilizer"]
            },

            "groundnut": {
                "Best": ["Gypsum", "DAP", "Potash"],
                "Good": ["Organic Compost", "Bone Meal", "Cow Dung"],
                "Average": ["Neem Cake", "Bio Fertilizer", "Vermicompost"]
            },

            "soybean": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Organic Compost", "Gypsum", "Bone Meal"],
                "Average": ["Neem Cake", "Cow Dung", "Vermicompost"]
            },

            "sunflower": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Gypsum", "Organic Compost", "Bone Meal"],
                "Average": ["Neem Cake", "Cow Dung", "Seaweed Fertilizer"]
            },

            "mustard": {
                "Best": ["DAP", "Potash", "Organic Compost"],
                "Good": ["Gypsum", "Bone Meal", "Cow Dung"],
                "Average": ["Neem Cake", "Micronutrient Mix", "Vermicompost"]
            },

            "barley": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Gypsum", "Organic Compost", "Bone Meal"],
                "Average": ["Neem Cake", "Cow Dung", "Seaweed Fertilizer"]
            },

            "millets": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Organic Compost", "Gypsum", "Bone Meal"],
                "Average": ["Neem Cake", "Cow Dung", "Vermicompost"]
            },

            "chilli": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Organic Compost", "Calcium Nitrate", "Gypsum"],
                "Average": ["Bone Meal", "Neem Cake", "Seaweed Fertilizer"]
            },

            "brinjal": {
                "Best": ["DAP", "Potash", "Organic Compost"],
                "Good": ["Gypsum", "Bone Meal", "Cow Dung"],
                "Average": ["Neem Cake", "Vermicompost", "Micronutrient Mix"]
            },

            "cucumber": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Organic Compost", "Gypsum", "Bone Meal"],
                "Average": ["Neem Cake", "Cow Dung", "Seaweed Fertilizer"]
            },

            "onion": {
                "Best": ["Urea", "DAP", "Potash"],
                "Good": ["Organic Compost", "Gypsum", "Cow Dung"],
                "Average": ["Neem Cake", "Vermicompost", "Micronutrient Mix"]
            },

            "garlic": {
                "Best": ["DAP", "Potash", "Organic Compost"],
                "Good": ["Gypsum", "Cow Dung", "Neem Cake"],
                "Average": ["Bone Meal", "NPK 19-19-19", "Vermicompost"]
            }

        }

        fertilizer_result = fertilizer_data.get(crop_name, {

            "Best": ["Urea", "DAP", "Potash"],

            "Good": ["Organic Compost", "Neem Cake", "Bio Fertilizer"],

            "Average": ["Bone Meal", "Cow Dung", "Vermicompost"]

        })

    return render_template(
        'fertilizer.html',
        fertilizer_result=fertilizer_result,
        crop_name=crop_name
    )
# =========================
# Disease Detection Module
# =========================
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
# =====================================================
# IMAGE UPLOAD SETTINGS
# =====================================================

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# =====================================================
# ADVANCED DISEASE DATABASE
# (see disease_data.py — imported at the top of this file)
# =====================================================

# =====================================================
# LEAF IMAGE ANALYSIS
# =====================================================

def analyze_leaf(filepath):

    report = {
        "green": 0,
        "yellow": 0,
        "brown": 0,
        "black": 0
    }

    try:

        image = cv2.imread(filepath)

        if image is None:
            return report

        image = cv2.resize(image, (400, 400))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([90, 255, 255])

        # Yellow
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([35, 255, 255])

        # Brown
        brown_lower = np.array([5, 80, 40])
        brown_upper = np.array([20, 255, 255])

        # Black
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])

        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        black_mask = cv2.inRange(hsv, black_lower, black_upper)

        total = image.shape[0] * image.shape[1]

        report["green"] = round(
            cv2.countNonZero(green_mask) * 100 / total,
            2
        )

        report["yellow"] = round(
            cv2.countNonZero(yellow_mask) * 100 / total,
            2
        )

        report["brown"] = round(
            cv2.countNonZero(brown_mask) * 100 / total,
            2
        )

        report["black"] = round(
            cv2.countNonZero(black_mask) * 100 / total,
            2
        )

        return report

    except Exception as e:

        print("Leaf Analysis Error:", e)

        return report
    # =====================================================
# DISEASE RISK CALCULATOR
# =====================================================

def disease_risk(temperature, humidity, rainfall):

    risk_score = 0

    reasons = []

    # ==========================
    # TEMPERATURE
    # ==========================

    if temperature >= 35:

        risk_score += 20

        reasons.append("High temperature increases plant stress.")

    elif temperature <= 15:

        risk_score += 15

        reasons.append("Low temperature favors fungal diseases.")

    else:

        risk_score += 10

    # ==========================
    # HUMIDITY
    # ==========================

    if humidity >= 85:

        risk_score += 35

        reasons.append("Very high humidity promotes fungal infections.")

    elif humidity >= 70:

        risk_score += 20

        reasons.append("Moderate humidity supports disease development.")

    else:

        risk_score += 5

    # ==========================
    # RAINFALL
    # ==========================

    if rainfall >= 150:

        risk_score += 35

        reasons.append("Heavy rainfall spreads bacterial and fungal diseases.")

    elif rainfall >= 80:

        risk_score += 20

        reasons.append("Moderate rainfall increases disease risk.")

    else:

        risk_score += 5

    # ==========================
    # LIMIT SCORE
    # ==========================

    if risk_score > 100:

        risk_score = 100

    # ==========================
    # RISK LEVEL
    # ==========================

    if risk_score >= 85:

        risk_level = "Very High"

    elif risk_score >= 65:

        risk_level = "High"

    elif risk_score >= 40:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    return risk_score, risk_level, reasons
# =====================================================
# DISEASE DIAGNOSS ENGINE
# =====================================================

def diagnose_disease(
    crop,
    leaf_color,
    spot_color,
    spot_size,
    leaf_curl,
    powder,
    stem,
    temperature,
    humidity,
    image_report
):

    best_match = None
    highest_score = 0

    for disease in disease_database:

        # Crop Match
        if disease["crop"] != crop:
            continue

        score = 0
        reasons = []

        # -------------------------
        # Leaf Color
        # -------------------------
        if disease["leaf_color"] == leaf_color:
            score += 20
            reasons.append("Leaf color matched")

        # -------------------------
        # Spot Color
        # -------------------------
        if disease["spot_color"] == spot_color:
            score += 15
            reasons.append("Spot color matched")

        # -------------------------
        # Spot Size
        # -------------------------
        if disease["spot_size"] == spot_size:
            score += 10
            reasons.append("Spot size matched")

        # -------------------------
        # Leaf Curl
        # -------------------------
        if disease["leaf_curl"] == leaf_curl:
            score += 10
            reasons.append("Leaf curl matched")

        # -------------------------
        # Powder
        # -------------------------
        if disease["powder"] == powder:
            score += 10
            reasons.append("Powder symptom matched")

        # -------------------------
        # Stem
        # -------------------------
        if disease["stem"] == stem:
            score += 10
            reasons.append("Stem condition matched")

        # -------------------------
        # Temperature
        # -------------------------
        if disease["temperature"][0] <= temperature <= disease["temperature"][1]:
            score += 15
            reasons.append("Temperature suitable")

        # -------------------------
        # Humidity
        # -------------------------
        if disease["humidity"][0] <= humidity <= disease["humidity"][1]:
            score += 10
            reasons.append("Humidity suitable")

        # -------------------------
        # Image Analysis Bonus
        # -------------------------
        if image_report:

            if leaf_color == "Green" and image_report["green"] > 60:
                score += 3

            if leaf_color == "Yellow" and image_report["yellow"] > 10:
                score += 3

            if leaf_color == "Brown" and image_report["brown"] > 5:
                score += 3

            if leaf_color == "Black" and image_report["black"] > 5:
                score += 3

        # -------------------------
        # Best Match
        # -------------------------
        if score > highest_score:

            highest_score = score

            best_match = disease.copy()

            best_match["confidence"] = min(score, 100)

            best_match["reasons"] = reasons

    return best_match
# =====================================================
# ADVANCED DISEASE ROUTE
# =====================================================

# =====================================================
# ADVANCED DISEASE ROUTE
# =====================================================

def find_disease_entry(disease_name, crop_name=None):
    """Look up rich metadata (medicine, dosage, prevention, etc.) for a
    disease name returned by the CNN, preferring an exact crop match but
    falling back to any crop that shares the same disease name."""

    disease_name_norm = disease_name.strip().lower()

    # Prefer same crop + same disease
    if crop_name:
        for entry in disease_database:
            if (entry["disease"].lower() == disease_name_norm
                    and entry["crop"].lower() == crop_name.strip().lower()):
                return entry.copy()

    # Fall back to any crop with a matching disease name
    for entry in disease_database:
        if entry["disease"].lower() == disease_name_norm:
            return entry.copy()

    # Unknown to our metadata table — return a minimal generic record
    return {
        "crop": crop_name or "Unknown",
        "disease": disease_name,
        "severity": "Unknown",
        "cause": "-",
        "medicine": "-",
        "dosage": "-",
        "spray": "-",
        "fertilizer": "-",
        "recovery": "-",
        "yield_loss": "-",
        "prevention": "Please consult an agricultural expert.",
        "medicine_image": "",
        "disease_image": "",
    }


@app.route("/disease", methods=["GET", "POST"])
def disease():

    result = None
    image_path = None

    if request.method == "POST":

        # ============================
        # GET FORM DATA
        # ============================

        crop = request.form["crop"]
        leaf_color = request.form["leaf_color"]
        spot_color = request.form["spot_color"]
        spot_size = request.form["spot_size"]
        leaf_curl = request.form["leaf_curl"]
        powder = request.form["powder"]
        stem = request.form["stem"]

        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        rainfall = float(request.form["rainfall"])

        # ============================
        # IMAGE UPLOAD
        # ============================

        image_report = None
        cnn_prediction = None

        file = request.files.get("image")

        if file and file.filename != "":

            filename = secure_filename(file.filename)

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            image_path = filepath

            # HSV colour heuristic (kept as a supporting signal / fallback)
            image_report = analyze_leaf(filepath)

            # Primary AI diagnosis: CNN image classifier
            cnn_prediction = predict_disease_cnn(filepath)

        # ============================
        # WEATHER RISK
        # ============================

        risk_score, risk_level, risk_reason = disease_risk(
            temperature,
            humidity,
            rainfall
        )

        # ============================
        # DISEASE DETECTION
        # ============================
        # If an image was uploaded and the CNN model is available, trust
        # the CNN's prediction (it's looking at the actual leaf, not just
        # a symptom questionnaire). Otherwise fall back to the original
        # symptom-based rule matching engine.

        if cnn_prediction:

            entry = find_disease_entry(cnn_prediction["disease"], crop)
            result = entry
            result["confidence"] = cnn_prediction["confidence"]
            result["diagnosis_method"] = "AI Image Analysis (CNN)"
            result["reasons"] = [
                f"🤖 CNN classified the uploaded leaf image as "
                f"'{cnn_prediction['disease']}' with "
                f"{cnn_prediction['confidence']}% confidence."
            ]

        else:

            # Rule-based scoring (kept as a cross-check / metadata source)
            rule_result = diagnose_disease(

                crop,
                leaf_color,
                spot_color,
                spot_size,
                leaf_curl,
                powder,
                stem,
                temperature,
                humidity,
                image_report

            )

            # AI symptom model: predicts disease straight from the
            # questionnaire, no leaf photo required. This is the primary
            # diagnosis whenever it's available.
            ai_predictions = ml_predict_disease_symptom(
                crop, leaf_color, spot_color, spot_size,
                leaf_curl, powder, stem, temperature, humidity,
            )

            if ai_predictions:

                ai_disease, ai_confidence = ai_predictions[0]
                entry = find_disease_entry(ai_disease, crop)
                result = entry
                result["diagnosis_method"] = "AI Symptom Model (RandomForest)"

                reasons = [
                    f"🤖 AI symptom model predicted '{ai_disease}' with "
                    f"{ai_confidence}% confidence based on the "
                    f"questionnaire answers."
                ]

                # Cross-check against the rule engine: if they agree,
                # say so and nudge confidence up slightly; if they
                # disagree, mention the rule engine's alternative too so
                # the farmer sees both signals.
                if rule_result and rule_result["disease"] == ai_disease:
                    ai_confidence = min(100, round(ai_confidence * 0.7 + rule_result["confidence"] * 0.3, 2))
                    reasons.append(
                        "✅ Confirmed by the rule-based symptom checker."
                    )
                elif rule_result:
                    reasons.append(
                        f"ℹ Rule-based checker's closest match was "
                        f"'{rule_result['disease']}' "
                        f"({rule_result['confidence']}% match) — shown here "
                        f"for comparison."
                    )

                if len(ai_predictions) > 1:
                    alt = ", ".join(
                        f"{name} ({conf}%)" for name, conf in ai_predictions[1:]
                    )
                    reasons.append(f"Other possibilities considered: {alt}.")

                result["confidence"] = ai_confidence
                result["reasons"] = reasons

            elif rule_result:

                result = rule_result
                result["diagnosis_method"] = (
                    "Symptom-Based Rule Matching (AI model unavailable)"
                )

            else:

                result = None

        # ============================
        # RESULT FOUND
        # ============================

        if result:

            result["risk_score"] = risk_score
            result["risk_level"] = risk_level
            result["risk_reason"] = risk_reason
            result["image_report"] = image_report

            # Save for PDF
            session["last_result"] = result

            # Save History

            try:

                username = session.get("user", "Guest")

                query = """
                INSERT INTO disease_history
                (
                    username,
                    crop,
                    disease,
                    confidence,
                    severity,
                    medicine,
                    image
                )
                VALUES
                (%s,%s,%s,%s,%s,%s,%s)
                """

                values = (

                    username,
                    result["crop"],
                    result["disease"],
                    result["confidence"],
                    result["severity"],
                    result["medicine"],
                    image_path

                )

                cursor.execute(query, values)
                conn.commit()

            except Exception as e:

                print("Database Error:", e)

        else:

            result = {

                "crop": crop,
                "disease": "No Matching Disease Found",
                "confidence": 0,
                "severity": "Unknown",
                "cause": "-",
                "medicine": "-",
                "dosage": "-",
                "spray": "-",
                "fertilizer": "-",
                "prevention": "Please consult an agricultural expert.",
                "recovery": "-",
                "yield_loss": "-",
                "diagnosis_method": "None",
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_reason": risk_reason,
                "image_report": image_report

            }
    print(result)

    return render_template(
        "disease.html",
        result=result,
        image=image_path,
        ai_active=DISEASE_MODEL_AVAILABLE,
        symptom_ai_active=SYMPTOM_MODEL_AVAILABLE
    )

# =========================
# Contact Page
# =========================
@app.route('/contact', methods=['GET', 'POST'])
def contact():

    if request.method == 'POST':

        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        query = """
        INSERT INTO feedback(name, email, message)
        VALUES(%s,%s,%s)
        """

        values = (name, email, message)

        cursor.execute(query, values)
        conn.commit()

        flash('Message Sent Successfully')

    return render_template('contact.html')

# =========================
# Admin Panel
# =========================
@app.route('/admin')
def admin():

    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()

    cursor.execute("SELECT * FROM feedback")
    feedbacks = cursor.fetchall()

    cursor.execute("SELECT * FROM disease_reports")
    reports = cursor.fetchall()

    return render_template(
        'admin.html',
        users=users,
        feedbacks=feedbacks,
        reports=reports
    )

# =========================
# Logout
# =========================
@app.route('/logout')
def logout():

    session.pop('user', None)
    flash('Logged Out Successfully')

    return redirect('/')

# =========================
# Run Flask App
# =========================
if __name__ == '__main__':

    # Create uploads folder automatically
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')

    app.run(debug=True)