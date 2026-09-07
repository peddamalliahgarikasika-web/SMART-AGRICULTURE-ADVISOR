# Smart Agriculture Advisor 🌱

Smart Agriculture Advisor is a modern farming web application developed using:

- Python Flask
- MySQL
- HTML
- CSS
- JavaScript
- Bootstrap

This project helps farmers with smart agriculture solutions.

---

# Features

✅ Farmer Registration & Login  
✅ Crop Recommendation  
✅ Weather Information  
✅ Fertilizer Suggestion  
✅ Disease Detection Upload  
✅ Agriculture Tips  
✅ Contact Admin  
✅ Admin Panel  
✅ MySQL Database Integration  
✅ Responsive Design  

---

# Technologies Used

Frontend:
- HTML5
- CSS3
- JavaScript
- Bootstrap

Backend:
- Python Flask

Database:
- MySQL

---

# Project Structure

smart_agriculture_advisor/

├── app.py  
├── requirements.txt  
├── database.sql  
├── README.md  

├── templates/  
│   ├── base.html  
│   ├── index.html  
│   ├── about.html  
│   ├── register.html  
│   ├── login.html  
│   ├── dashboard.html  
│   ├── crop.html  
│   ├── weather.html  
│   ├── fertilizer.html  
│   ├── disease.html  
│   ├── contact.html  
│   ├── admin.html  
│   └── tips.html  

├── static/  
│   ├── css/  
│   │   └── style.css  
│   ├── js/  
│   │   └── script.js  
│   ├── images/  
│   └── uploads/  

---

# Installation Steps

## Step 1
Install Python

## Step 2
Install MySQL

## Step 3
Install VS Code

## Step 4
Install Required Packages

Run:

pip install -r requirements.txt

---

# Database Setup

1. Open MySQL Workbench
2. Open database.sql
3. Execute all SQL queries

---

# Run Project

Open terminal and run:

python app.py

---

# Open Browser

Visit:

http://127.0.0.1:5000

---

# Default Database Details

Host: localhost  
User: root  
Password: yourpassword  
Database: smart_agriculture  

---

# AI / Machine Learning

Crop Recommendation is powered by a trained RandomForest model (60 crops), blended with the original agronomic rule scoring.

Disease Detection now has **two** AI layers:
1. **CNN image classifier** (MobileNetV2) — used automatically when a leaf photo is uploaded *and* you've trained it (needs a real leaf-image dataset — see `ml/train_disease_model.py`). Not trained by default since that requires downloading thousands of photos.
2. **AI Symptom Model (RandomForest)** — trained instantly from the existing disease knowledge base (`disease_data.py`), no photo or external dataset required. It predicts the disease straight from the questionnaire (crop, leaf colour, spot colour/size, leaf curl, powder, stem condition, temperature, humidity) and is cross-checked against the original rule engine. This one ships already trained (`ml/models/disease_symptom_model.pkl`) so disease prediction is "real AI" out of the box, even without any photos.

To retrain the symptom model after editing `disease_data.py`:
```
cd ml
python generate_disease_dataset.py
python train_disease_symptom_model.py
```

See ml/README_ML.md for full details and retraining instructions.

---

# Light / Dark Theme

Use the 🌙/☀️ toggle button in the navbar (next to the language switcher) to flip between a light theme and a dark theme. The choice is saved in the browser's `localStorage` and applied instantly on every page (no flash of the wrong theme), and it also respects your OS's dark-mode preference the very first time you visit.

---

# AI Chatbot (Flask + Gemini API)

A floating "Krishi Mitra" chat assistant (💬 button, bottom-right) is available on every page, backed by Google's Gemini API.

## Setup

1. Get a free Gemini API key: https://aistudio.google.com/app/apikey
2. Set it as an environment variable before starting the app:

   Windows (PowerShell):
   ```
   $env:GEMINI_API_KEY="your_key_here"
   ```

   macOS / Linux:
   ```
   export GEMINI_API_KEY="your_key_here"
   ```

3. (Optional) Choose a different model, default is `gemini-2.0-flash`:
   ```
   export GEMINI_MODEL="gemini-2.0-flash"
   ```

4. Run the app as usual (`python main.py`). If `GEMINI_API_KEY` is not set, the chat widget still works but replies with a friendly "service unavailable" message instead of crashing the app.

The backend route is `POST /api/chatbot` (JSON: `{"message": "...", "lang": "hi"}`), implemented in `main.py`. It keeps a short rolling conversation history per browser session and instructs Gemini to reply in whichever language the user currently has selected on the site.

---

# Multi-Language Support

The site UI (navigation, footer, chatbot, disease detection page) can be switched between:

- English (en)
- Hindi (hi)
- Telugu (te)
- Tamil (ta)
- Kannada (kn)
- Malayalam (ml)
- Marathi (mr)
- Gujarati (gu)
- Bengali (bn)
- Punjabi (pa)
- Odia (or)

Use the 🌐 language dropdown in the navbar to switch — the choice is stored in the session and also tells the Gemini chatbot which language to reply in.

**English, Hindi and Telugu have full coverage** — every label, dropdown option (states, soil types, seasons, crops, symptoms), and results table on the Crop Recommendation and Disease Detection pages is translated, not just the navbar. The other 8 languages currently cover navigation/chat only; any key they're missing automatically falls back to English rather than showing a blank label.

Translations live in `translations.py`. To add more translated strings or another language, see the instructions in that file's docstring — any key missing for a language automatically falls back to English so the UI never breaks.

---

# Future Improvements

- SMS Alerts
- Mobile App
- Voice input for the chatbot

---

# Developed For

College Final Year Project

---

# Author

Smart Agriculture Advisor Project