import os
import traceback
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

from predict import predict_pm25

# ======================================================
# Load Environment Variables
# ======================================================

load_dotenv()

app = Flask(__name__)

# ======================================================
# Gemini Configuration
# ======================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("===================================")
if GEMINI_API_KEY:
    print("✅ Gemini API Key Loaded")
else:
    print("❌ GEMINI_API_KEY NOT FOUND")
print("===================================")

client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini Client Initialized")
    except Exception as e:
        print("❌ Gemini Client Error:", e)

# ======================================================
# Home Page
# ======================================================

@app.route("/")
def index():
    return render_template("index.html")


# ======================================================
# Result Page
# ======================================================

@app.route("/result")
def result():
    return render_template("result.html")


# ======================================================
# Prediction API
# ======================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        city = data["city"]
        aod = float(data["aod"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        wind_speed = float(data["wind_speed"])
        date = data.get("date")

        result = predict_pm25(
            city=city,
            aod=aod,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            date=date
        )

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:

        print("\n========== PREDICTION ERROR ==========")
        traceback.print_exc()
        print("======================================\n")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ======================================================

# Gemini AI Health Advice

# ======================================================

@app.route("/health-advice", methods=["POST"])

def health_advice():

    try:

        if client is None:

            return jsonify({

                "success": False,

                "advice": "Gemini API Key not configured."

            })

        data = request.get_json()

        city = data.get("city", "Unknown")

        pm25 = float(data.get("pm25", 0))

        category = data.get("category", "Unknown")

        aod = float(data.get("aod", 0))

        temperature = float(data.get("temperature", 0))

        humidity = float(data.get("humidity", 0))

        wind_speed = float(data.get("wind_speed", 0))

        prompt = f"""

You are AeroSense AI, an environmental scientist and air quality analyst.

Analyze today's predicted air quality.

City: {city}

Predicted PM2.5: {pm25} μg/m³

Air Quality Category: {category}

Weather Conditions:

- Temperature: {temperature} °C

- Humidity: {humidity} %

- Wind Speed: {wind_speed} m/s

- Aerosol Optical Depth (AOD): {aod}

Instructions:

1. Explain today's air quality in simple language.

2. Explain WHY the PM2.5 is likely at this level using AOD, wind speed, humidity and temperature.

3. Mention pollution sources that are COMMON in {city}. Make them city-specific.

4. Give health recommendations based on the pollution level.

5. Recommend outdoor activities according to the PM2.5 value.

6. Mention whether wearing a mask is necessary.

7. Give one practical environmental suggestion for residents of {city}.

IMPORTANT RULES:

- The advice MUST change depending on the PM2.5 value.

- If PM2.5 is below 12:

  Mention that the air is clean and outdoor activities are safe.

- If PM2.5 is between 12 and 35:

  Mention moderate pollution and advise only sensitive groups to be cautious.

- If PM2.5 is between 35 and 55:

  Mention that children, elderly people and asthma patients should reduce prolonged outdoor exposure.

- If PM2.5 is between 55 and 150:

  Mention that everyone should reduce outdoor exposure, recommend an N95 mask and avoid strenuous exercise.

- If PM2.5 is above 150:

  Mention serious health risks, recommend staying indoors and avoiding unnecessary travel.

Do NOT use the same wording for every response.

Avoid generic answers.

Write naturally in one concise report (120–170 words).

Do NOT use markdown headings or bullet points.

"""

        response = client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt

        )

        advice = response.text if response.text else "No response generated."

        return jsonify({

            "success": True,

            "advice": advice

        })

    except Exception as e:

        print("\n========== GEMINI ERROR ==========")

        traceback.print_exc()

        print("==================================\n")

        return jsonify({

            "success": False,

            "advice": f"Gemini Error: {str(e)}"

        })

# ======================================================
# Run App
# ======================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )