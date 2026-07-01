import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from predict import predict_pm25

# ── Setup ─────────────────────────────────────────────────────
load_dotenv()
app = Flask(__name__)

# Configure Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Route 1: Homepage ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Route 2: Result Page ──────────────────────────────────────
@app.route("/result")
def result():
    return render_template("result.html")

# ── Route 3: Predict ─────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        city        = data.get("city", "Delhi")
        aod         = float(data.get("aod", 0.5))
        temperature = float(data.get("temperature", 30))
        humidity    = float(data.get("humidity", 60))
        wind_speed  = float(data.get("wind_speed", 3.0))
        date        = data.get("date", None)

        result = predict_pm25(
            city=city,
            aod=aod,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            date=date
        )

        return jsonify({"success": True, "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ── Route 4: AI Health Advice ─────────────────────────────────
@app.route("/health-advice", methods=["POST"])
def health_advice():
    try:
        data     = request.get_json()
        pm25     = data.get("pm25")
        city     = data.get("city", "the city")
        category = data.get("category", "")

        prompt = f"""
You are an air quality health expert.
PM2.5 level in {city} is {pm25} ug/m3 (Category: {category}).

Give a helpful health advisory with these sections:
1. What this pollution level means
2. Who is most at risk
3. Health precautions to take today
4. Outdoor activity recommendations
5. One simple tip to reduce personal exposure

Keep it concise, friendly, and practical.
"""

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        advice = response.text

        return jsonify({"success": True, "advice": advice})

    except Exception as e:
        return jsonify({"success": False,
                        "error": f"AI advice unavailable: {str(e)}"}), 500

# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)