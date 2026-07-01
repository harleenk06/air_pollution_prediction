import numpy as np
import pandas as pd
import pickle
import joblib
import json
from datetime import datetime

# ==========================
# Load model and artifacts
# ==========================

with open("models/chronological/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Use the scaler that belongs to this model
scaler = joblib.load("models/chronological/scaler.pkl")

city_encoder = joblib.load("models/city_encoder.pkl")

with open("models/chronological/model_info.json") as f:
    model_info = json.load(f)

FEATURES = model_info["features"]

print(f"Model loaded: {model_info['best_model_name']} (Test R2 = {model_info['r2_score']})")
print(f"Expected features: {FEATURES}\n")


# ==========================
# AQI Category
# ==========================

def get_aqi_category(pm25):

    if pm25 <= 12:
        return {
            "category": "Good",
            "color": "green",
            "message": "Air quality is satisfactory."
        }

    elif pm25 <= 35.4:
        return {
            "category": "Moderate",
            "color": "yellow",
            "message": "Acceptable. Sensitive people should limit outdoor exertion."
        }

    elif pm25 <= 55.4:
        return {
            "category": "Unhealthy for Sensitive Groups",
            "color": "orange",
            "message": "Sensitive groups may experience health effects."
        }

    elif pm25 <= 150.4:
        return {
            "category": "Unhealthy",
            "color": "red",
            "message": "Everyone may begin to experience health effects."
        }

    elif pm25 <= 250.4:
        return {
            "category": "Very Unhealthy",
            "color": "purple",
            "message": "Health alert: everyone may experience serious effects."
        }

    else:
        return {
            "category": "Hazardous",
            "color": "maroon",
            "message": "Health emergency — everyone is affected."
        }


# ==========================
# Prediction Function
# ==========================

def predict_pm25(city,
                 aod,
                 temperature,
                 humidity,
                 wind_speed,
                 date=None):

    # Encode city
    try:
        city_encoded = int(city_encoder.transform([city])[0])

    except ValueError:

        raise ValueError(
            f"City '{city}' not found.\n"
            f"Available cities: {list(city_encoder.classes_)}"
        )

    # Date Features
    dt = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()

    year = int(dt.year)
    month = int(dt.month)
    day = int(dt.day)
    hour = int(dt.hour)
    day_of_week = int(dt.weekday())
    is_weekend = int(day_of_week >= 5)

    # Feature Dictionary
    raw = {
        "City": city_encoded,
        "Temperature": float(temperature),
        "Humidity": float(humidity),
        "Wind_Speed": float(wind_speed),
        "AOD": float(aod),
        "Year": year,
        "Month": month,
        "Day": day,
        "Hour": hour,
        "DayOfWeek": day_of_week,
        "Is_Weekend": is_weekend
    }

    # Arrange Features
    feature_vector = pd.DataFrame(
        [[raw[col] for col in FEATURES]],
        columns=FEATURES
    )

    # Scale
    feature_scaled = scaler.transform(feature_vector)

    # Predict
    pm25 = float(model.predict(feature_scaled)[0])

    pm25 = round(max(pm25, 0), 2)

    aqi = get_aqi_category(pm25)

    return {

        "city": str(city),

        "pm25": float(pm25),

        "category": str(aqi["category"]),

        "color": str(aqi["color"]),

        "message": str(aqi["message"]),

        "inputs": {

            "City": int(city_encoded),

            "Temperature": float(temperature),

            "Humidity": float(humidity),

            "Wind_Speed": float(wind_speed),

            "AOD": float(aod),

            "Year": int(year),

            "Month": int(month),

            "Day": int(day),

            "Hour": int(hour),

            "DayOfWeek": int(day_of_week),

            "Is_Weekend": int(is_weekend)
        }
    }


# ==========================
# Test
# ==========================

if __name__ == "__main__":

    result = predict_pm25(

        city="Delhi",

        aod=0.82,

        temperature=34,

        humidity=68,

        wind_speed=2.0,

        date="2026-06-30"

    )

    print(result)