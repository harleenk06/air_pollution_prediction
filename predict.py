import numpy as np
import pandas as pd
import pickle
import joblib
import json
from datetime import datetime

# Loading model, scaler, encoder, and feature list
with open("models/chronological/best_model.pkl", "rb") as f:
    model = pickle.load(f)

scaler = joblib.load("models/scaler.pkl")
city_encoder = joblib.load("models/city_encoder.pkl")

with open("models/chronological/model_info.json") as f:
    model_info = json.load(f)

FEATURES = model_info["features"]
print(f"Model loaded: {model_info['best_model_name']} (Test R2 = {model_info['r2_score']})")
print(f"Expected features: {FEATURES}\n")

# AQI category helper function
def get_aqi_category(pm25: float) -> dict:
    if pm25 <= 12:
        return {"category": "Good", "color": "green",
                "message": "Air quality is satisfactory."}
    elif pm25 <= 35.4:
        return {"category": "Moderate", "color": "yellow",
                "message": "Acceptable. Sensitive people limit outdoor exertion."}
    elif pm25 <= 55.4:
        return {"category": "Unhealthy for Sensitive Groups", "color": "orange",
                "message": "Sensitive groups may experience health effects."}
    elif pm25 <= 150.4:
        return {"category": "Unhealthy", "color": "red",
                "message": "Everyone may begin to experience health effects."}
    elif pm25 <= 250.4:
        return {"category": "Very Unhealthy", "color": "purple",
                "message": "Health alert: everyone may experience serious effects."}
    else:
        return {"category": "Hazardous", "color": "maroon",
                "message": "Health emergency — everyone is affected."}

# Core prediction function 
def predict_pm25(city, aod, temperature, humidity, wind_speed, date: str = None):
    """
    city        : str, e.g. "Delhi"
    aod         : float, Aerosol Optical Depth
    temperature : float, °C
    humidity    : float, %
    wind_speed  : float, m/s
    date        : optional "YYYY-MM-DD" string. Defaults to today if not given.
    """
    # Encode city 
    try:
        city_encoded = city_encoder.transform([city])[0]
    except ValueError:
        raise ValueError(
            f"City '{city}' was not seen during training. "
            f"Known cities: {list(city_encoder.classes_)}"
        )

    # Derive date-based features 
    dt = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
    year, month, day, hour = dt.year, dt.month, dt.day, dt.hour
    day_of_week = dt.weekday()          # Monday=0 ... Sunday=6
    is_weekend = 1 if day_of_week >= 5 else 0

    # Build the raw feature dict 
    raw = {
        "City": city_encoded,
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind_Speed": wind_speed,
        "AOD": aod,
        "Year": year,
        "Month": month,
        "Day": day,
        "Hour": hour,
        "DayOfWeek": day_of_week,
        "Is_Weekend": is_weekend
    }

    # Assembling in the exact order the model expects 
    feature_vector = pd.DataFrame([[raw[f] for f in FEATURES]], columns=FEATURES)

    # Scale and predict
    feature_scaled = scaler.transform(feature_vector)
    pm25 = float(model.predict(feature_scaled)[0])
    pm25 = round(max(pm25, 0), 2)  # PM2.5 can't be negative

    aqi = get_aqi_category(pm25)

    return {
        "city": city,
        "pm25": pm25,
        "category": aqi["category"],
        "color": aqi["color"],
        "message": aqi["message"],
        "inputs": raw
    }

# Testing it directly 
if __name__ == "__main__":
    result = predict_pm25(
        city="Delhi",
        aod=0.82,
        temperature=34,
        humidity=68,
        wind_speed=2.0,
        date="2026-06-30"
    )
    print(f"City     : {result['city']}")
    print(f"PM2.5    : {result['pm25']} ug/m3")
    print(f"Category : {result['category']}")
    print(f"Message  : {result['message']}")