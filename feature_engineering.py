import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned/cleaned_data.csv")

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Extract time-based features
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Day"] = df["Timestamp"].dt.day
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

# Weekend feature
df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

# Encode City
city_encoder = LabelEncoder()
df["City"] = city_encoder.fit_transform(df["City"])

# Save encoder
joblib.dump(city_encoder, "models/city_encoder.pkl")

# Drop Timestamp
df.drop(columns=["Timestamp"], inplace=True)

# Reorder columns
df = df[
    [
        "City",
        "Temperature",
        "Humidity",
        "Wind_Speed",
        "AOD",
        "Year",
        "Month",
        "Day",
        "Hour",
        "DayOfWeek",
        "Is_Weekend",
        "PM25",
    ]
]

# Save processed dataset
df.to_csv("dataset/processed/processed_data.csv", index=False)

print(df.head())
print(df.info())

print("Feature Engineering Completed Successfully!")