import pandas as pd

# Load dataset
df = pd.read_csv("dataset/raw/prediction_data.csv")

# Basic information
print(df.head())
print(df.shape)
print(df.columns)
df.info()
print(df.describe())

# Rename columns
df.rename(columns={
    "Time": "Timestamp",
    "Temp": "Temperature",
    "Wind Speed": "Wind_Speed",
    "PM2.5": "PM25"
}, inplace=True)

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Standardize city names
df["City"] = df["City"].str.strip().str.title()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove invalid values
df = df[df["Temperature"] > -30]
df = df[df["Humidity"].between(0, 100)]
df = df[df["Wind_Speed"] >= 0]
df = df[df["PM25"] >= 0]
df = df[df["AOD"] >= 0]

# Sort dataset by timestamp
df.sort_values("Timestamp", inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Final information
print("\nFinal Shape:", df.shape)
print(df.info())

# Save cleaned dataset
df.to_csv("dataset/cleaned/cleaned_data.csv", index=False)

print("\nCleaning Completed Successfully!")