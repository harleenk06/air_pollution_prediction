import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset/processed/processed_data.csv")

X = df.drop("PM25", axis=1)
y = df["PM25"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
    "dataset/processed/X_train.csv",
    index=False
)

pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
    "dataset/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "dataset/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "dataset/processed/y_test.csv",
    index=False
)

print(X_train.shape)
print(X_test.shape)

print("Preprocessing Completed Successfully!")