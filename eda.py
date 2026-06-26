import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv("dataset/cleaned/cleaned_data.csv")

print(df.head())
print(df.shape)
print(df.describe())


# Missing Values

plt.figure(figsize=(6,4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values")
plt.savefig("reports/figures/missing_values.png")
plt.close()


# Histograms

numeric_columns = ["Temperature","Humidity","Wind_Speed","PM25","AOD"]

df[numeric_columns].hist(figsize=(12,8), bins=30)
plt.tight_layout()
plt.savefig("reports/figures/histograms.png")
plt.close()


# Correlation Heatmap

plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_columns].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f")

plt.title("Correlation Heatmap")
plt.savefig("reports/figures/correlation_heatmap.png")
plt.close()


# Boxplots

for col in numeric_columns:

    plt.figure(figsize=(6,4))

    sns.boxplot(x=df[col])

    plt.title(col)

    plt.savefig(f"reports/figures/{col}_boxplot.png")

    plt.close()


# PM2.5 Distribution

plt.figure(figsize=(7,5))

sns.histplot(df["PM25"], bins=30, kde=True)

plt.title("PM2.5 Distribution")

plt.savefig("reports/figures/pm25_distribution.png")

plt.close()


# City-wise Average PM2.5

plt.figure(figsize=(12,6))

city_pm = df.groupby("City")["PM25"].mean().sort_values()

city_pm.plot(kind="bar")

plt.ylabel("Average PM2.5")

plt.tight_layout()

plt.savefig("reports/figures/city_pm25.png")

plt.close()

print("EDA Completed Successfully!")