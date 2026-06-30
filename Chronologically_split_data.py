import pandas as pd

#  1. Load the processed data 
df = pd.read_csv('dataset/processed/processed_data.csv')

print(f"Total rows: {len(df)}")
print(f"Columns   : {list(df.columns)}\n")

# 2. Sort chronologically 
df = df.sort_values(by=['Year', 'Month', 'Day', 'Hour']).reset_index(drop=True)

print(f"Date range: {df[['Year','Month','Day']].iloc[0].to_dict()}  -->  {df[['Year','Month','Day']].iloc[-1].to_dict()}\n")

#  chronological split 
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

print(f"Train rows : {len(train_df)}  ({train_df['Year'].min()} to {train_df['Year'].max()})")
print(f"Test rows  : {len(test_df)}   ({test_df['Year'].min()} to {test_df['Year'].max()})\n")

#  Separate features and target 
TARGET = 'PM25'

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

#  5. Save to a new folder — originals stay untouched 
import os
os.makedirs('dataset/processed/chronological', exist_ok=True)

X_train.to_csv('dataset/processed/chronological/X_train.csv', index=False)
y_train.to_csv('dataset/processed/chronological/y_train.csv', index=False)
X_test.to_csv('dataset/processed/chronological/X_test.csv',  index=False)
y_test.to_csv('dataset/processed/chronological/y_test.csv',  index=False)

print("Saved to dataset/processed/chronological/")
print("  X_train.csv")
print("  y_train.csv")
print("  X_test.csv")
print("  y_test.csv")
