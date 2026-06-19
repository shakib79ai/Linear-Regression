import os
import pandas as pd
import openml

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.csv")

print("Downloading Pima Indians Diabetes dataset from OpenML...")
# OpenML dataset ID 37 = diabetes (Pima Indians)
dataset = openml.datasets.get_dataset(37, download_data=True)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

df = X.copy()
df["Outcome"] = (y == "tested_positive").astype(int)

# Rename columns to match the expected names in the script
col_map = {
    "preg":  "Pregnancies",
    "plas":  "Glucose",
    "pres":  "BloodPressure",
    "skin":  "SkinThickness",
    "insu":  "Insulin",
    "mass":  "BMI",
    "pedi":  "DiabetesPedigreeFunction",
    "age":   "Age",
}
df.rename(columns=col_map, inplace=True)

df.to_csv(dest, index=False)
print(f"Done — {len(df)} rows, columns: {list(df.columns)}")
