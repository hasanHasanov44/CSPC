import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# 1. Fetch the Concrete dataset with ID 165 from the UCI repository
concrete = fetch_ucirepo(id=165)

# 2. Extract features (X) and target variable (y)
X = concrete.data.features.copy()
y = concrete.data.targets.iloc[:, 0].rename("strength")

# 3. Rename columns with concise and clean names
X.columns = [
    "cement", "slag", "ash", "water", 
    "superplastic", "coarse_agg", "fine_agg", "age"
]

# 4. Print metadata and variables table
print("--- METADATA ---")
print(concrete.metadata)

print("\n--- VARIABLES TABLE ---")
print(concrete.variables)

# 5. Create data/ directory and save as concrete.csv
os.makedirs("data", exist_ok=True)
X.assign(strength=y).to_csv("data/concrete.csv", index=False)

print("\nData successfully fetched and saved to 'data/concrete.csv'!")