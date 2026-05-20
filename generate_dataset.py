"""
generate_dataset.py
-------------------
Generates a synthetic raw retail sales dataset with intentional:
  - Missing values
  - Duplicate rows
  - Outliers
Saves to raw_sales_data.csv
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 500

categories  = ["Electronics", "Clothing", "Food & Beverages", "Home & Kitchen", "Sports"]
products    = {
    "Electronics":       ["Laptop", "Smartphone", "Headphones", "Tablet", "Smartwatch"],
    "Clothing":          ["T-Shirt", "Jeans", "Jacket", "Dress", "Shoes"],
    "Food & Beverages":  ["Coffee", "Tea", "Juice", "Snacks", "Protein Bar"],
    "Home & Kitchen":    ["Blender", "Toaster", "Vacuum", "Cookware Set", "Air Fryer"],
    "Sports":            ["Yoga Mat", "Dumbbells", "Running Shoes", "Bicycle", "Tennis Racket"],
}
regions = ["North", "South", "East", "West", "Central"]

cat_list  = np.random.choice(categories, N)
prod_list = [np.random.choice(products[c]) for c in cat_list]

base_price = {
    "Electronics": 500, "Clothing": 50, "Food & Beverages": 15,
    "Home & Kitchen": 120, "Sports": 80,
}

price    = np.array([base_price[c] * np.random.uniform(0.6, 1.8) for c in cat_list]).round(2)
quantity = np.random.randint(1, 20, N)
discount = np.random.choice([0, 0.05, 0.10, 0.15, 0.20], N)
profit_margin = np.random.uniform(0.05, 0.35, N)

sales  = (price * quantity * (1 - discount)).round(2)
profit = (sales * profit_margin).round(2)

dates = pd.date_range("2023-01-01", periods=N, freq="D")
np.random.shuffle(dates.values)

df = pd.DataFrame({
    "Date":     [d.strftime("%Y-%m-%d") for d in dates],
    "Region":   np.random.choice(regions, N),
    "Category": cat_list,
    "Product":  prod_list,
    "Quantity": quantity,
    "Price":    price,
    "Discount": discount,
    "Sales":    sales,
    "Profit":   profit,
    "CustomerAge": np.random.randint(18, 70, N),
    "Rating":   np.random.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.10, 0.20, 0.40, 0.25]),
})

# ── Inject Missing Values (~8%) ──────────────────────────────────────────────
for col in ["Sales", "Profit", "CustomerAge", "Rating", "Discount"]:
    mask = np.random.choice([True, False], N, p=[0.08, 0.92])
    df.loc[mask, col] = np.nan

# ── Inject Outliers ──────────────────────────────────────────────────────────
outlier_idx = np.random.choice(N, 15, replace=False)
df.loc[outlier_idx[:5],  "Sales"]  = np.random.choice([50000, 75000, 99000], 5)
df.loc[outlier_idx[5:10], "Profit"] = np.random.choice([-5000, 30000, 45000], 5)
df.loc[outlier_idx[10:],  "CustomerAge"] = np.random.choice([120, 150, 200], 5)

# ── Inject Duplicates (30 rows) ──────────────────────────────────────────────
dup_rows = df.sample(30, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)
df = df.sample(frac=1, random_state=99).reset_index(drop=True)

df.to_csv("raw_sales_data.csv", index=False)
print(f"[SUCCESS] raw_sales_data.csv created - {len(df)} rows x {len(df.columns)} columns")
print(f"    Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"    Duplicate rows : {df.duplicated().sum()}")
