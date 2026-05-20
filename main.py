"""
main.py - Data Cleaning & Visualization Project
==================================================
Steps:
  1. Load raw data
  2. Explore / profile the dataset
  3. Handle missing values
  4. Remove duplicates
  5. Detect & treat outliers (IQR method)
  6. Feature engineering
  7. Visualizations (individual plots + a combined dashboard)

Run:
  python generate_dataset.py   # create raw_sales_data.csv first
  python main.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# --- Output folder -------------------------------------------------------------
os.makedirs("output", exist_ok=True)

# --- Seaborn / Matplotlib theme ------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font="DejaVu Sans")
PALETTE   = sns.color_palette("coolwarm", 5)
PRIMARY   = "#2563eb"
ACCENT    = "#16a34a"
WARN      = "#dc2626"

# ===============================================================================
# 1 : LOAD DATA
# ===============================================================================
print("=" * 60)
print("  DATA CLEANING & VISUALIZATION PROJECT")
print("=" * 60)

CSV_PATH = "raw_sales_data.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"'{CSV_PATH}' not found.\nRun  python generate_dataset.py  first."
    )

df_raw = pd.read_csv(CSV_PATH, parse_dates=["Date"])
df     = df_raw.copy()

print(f"\n[1] Dataset loaded : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"    Columns        : {list(df.columns)}\n")

# ===============================================================================
# 2 : EXPLORE / PROFILE
# ===============================================================================
print("[2] Basic Info")
print(df.dtypes.to_string())

print("\n[2] Descriptive Statistics")
print(df.describe(include="all").T.to_string())

missing_before = df.isnull().sum()
dup_before     = df.duplicated().sum()

print(f"\n[2] Missing values per column:\n{missing_before[missing_before > 0].to_string()}")
print(f"\n[2] Duplicate rows : {dup_before}")

# ===============================================================================
# 3 : HANDLE MISSING VALUES
# ===============================================================================
print("\n[3] Handling missing values...")

# Numeric cols -> fill with median (robust to outliers)
num_cols = ["Sales", "Profit", "Discount", "CustomerAge"]
for col in num_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"    {col:15s} -> filled NaN with median ({median_val:.2f})")

# Rating -> fill with mode
mode_rating = df["Rating"].mode()[0]
df["Rating"].fillna(mode_rating, inplace=True)
print(f"    {'Rating':15s} -> filled NaN with mode  ({mode_rating})")

missing_after = df.isnull().sum().sum()
print(f"    Total missing values after cleaning : {missing_after}")

# ===============================================================================
# 4 : REMOVE DUPLICATES
# ===============================================================================
print(f"\n[4] Removing duplicates...")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
dup_after = df.duplicated().sum()
print(f"    Rows before : {df_raw.shape[0]}  |  Rows after : {df.shape[0]}")
print(f"    Duplicate rows remaining : {dup_after}")

# ===============================================================================
# 5 : OUTLIER DETECTION & TREATMENT (IQR method)
# ===============================================================================
print("\n[5] Outlier detection (IQR method)...")

def remove_outliers_iqr(data: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    Q1  = data[column].quantile(0.25)
    Q3  = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask    = data[column].between(lower, upper)
    removed = (~mask).sum()
    return data[mask].reset_index(drop=True), removed

outlier_cols  = ["Sales", "Profit", "CustomerAge"]
outlier_stats = {}

for col in outlier_cols:
    before = len(df)
    df, removed = remove_outliers_iqr(df, col)
    outlier_stats[col] = removed
    print(f"    {col:15s} -> {removed} outlier rows removed")

print(f"    Rows after outlier removal : {len(df)}")

# ===============================================================================
# 6 : FEATURE ENGINEERING
# ===============================================================================
print("\n[6] Feature engineering...")

df["Month"]         = df["Date"].dt.month_name()
df["Quarter"]       = "Q" + df["Date"].dt.quarter.astype(str)
df["ProfitMargin"]  = (df["Profit"] / df["Sales"] * 100).round(2)
df["AgeGroup"]      = pd.cut(
    df["CustomerAge"],
    bins  = [17, 25, 35, 50, 70],
    labels= ["18-25", "26-35", "36-50", "51-70"]
)

print(f"    New columns: Month, Quarter, ProfitMargin, AgeGroup")
print(f"    Final dataset : {df.shape[0]} rows x {df.shape[1]} columns")

# Save cleaned data
df.to_csv("output/cleaned_sales_data.csv", index=False)
print("    Cleaned data saved -> output/cleaned_sales_data.csv")

# ===============================================================================
# 7 : INDIVIDUAL VISUALIZATIONS
# ===============================================================================
print("\n[7] Generating visualizations...")

# --- Plot helpers --------------------------------------------------------------
def save(name):
    path = f"output/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [SAVED] {path}")


# --- 7.1  Missing-value heatmap (before vs after) ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Missing Value Analysis", fontsize=15, fontweight="bold")

miss_pct_before = (df_raw.isnull().sum() / len(df_raw) * 100).sort_values(ascending=False)
miss_pct_after  = (df.isnull().sum()    / len(df)     * 100).sort_values(ascending=False)

colors_b = [WARN if v > 0 else ACCENT for v in miss_pct_before]
colors_a = [WARN if v > 0 else ACCENT for v in miss_pct_after]

axes[0].barh(miss_pct_before.index, miss_pct_before.values, color=colors_b)
axes[0].set_title("Before Cleaning", fontweight="bold")
axes[0].set_xlabel("Missing %")
axes[0].axvline(0, color="black", linewidth=0.5)

axes[1].barh(miss_pct_after.index, miss_pct_after.values, color=colors_a)
axes[1].set_title("After Cleaning", fontweight="bold")
axes[1].set_xlabel("Missing %")

plt.tight_layout()
save("01_missing_values")


# --- 7.2  Sales distribution before & after outlier removal -------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Sales Distribution - Outlier Treatment", fontsize=15, fontweight="bold")

for ax, data, title, color in zip(
    axes,
    [df_raw["Sales"].dropna(), df["Sales"]],
    ["Before Outlier Removal", "After Outlier Removal"],
    [WARN, ACCENT],
):
    ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(data.mean(),   color="navy",  linestyle="--", linewidth=1.5, label=f"Mean  {data.mean():.0f}")
    ax.axvline(data.median(), color="black", linestyle=":",  linewidth=1.5, label=f"Median {data.median():.0f}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Sales (INR)")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()
save("02_sales_distribution")


# --- 7.3  Sales by Category (boxplot) -----------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x="Category", y="Sales", palette="coolwarm", ax=ax)
ax.set_title("Sales Distribution by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Sales (INR)")
plt.xticks(rotation=15)
plt.tight_layout()
save("03_sales_by_category_boxplot")


# --- 7.4  Monthly Sales Trend --------------------------------------------------
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
monthly = (
    df.groupby("Month")["Sales"]
    .sum()
    .reindex([m for m in month_order if m in df["Month"].unique()])
)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(monthly.index, monthly.values, alpha=0.2, color=PRIMARY)
ax.plot(monthly.index, monthly.values, marker="o", color=PRIMARY, linewidth=2.5)
ax.set_title("Monthly Sales Trend", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Total Sales (INR)")
plt.xticks(rotation=30)
plt.tight_layout()
save("04_monthly_sales_trend")


# --- 7.5  Top 10 Products by Sales --------------------------------------------
top_products = df.groupby("Product")["Sales"].sum().nlargest(10).sort_values()

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_products.index, top_products.values, color=sns.color_palette("Blues_r", 10))
ax.set_title("Top 10 Products by Total Sales", fontsize=14, fontweight="bold")
ax.set_xlabel("Total Sales (INR)")
for bar in bars:
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
            f"INR {bar.get_width():,.0f}", va="center", fontsize=8)
plt.tight_layout()
save("05_top10_products")


# --- 7.6  Sales by Region (pie chart) -----------------------------------------
region_sales = df.groupby("Region")["Sales"].sum()

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    region_sales,
    labels    = region_sales.index,
    autopct   = "%1.1f%%",
    startangle= 140,
    colors    = sns.color_palette("pastel"),
    wedgeprops= {"edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_fontweight("bold")
ax.set_title("Sales Distribution by Region", fontsize=14, fontweight="bold")
plt.tight_layout()
save("06_sales_by_region_pie")


# --- 7.7  Correlation Heatmap --------------------------------------------------
num_df = df[["Sales", "Profit", "Quantity", "Price", "Discount", "CustomerAge", "Rating", "ProfitMargin"]]
corr   = num_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", linewidths=0.5,
    vmin=-1, vmax=1, ax=ax,
    annot_kws={"size": 9}
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
save("07_correlation_heatmap")


# --- 7.8  Profit Margin by Category (violin) -----------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(data=df, x="Category", y="ProfitMargin", palette="muted", inner="box", ax=ax)
ax.set_title("Profit Margin Distribution by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Profit Margin (%)")
plt.xticks(rotation=15)
plt.tight_layout()
save("08_profit_margin_violin")


# --- 7.9  Customer Age Group vs Rating (heatmap) -------------------------------
pivot = df.pivot_table(index="AgeGroup", columns="Rating", values="Sales", aggfunc="mean")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=ax)
ax.set_title("Avg Sales by Customer Age Group & Rating", fontsize=14, fontweight="bold")
ax.set_xlabel("Rating (Stars)")
ax.set_ylabel("Age Group")
plt.tight_layout()
save("09_agegroup_rating_heatmap")


# --- 7.10  Sales vs Profit scatter (coloured by category) ---------------------
fig, ax = plt.subplots(figsize=(10, 7))
for cat, color in zip(df["Category"].unique(), sns.color_palette("tab10")):
    sub = df[df["Category"] == cat]
    ax.scatter(sub["Sales"], sub["Profit"], label=cat, color=color, alpha=0.55, s=30)

ax.set_title("Sales vs Profit by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Sales (INR)")
ax.set_ylabel("Profit (INR)")
ax.legend(title="Category", fontsize=9)
plt.tight_layout()
save("10_sales_vs_profit_scatter")


# ===============================================================================
# 8 : COMBINED DASHBOARD
# ===============================================================================
print("\n[8] Generating combined dashboard...")

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor("#f8fafc")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

fig.suptitle(
    "Data Cleaning & Visualization Dashboard - Retail Sales 2023",
    fontsize=18, fontweight="bold", y=0.98, color="#111827"
)

# Panel A - Monthly Sales Trend
ax_a = fig.add_subplot(gs[0, :2])
ax_a.fill_between(monthly.index, monthly.values, alpha=0.15, color=PRIMARY)
ax_a.plot(monthly.index, monthly.values, marker="o", color=PRIMARY, linewidth=2)
ax_a.set_title("A  Monthly Sales Trend", fontweight="bold")
ax_a.set_ylabel("Total Sales (INR)")
ax_a.tick_params(axis="x", rotation=30)
ax_a.grid(True, linestyle="--", alpha=0.5)

# Panel B - Region Pie
ax_b = fig.add_subplot(gs[0, 2])
ax_b.pie(
    region_sales, labels=region_sales.index, autopct="%1.1f%%",
    startangle=140, colors=sns.color_palette("pastel"),
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    textprops={"fontsize": 8},
)
ax_b.set_title("B  Sales by Region", fontweight="bold")

# Panel C - Sales by Category boxplot
ax_c = fig.add_subplot(gs[1, :2])
sns.boxplot(data=df, x="Category", y="Sales", palette="coolwarm", ax=ax_c)
ax_c.set_title("C  Sales by Category", fontweight="bold")
ax_c.set_xlabel("")
ax_c.tick_params(axis="x", rotation=15)

# Panel D - Outlier stats bar
ax_d = fig.add_subplot(gs[1, 2])
ax_d.bar(outlier_stats.keys(), outlier_stats.values(), color=[WARN, ACCENT, PRIMARY])
ax_d.set_title("D  Outliers Removed", fontweight="bold")
ax_d.set_ylabel("Count")

# Panel E - Top 10 Products
top5 = df.groupby("Product")["Sales"].sum().nlargest(5).sort_values()
ax_e = fig.add_subplot(gs[2, :2])
ax_e.barh(top5.index, top5.values, color=sns.color_palette("Blues_r", 5))
ax_e.set_title("E  Top 5 Products by Sales", fontweight="bold")
ax_e.set_xlabel("Total Sales (INR)")

# Panel F - Correlation heatmap (mini)
ax_f = fig.add_subplot(gs[2, 2])
mini_corr = df[["Sales", "Profit", "Quantity", "ProfitMargin"]].corr()
sns.heatmap(mini_corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax_f, cbar=False, annot_kws={"size": 8})
ax_f.set_title("F  Correlation (Key Features)", fontweight="bold")

dashboard_path = "output/dashboard.png"
plt.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"    [SAVED] {dashboard_path}")

# ===============================================================================
# 9 : SUMMARY REPORT
# ===============================================================================
print("\n" + "=" * 60)
print("  CLEANING SUMMARY REPORT")
print("=" * 60)
print(f"  Raw rows          : {df_raw.shape[0]}")
print(f"  After dedup       : {df_raw.shape[0] - dup_before}")
print(f"  After outliers    : {len(df)}")
print(f"  Missing fixed     : {missing_before.sum()} values -> 0")
print(f"  Duplicates removed: {dup_before}")
print(f"  Outliers removed  : {sum(outlier_stats.values())} rows")
print(f"  Final dataset     : {len(df)} rows x {len(df.columns)} columns")
print(f"  Outputs saved in  : ./output/")
print("=" * 60)
print("\n  Individual plots  : 01_missing_values.png ... 10_sales_vs_profit_scatter.png")
print("  Dashboard         : dashboard.png")
print("\n[SUCCESS] All done!\n")
