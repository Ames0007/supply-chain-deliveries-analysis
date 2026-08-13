from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "supply_chain_deliveries.csv"
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df["WorkDate"] = pd.to_datetime(df["WorkDate"])

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

# 1. Order Count Distribution
plt.figure(figsize=(10, 6))
plt.hist(df["OrderCount"], bins=30, edgecolor="black")
plt.title("Distribution of Order Count")
plt.xlabel("Order Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    IMAGES_DIR / "order_count_distribution.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()

# 2. Total Revenue Boxplot
plt.figure(figsize=(10, 5))
plt.boxplot(df["TotalRevenue"], vert=False)
plt.title("Distribution of Total Revenue")
plt.xlabel("Total Revenue")
plt.tight_layout()
plt.savefig(
    IMAGES_DIR / "total_revenue_boxplot.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()

# 3. Business Type Distribution
business_counts = df["BusinessType"].value_counts()

plt.figure(figsize=(9, 6))
business_counts.plot(kind="bar")
plt.title("Deliveries by Business Type")
plt.xlabel("Business Type")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    IMAGES_DIR / "business_type_count.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()

# 4. Relationships Between Numeric Features
numeric_columns = [
    "OrderCount",
    "NumberOfPieces",
    "TotalRevenue",
]

sample_df = df[numeric_columns].sample(
    min(3000, len(df)),
    random_state=42,
)

pd.plotting.scatter_matrix(
    sample_df,
    figsize=(10, 10),
    diagonal="hist",
)

plt.suptitle(
    "Relationships Between Numeric Features",
    y=1.02,
)
plt.tight_layout()
plt.savefig(
    IMAGES_DIR / "numeric_features_pairplot.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()

# Revenue Prediction
X = df[["OrderCount", "NumberOfPieces"]]
y = df["TotalRevenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nRevenue Prediction Results")
print("--------------------------")
print(f"RMSE: {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

print("\nModel coefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print(f"Intercept: {model.intercept_:.4f}")

print("\nCharts generated successfully:")
for image in sorted(IMAGES_DIR.glob("*.png")):
    print(image.name)