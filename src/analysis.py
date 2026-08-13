from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "supply_chain_deliveries.csv"
CHARTS = ROOT / "images" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["WorkDate"]).sort_values("WorkDate")
print(df.info())
print(df.describe(include="all"))

monthly = df.set_index("WorkDate").resample("MS")["TotalRevenue"].sum()
ax = monthly.plot(figsize=(11, 5), title="Monthly Revenue")
ax.set_xlabel("Month"); ax.set_ylabel("Revenue ($)")
plt.tight_layout(); plt.savefig(CHARTS / "monthly_revenue.png", dpi=160); plt.close()

customer = df.groupby("Customer")["TotalRevenue"].sum().sort_values(ascending=True)
ax = customer.plot.barh(figsize=(9, 6), title="Revenue by Customer")
ax.set_xlabel("Revenue ($)")
plt.tight_layout(); plt.savefig(CHARTS / "revenue_by_customer.png", dpi=160); plt.close()

model_df = df.copy()
model_df["Year"] = model_df.WorkDate.dt.year
model_df["Month"] = model_df.WorkDate.dt.month
model_df["DayOfWeek"] = model_df.WorkDate.dt.dayofweek
features = ["Customer", "Location", "BusinessType", "OrderCount", "NumberOfPieces", "Year", "Month", "DayOfWeek"]
cat = ["Customer", "Location", "BusinessType"]
num = [c for c in features if c not in cat]
split_date = model_df.WorkDate.quantile(0.80)
train = model_df[model_df.WorkDate <= split_date]
test = model_df[model_df.WorkDate > split_date]
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", "passthrough", num)])
model = Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=18))])
model.fit(train[features], train["TotalRevenue"])
pred = model.predict(test[features])
print(f"Split date: {split_date.date()}")
print(f"MAE: ${mean_absolute_error(test['TotalRevenue'], pred):,.2f}")
print(f"RMSE: ${mean_squared_error(test['TotalRevenue'], pred) ** 0.5:,.2f}")
print(f"R2: {r2_score(test['TotalRevenue'], pred):.3f}")
