# Supply Chain Deliveries Analysis & Revenue Prediction

A portfolio-ready data analytics project exploring supply-chain delivery activity and revenue across customers, locations, and business types. The dataset contains **126,255 records** from **January 2020 through June 2025**.

## Project goals

- Explore delivery volume, orders, pieces, and revenue over time.
- Compare performance across customers, locations, and business types.
- Build a reproducible baseline model for revenue prediction.
- Provide reusable Python code and a notebook for further experimentation.

## Dataset

| Field | Description |
|---|---|
| `WorkDate` | Date of delivery activity |
| `Customer` | Customer/account |
| `Location` | Operating market/location |
| `BusinessType` | First Mile, Middle Mile, or Final Mile |
| `OrderCount` | Number of orders |
| `NumberOfPieces` | Number of pieces handled |
| `TotalRevenue` | Revenue for the record |

The supplied data has no missing values in these seven columns.

## Snapshot

- **126,255** rows
- **12** customers
- **17** locations
- **3** business types
- **3,470,467** orders
- **17,353,481** pieces
- **$330.7M** total revenue
- Final Mile is the largest business type by revenue at approximately **$197.1M**.
- Home Depot is the largest customer by revenue at approximately **$116.4M**.
- Chicago is the largest location by revenue at approximately **$51.4M**.

## Repository structure

```text
supply-chain-deliveries-analysis/
├── data/
│   └── supply_chain_deliveries.csv
├── images/
│   └── charts/
├── notebooks/
│   └── supply_chain_analysis.ipynb
├── reports/
│   └── original_kaggle_reference.pdf
├── src/
│   └── analysis.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Quick start

```bash
git clone <your-repository-url>
cd supply-chain-deliveries-analysis
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/analysis.py
```

Or open `notebooks/supply_chain_analysis.ipynb` in Jupyter or VS Code.

## Modeling approach

The baseline model predicts `TotalRevenue` from date features, customer, location, business type, order count, and number of pieces. It uses a chronological train/test split to reduce time leakage, one-hot encoding for categorical variables, and a `RandomForestRegressor`. Evaluation includes MAE, RMSE, and R².

> This is a baseline portfolio model rather than a production forecasting system. Future work could aggregate revenue to daily/weekly time series and compare dedicated forecasting models.

## Future improvements

- Add time-series forecasting and backtesting.
- Engineer lag and rolling-window features.
- Add model explainability and feature importance.
- Build an interactive dashboard.
- Add automated tests and CI.

## Reference

`reports/original_kaggle_reference.pdf` preserves the supplied one-page Kaggle notebook reference screenshot. The analysis in this repository is independently reproducible from the included CSV.

## License

Code is released under the MIT License. Review the source dataset's original terms before redistributing or reusing the data outside this project.
