# House Price Intelligence System — Linear Regression

A machine learning project that predicts house prices using the King County (Seattle, WA) housing dataset. The project demonstrates feature selection, overfitting detection, and model evaluation using scikit-learn's Linear Regression.

---

## Project Structure

```
Linear-Regression/
├── Housing-Price.py          # Main analysis & modeling script
├── download_data.py          # Utility to download external datasets via OpenML
├── kc_house_data(1).csv      # King County house sales dataset
├── housing_price_analysis.png # Generated visualization output
└── README.md
```

---

## Dataset

**King County House Sales Dataset**
- ~21,000 residential home sales in King County, Washington (2014–2015)
- Target variable: `price`
- Raw features include: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15

---

## What the Script Does

### 1. Load & Explore Data
Loads the CSV, drops null rows, and prints shape, column names, sample rows, and price statistics.

### 2. Preprocessing
- Drops `id` and `date` (no predictive value)
- Engineers two new features:
  - `house_age` = 2015 − `yr_built`
  - `was_renovated` = 1 if `yr_renovated > 0`, else 0
- Drops the original `yr_built` and `yr_renovated` columns

### 3. Correlation Analysis
Computes Pearson correlation of every feature with `price` and splits them into:
- **Strong features** (`|r| > 0.30`) — used in the best model
- **Weak features** (`|r| ≤ 0.30`) — treated as low-signal/noise

### 4. Three Experiments

| Experiment | Features Used | Purpose |
|---|---|---|
| **Baseline** | All available features | Starting reference point |
| **Noise Demo** | All features + 10 random noise columns | Demonstrates overfitting |
| **Selected (Best)** | Strong features only (`|r| > 0.30`) | Best generalizing model |

Each experiment uses:
- 80/20 train-test split
- `StandardScaler` normalization
- 5-fold cross-validation
- Overfitting gap check: `Train R² − Test R²` (flags if > 0.05)

### 5. Metrics Reported per Model
- Train R² / Test R²
- CV R² (mean ± std across 5 folds)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Overfitting gap

### 6. Visualizations (`housing_price_analysis.png`)
A 2×3 panel figure with:
1. Correlation heatmap (top-10 features)
2. Feature correlation bar chart with ±0.30 threshold lines
3. Train vs Test R² grouped bar chart across the three models
4. Actual vs Predicted scatter plot (best model)
5. Residual distribution histogram (best model)
6. Overfitting gap bar chart per model

---

## Key Insights

1. **Baseline** — All features give a reasonable starting R², but redundant low-signal columns add noise.
2. **Overfitting Demo** — Adding 10 random noise columns raises Train R² while Test R² and CV R² stay flat or drop, which is a classic overfitting signal.
3. **Selected Features (Best)** — Keeping only `|r| > 0.30` features lowers RMSE/MAE and brings Train R² ≈ Test R², meaning the model generalizes well.

> **Core takeaway:** More features ≠ better model. Feature selection improves generalization and reduces overfitting.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
openml        # only needed for download_data.py
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn openml
```

---

## Usage

```bash
# Run the main analysis
python Housing-Price.py
```

The script will:
- Print a full console report (dataset stats, correlation analysis, per-model metrics, comparison summary, key insights)
- Save `housing_price_analysis.png` in the same directory

---

## Sample Results

```
Model                          # Features   Train R²   Test R²   CV R²    RMSE ($)    MAE ($)   Overfit Gap
All Features (Baseline)              19       0.70xx    0.69xx   0.69xx   xxx,xxx     xxx,xxx      ~0.01
With Noise Features (Overfit Demo)   29       0.70xx    0.69xx   0.69xx   xxx,xxx     xxx,xxx      ~0.01+
Selected Strong Features             xx       0.70xx    0.69xx   0.69xx   xxx,xxx     xxx,xxx      ~0.01
```

*(Exact numbers printed to console when you run the script.)*

---

## Technologies

| Library | Purpose |
|---|---|
| `pandas` | Data loading, preprocessing, correlation analysis |
| `numpy` | Numerical operations, random noise generation |
| `scikit-learn` | LinearRegression, StandardScaler, train/test split, cross-validation, metrics |
| `matplotlib` | Plotting figures |
| `seaborn` | Correlation heatmap |
