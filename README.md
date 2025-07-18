
# Wallet Credit Scoring using Blockchain Transaction Analysis

This project builds a **credit scoring system** for crypto wallets based on their DeFi transaction history. It extracts features from user transactions, performs feature engineering and normalization, and assigns a credit score between **0–1000** using a weighted scoring model. It also trains a simple regression model to predict the score.

---

## Objective

The goal is to assign a **credit score** to crypto wallets by analyzing:
- Deposit/borrow behavior
- Repayment patterns
- Activity with stablecoins
- Risk indicators like liquidation frequency
- Asset diversification and value of transactions


## Tech Stack

- **Language**: Python
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `joblib`
- **Model**: Random Forest Regressor
- **Input**: JSON file with DeFi wallet transactions
- **Output**: Credit score (0–1000) and trained model file

---

##  Data Flow & Architecture

```plaintext
         JSON File (user-wallet-transactions.json)
                       ↓
         Transaction Data Preprocessing
                       ↓
        Feature Extraction & Engineering
                       ↓
     Feature Normalization using MinMaxScaler
                       ↓
     Weighted Rule-Based Credit Score Calculation
                       ↓
     → Credit Score CSV Output (wallet_credit_scores.csv)
     → Train & Save ML Model (credit_score_model.pkl)
````

---

##  Processing Pipeline

### 1. **Load JSON and Extract Relevant Fields**

* Extracts `amount`, `assetSymbol`, `assetPriceUSD` from nested `actionData`
* Converts UNIX timestamps to datetime format

### 2. **USD Value Calculation**

* Converts token `amount` to USD using token price and decimals (USDC = 6, others = 18)

### 3. **Feature Engineering per Wallet**

Computed features include:

* Total and net deposits, borrows, repayments
* Repayment/borrow ratio
* Liquidation count and ratio
* Average transaction value
* Asset diversity
* Deposit-to-redeem ratio
* Stablecoin usage
* High-value transaction count
* Transaction frequency per day

### 4. **Feature Normalization**

Using `MinMaxScaler` for fair contribution in score computation.

### 5. **Weighted Credit Score Calculation**

Each feature is assigned a weight (based on assumed importance). Features like `liquidation_ratio` are inversely scored.

The final score is scaled to a **0–1000** range using another `MinMaxScaler`.

### 6. **Machine Learning Model**

A `RandomForestRegressor` is trained on the features with the calculated score as a synthetic target. The model is saved as `credit_score_model.pkl` for further use or API deployment.

---

## Output Files

* `credit_score_model.pkl`: Trained model file (useful for inference on new wallets)

---

## Example Credit Score Output

```plaintext
       wallet  credit_score
0  0xabc123...           872
1  0xdef456...           610
2  0xghi789...           940
```

---

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-repo/Credit-score-prediction.git
cd Credit-score-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place your JSON file

Place `user-wallet-transactions.json` inside a `zeru/` folder.

### 4. Run the script

```bash
python credit_score_predictor.py
```

---

## Requirements (`requirements.txt`)

```txt
pandas
numpy
scikit-learn
joblib
```

---

## Future Enhancements

* Integrate with live blockchain APIs to pull wallet histories
* Use deep learning for complex patterns
* Add explainability to credit scores using SHAP
* Deploy as an API using Flask/FastAPI

---

