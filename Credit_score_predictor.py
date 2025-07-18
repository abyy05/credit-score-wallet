import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from datetime import datetime

# Load JSON data
json_path = 'zeru/user-wallet-transactions.json'
if not os.path.exists(json_path):
    raise FileNotFoundError(f"File not found: {json_path}")

with open(json_path, 'r') as file:
    data = json.load(file)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Extract relevant fields from actionData
df['amount'] = df['actionData'].apply(lambda x: int(x['amount']))
df['asset_symbol'] = df['actionData'].apply(lambda x: x['assetSymbol'])
df['asset_price_usd'] = df['actionData'].apply(lambda x: float(x['assetPriceUSD']))

# Adjust for token decimals
df['usd_value'] = df.apply(
    lambda row: (row['amount'] * row['asset_price_usd']) / (10*6 if row['asset_symbol'] == 'USDC' else 10*18), axis=1
)

# Timestamp conversion
df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')

# Define stablecoins
stablecoins = ['USDC', 'USDT', 'DAI']

# Feature Engineering
def engineer_features(df):
    features = []
    for wallet in df['userWallet'].unique():
        wallet_df = df[df['userWallet'] == wallet]
        tx_count = len(wallet_df)

        deposit_df = wallet_df[wallet_df['action'] == 'deposit']
        total_deposit_usd = deposit_df['usd_value'].sum()

        redeem_df = wallet_df[wallet_df['action'] == 'redeemunderlying']
        total_redeem_usd = redeem_df['usd_value'].sum()

        borrow_df = wallet_df[wallet_df['action'] == 'borrow']
        total_borrow_usd = borrow_df['usd_value'].sum()

        repay_df = wallet_df[wallet_df['action'] == 'repay']
        total_repay_usd = repay_df['usd_value'].sum()

        liquidation_count = len(wallet_df[wallet_df['action'] == 'liquidationcall'])

        net_deposit_usd = total_deposit_usd - total_redeem_usd
        net_contribution = total_deposit_usd - total_borrow_usd
        repay_borrow_ratio = total_repay_usd / (total_borrow_usd + 1e-6) if total_borrow_usd > 0 else 1.0
        liquidation_ratio = liquidation_count / tx_count if tx_count > 0 else 0.0
        asset_diversity = len(wallet_df['asset_symbol'].unique())
        avg_tx_value = wallet_df['usd_value'].mean() if tx_count > 0 else 0
        time_span = (wallet_df['timestamp'].max() - wallet_df['timestamp'].min()).days + 1
        deposit_to_redeem_ratio = total_deposit_usd / (total_redeem_usd + 1e-6)
        stablecoin_tx_count = len(wallet_df[wallet_df['asset_symbol'].isin(stablecoins)])
        stablecoin_proportion = stablecoin_tx_count / tx_count if tx_count > 0 else 0
        high_value_tx_count = len(wallet_df[wallet_df['usd_value'] > 10000])
        tx_per_day = tx_count / time_span if time_span > 0 else 0

        features.append({
            'wallet': wallet,
            'tx_count': tx_count,
            'total_deposit_usd': total_deposit_usd,
            'total_redeem_usd': total_redeem_usd,
            'total_borrow_usd': total_borrow_usd,
            'total_repay_usd': total_repay_usd,
            'net_deposit_usd': net_deposit_usd,
            'net_contribution': net_contribution,
            'repay_borrow_ratio': repay_borrow_ratio,
            'liquidation_ratio': liquidation_ratio,
            'asset_diversity': asset_diversity,
            'avg_tx_value': avg_tx_value,
            'time_span': time_span,
            'deposit_to_redeem_ratio': deposit_to_redeem_ratio,
            'stablecoin_proportion': stablecoin_proportion,
            'high_value_tx_count': high_value_tx_count,
            'tx_per_day': tx_per_day
        })

    return pd.DataFrame(features)

# Generate features
features_df = engineer_features(df)

# Normalize features
scaler = MinMaxScaler()
feature_columns = [
    'tx_count', 'total_deposit_usd', 'net_deposit_usd', 'net_contribution',
    'repay_borrow_ratio', 'liquidation_ratio', 'asset_diversity',
    'avg_tx_value', 'time_span', 'deposit_to_redeem_ratio',
    'stablecoin_proportion', 'high_value_tx_count', 'tx_per_day'
]
features_df[feature_columns] = scaler.fit_transform(features_df[feature_columns])

# Define weights
weights = {
    'tx_count': 0.05,
    'total_deposit_usd': 0.15,
    'net_deposit_usd': 0.15,
    'net_contribution': 0.15,
    'repay_borrow_ratio': 0.15,
    'liquidation_ratio': 0.1,  # Invert in scoring
    'asset_diversity': 0.05,
    'avg_tx_value': 0.05,
    'time_span': 0.05,
    'deposit_to_redeem_ratio': 0.1,
    'stablecoin_proportion': 0.05,
    'high_value_tx_count': 0.05,
    'tx_per_day': 0.05
}

# Compute credit score
features_df['credit_score'] = (
    features_df['tx_count'] * weights['tx_count'] +
    features_df['total_deposit_usd'] * weights['total_deposit_usd'] +
    features_df['net_deposit_usd'] * weights['net_deposit_usd'] +
    features_df['net_contribution'] * weights['net_contribution'] +
    features_df['repay_borrow_ratio'] * weights['repay_borrow_ratio'] +
    (1 - features_df['liquidation_ratio']) * weights['liquidation_ratio'] +
    features_df['asset_diversity'] * weights['asset_diversity'] +
    features_df['avg_tx_value'] * weights['avg_tx_value'] +
    features_df['time_span'] * weights['time_span'] +
    features_df['deposit_to_redeem_ratio'] * weights['deposit_to_redeem_ratio'] +
    features_df['stablecoin_proportion'] * weights['stablecoin_proportion'] +
    features_df['high_value_tx_count'] * weights['high_value_tx_count'] +
    features_df['tx_per_day'] * weights['tx_per_day']
)

# Scale to 0–1000
features_df['credit_score'] = MinMaxScaler(feature_range=(0, 1000)).fit_transform(
    features_df[['credit_score']]
)

# Output result
result = features_df[['wallet', 'credit_score']].copy()
result['credit_score'] = result['credit_score'].round().astype(int)
print(result.to_string(index=False))

# Save to CSV
result.to_csv('wallet_credit_scores.csv', index=False)

# Train dummy model (replace with real targets if available)
X = features_df[feature_columns]
synthetic_target = features_df['credit_score']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, synthetic_target)

# Save model
joblib.dump(model, 'credit_score_model.pkl')
print("Model saved to credit_score_model.pkl")