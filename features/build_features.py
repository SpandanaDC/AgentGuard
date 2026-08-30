import os
import pandas as pd
import numpy as np

def build_features(input_path="data/transactions.csv", output_path="features/feature_table.csv"):
    print("⚙️ Building leakage-free features for ImposterAgent...")
    
    # Check potential input paths
    if not os.path.exists(input_path):
        alt_path = "data/raw_transactions.csv"
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            print(f"⚠️ Error: Raw transactions not found at {input_path} or {alt_path}. Run python data/simulator.py first.")
            return
            
    df = pd.read_csv(input_path)
    
    # Ensure txn_seq_num exists; create it dynamically if absent from simulator output
    if 'txn_seq_num' not in df.columns:
        sort_cols = ['agent_id', 'day'] if 'day' in df.columns else ['agent_id']
        df = df.sort_values(by=sort_cols).reset_index(drop=True)
        df['txn_seq_num'] = df.groupby('agent_id').cumcount() + 1
    else:
        df = df.sort_values(by=['agent_id', 'txn_seq_num']).reset_index(drop=True)
        
    # Ensure helper columns exist safely
    if 'device_id' not in df.columns:
        df['device_id'] = 'device_primary'
    if 'agent_age_at_txn' not in df.columns:
        df['agent_age_at_txn'] = df['txn_seq_num'] * 2
    if 'token_scope_ok' not in df.columns:
        df['token_scope_ok'] = 1

    # 1. Leakage-free expanding z-score for amount (using shift(1))
    df['agent_cum_mean'] = df.groupby('agent_id')['amount'].transform(lambda x: x.expanding().mean().shift(1))
    df['agent_cum_std'] = df.groupby('agent_id')['amount'].transform(lambda x: x.expanding().std().shift(1)).fillna(1.0)
    df['amount_zscore'] = (df['amount'] - df['agent_cum_mean']) / df['agent_cum_std']
    df['amount_zscore'] = df['amount_zscore'].fillna(0.0)
    
    # 2. Category seen before (prior-only expanding set check)
    def running_category_seen(series):
        seen = set()
        result = []
        for cat in series:
            result.append(1 if cat in seen else 0)
            seen.add(cat)
        return pd.Series(result, index=series.index)
        
    if 'category' in df.columns:
        df['category_seen_before'] = df.groupby('agent_id')['category'].transform(running_category_seen)
    else:
        df['category_seen_before'] = 0
    
    # 3. Running daily transaction count
    if 'day' in df.columns:
        df['day_txn_count'] = df.groupby(['agent_id', 'day']).cumcount() + 1
    else:
        df['day_txn_count'] = 1
    
    # 4. Device matches primary (Computed strictly from prior transactions)
    def running_primary_device(sub_df):
        devices = []
        result = []
        for dev in sub_df['device_id']:
            if not devices:
                result.append(1)
            else:
                primary = max(set(devices), key=devices.count)
                result.append(1 if dev == primary else 0)
            devices.append(dev)
        return pd.Series(result, index=sub_df.index)
        
    df['device_matches_primary'] = df.groupby('agent_id').apply(running_primary_device).reset_index(level=0, drop=True)
    
    # 5. Amount to limit ratio & Spending limits
    df['amount_to_limit_ratio'] = df['amount'] / 5000.0
    df['within_spending_limit'] = (df['amount'] <= 5000.0).astype(int)
    
    # Select clean available feature table columns
    desired_cols = [
        'txn_id', 'agent_id', 'user_id', 'merchant_id', 'category', 'day', 'label',
        'txn_seq_num', 'amount', 'amount_zscore', 'category_seen_before', 
        'device_matches_primary', 'day_txn_count', 'amount_to_limit_ratio', 
        'agent_age_at_txn', 'token_scope_ok', 'within_spending_limit'
    ]
    feature_cols = [c for c in desired_cols if c in df.columns]
    
    feature_df = df[feature_cols].copy()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    print(f"✅ Built {feature_df.shape[0]} rows x {feature_df.shape[1]} columns successfully. Saved to {output_path}")

if __name__ == "__main__":
    build_features()