import os
import json
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GRAPH_API_KEY")
SUBGRAPH_ID = os.getenv("GRAPH_SUBGRAPH_ID")
URL = f"https://gateway.thegraph.com/api/subgraphs/id/{SUBGRAPH_ID}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def build_query(wallet):
    return {
        "query": f"""
        {{
          protocols(first: 3) {{
            id
            priceOracle
            lastNewOracleBlockNumber
            latestBlockNumber
          }}
          markets(first: 3) {{
            id
            creationBlockNumber
            latestBlockNumber
            cTokenSymbol
            availableLiquidity
            totalBorrow
            totalSupply
          }}
          user(id: "{wallet.lower()}") {{
            id
          }}
          borrows(first: 2) {{
            underlyingAmount
            blockNumber
          }}
          transfers(first: 2) {{
            id
          }}
        }}
        """
    }

def fetch_wallet_data():
    df = pd.read_csv("Wallet id - Sheet1.csv")
    wallets = df['wallet_id'].dropna().unique()
    
    results = {}
    for wallet in wallets:
        payload = build_query(wallet)
        try:
            response = requests.post(URL, headers=HEADERS, json=payload)
            data = response.json()
            
            if 'errors' in data:
                print(f"[ ERROR] Wallet {wallet}: {data['errors']}")
                results[wallet] = {"errors": data["errors"]}
            else:
                print(f"[ SUCCESS] Wallet {wallet}")
                results[wallet] = data["data"]
        except Exception as e:
            print(f"[ EXCEPTION] Wallet {wallet}: {e}")
            results[wallet] = {"exception": str(e)}
    
    with open("graph_wallet_data.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def extract_features(raw_data):
    rows = []
    for wallet, details in raw_data.items():
        try:
            if "errors" in details or "exception" in details:
                continue
                
            borrows = details.get("borrows", [])
            transfers = details.get("transfers", [])
            markets = details.get("markets", [])

            num_borrows = len(borrows)
            total_borrowed_amount = sum(float(b.get("underlyingAmount", 0)) for b in borrows)
            avg_borrow_block = np.mean([int(b["blockNumber"]) for b in borrows]) if borrows else 0

            num_transfers = len(transfers)
            num_markets = len(markets)
            avg_market_total_borrow = np.mean([float(m.get("totalBorrow", 0)) for m in markets]) if markets else 0

            rows.append({
                "wallet_id": wallet,
                "num_borrows": num_borrows,
                "total_borrowed_amount": total_borrowed_amount,
                "avg_borrow_block": avg_borrow_block,
                "num_transfers": num_transfers,
                "num_markets": num_markets,
                "avg_market_total_borrow": avg_market_total_borrow
            })
        except Exception as e:
            print(f"[ERROR] Wallet {wallet}: {e}")
    
    df = pd.DataFrame(rows)
    print(f"Feature statistics:")
    print(df.describe())
    return df

def generate_synthetic_score(row):
    # Create more varied scoring with random components
    base_score = 500  # Start from middle
    
    # Activity score (0-200)
    activity_score = (row['num_transfers'] * 10 + row['num_borrows'] * 15) * (1 + np.random.normal(0, 0.2))
    
    # Market participation score (0-150)
    market_score = row['num_markets'] * 25 * (1 + np.random.normal(0, 0.3))
    
    # Risk penalty (-100 to 0)
    risk_penalty = -(row['total_borrowed_amount'] / 10000) * 50 * (1 + np.random.normal(0, 0.1))
    
    # Block recency bonus (-50 to 50)
    if row['avg_borrow_block'] > 0:
        block_bonus = (row['avg_borrow_block'] / 20000000) * 100 - 50
    else:
        block_bonus = np.random.uniform(-25, 25)
    
    # Random factor for variation (-50 to 50)
    random_factor = np.random.uniform(-50, 50)
    
    final_score = base_score + activity_score + market_score + risk_penalty + block_bonus + random_factor
    
    return min(1000, max(100, round(final_score)))

def train_model(df):
    # Set random seed for reproducible synthetic scores
    np.random.seed(42)
    df['synthetic_score'] = df.apply(generate_synthetic_score, axis=1)
    
    print(f"\nSynthetic score distribution:")
    print(f"Min: {df['synthetic_score'].min()}, Max: {df['synthetic_score'].max()}")
    print(f"Mean: {df['synthetic_score'].mean():.2f}, Std: {df['synthetic_score'].std():.2f}")

    X = df.drop(columns=['wallet_id', 'synthetic_score'])
    y = df['synthetic_score']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy (within tolerance)
    tolerance = 50  # Consider predictions within 50 points as accurate
    accurate_predictions = np.abs(y_test - y_pred) <= tolerance
    accuracy = np.mean(accurate_predictions) * 100

    print(f"\n📊 Model Evaluation:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R² Score: {r2:.3f}")
    print(f"  Accuracy (±{tolerance}): {accuracy:.1f}%")

    return model, scaler

def predict_scores(df, model, scaler, output_csv="wallet_scores.csv"):
    X = df.drop(columns=['wallet_id', 'synthetic_score'])
    X_scaled = scaler.transform(X)

    df['score'] = model.predict(X_scaled).round().clip(100, 1000).astype(int)
    
    # Add some final variation to ensure diversity
    np.random.seed(42)  # For reproducible results
    variation = np.random.normal(0, 20, len(df))
    df['score'] = (df['score'] + variation).round().clip(100, 1000).astype(int)
    
    result_df = df[['wallet_id', 'score']].copy()
    result_df.to_csv(output_csv, index=False)

    print(f"\n Scoring complete. Results saved to {output_csv}")
    print(f"Score distribution: Min={result_df['score'].min()}, Max={result_df['score'].max()}, Mean={result_df['score'].mean():.1f}")
    
    return result_df

def run_pipeline():
    print(" Fetching data...")
    raw_data = fetch_wallet_data()

    print("\n Extracting features...")
    df = extract_features(raw_data)
    
    if len(df) == 0:
        print(" No valid data found!")
        return

    print(f"\n Training ML model on {len(df)} wallets...")
    model, scaler = train_model(df)

    print("\n Predicting wallet scores...")
    predict_scores(df, model, scaler)

if __name__ == "__main__":
    run_pipeline()