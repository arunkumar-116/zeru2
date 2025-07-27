#  DeFi Wallet Risk Scoring using Compound V2/V3 & GraphQL

This project implements a data-driven pipeline to assign **risk scores (100–1000)** to wallet addresses based on historical activity from **Compound V2/V3 protocols**, using The Graph’s hosted API.

##  Project Overview

The goal is to analyze borrowing and transfer behavior from DeFi wallets to compute a **synthetic credit score** using:
- Blockchain data queried from Compound's subgraph
- Engineered features from protocol and user activity
- A trained regression model (Random Forest)

##  Data Collection

Wallet-level data was collected using [The Graph](https://thegraph.com/) GraphQL endpoint by querying:
- `borrows` (loan activity)
- `transfers` (token movement)
- `markets` (protocol-level liquidity/borrowing info)
- `user` (wallet-level identity)

**Tools used:**
- GraphQL API via `requests` module
- Wallet list from `Wallet id - Sheet1.csv`
- Subgraph ID and API key passed via `.env`

### 🔍 Sample GraphQL Query
```graphql
{
  user(id: "0x123...") {
    id
  }
  borrows(first: 2) {
    underlyingAmount
    blockNumber
  }
  transfers(first: 2) {
    id
  }
  markets(first: 3) {
    id
    totalBorrow
    totalSupply
    cTokenSymbol
  }
}
```
##  Feature Engineering

Each wallet is represented using a set of derived features from GraphQL queries to the Compound protocol subgraph. These features capture wallet behavior, activity, and interaction patterns within the protocol.

| Feature | Description |
|---------|-------------|
| `num_borrows` | Total number of borrow transactions by the wallet |
| `total_borrowed_amount` | Sum of the `underlyingAmount` field in all borrow transactions |
| `avg_borrow_block` | Average block height of all borrow transactions (used to estimate how recent borrowing activity is) |
| `num_transfers` | Number of token transfer events associated with the wallet |
| `num_markets` | Number of different lending/borrowing markets the wallet is involved in |
| `avg_market_total_borrow` | Average total borrow volume across all markets the wallet interacts with (used to infer market pressure and risk exposure) |

#  Feature Selection Rationale

This section explains the reasoning behind each feature used in the DeFi wallet risk scoring model.

- **Activity volume**: Indicates engagement with the protocol (transfers & borrows).
- **Borrow size**: Suggests the financial impact and potential risk exposure.
- **Block height average**: Newer blocks imply recent activity (recency → reliability).
- **Market interaction**: Higher diversity suggests broader participation in the protocol.
- **Market-level borrow pressure**: Proxy for market conditions the user interacts with.

#  Scoring Logic

We designed a synthetic scoring model using a combination of heuristics and supervised learning to evaluate wallet risk.

##  Synthetic Score Generation

Each wallet is initially assigned a synthetic credit score (ranging from **100 to 1000**) using the following components:

- **Base score**: 500  
- **Activity score**: Based on `num_borrows` and `num_transfers`  
- **Market score**: Based on `num_markets`  
- **Risk penalty**: Derived from `total_borrowed_amount` (scaled negatively)  
- **Recency bonus**: Based on `avg_borrow_block`  
- **Random variation**: Introduces noise to simulate real-world score dispersion  

---

##  ML Model

- **Model**: `RandomForestRegressor` from scikit-learn  
- **Target**: Synthetic score (100–1000)  
- **Features**: All engineered features (excluding `wallet_id`)  
- **Normalization**: Features scaled using `StandardScaler`  
#  Scalability & Extensibility

- Supports any number of wallets from a CSV  
- **Schema-agnostic**: Easily adaptable to other protocols (e.g., Aave, MakerDAO)  
- Score generation & model training are **fully automated**  
- Clean and modular pipeline for future enhancements (e.g., time-series models)  

---

#  Output

- **`wallet_scores.csv`**: Final predicted wallet scores  
- **`graph_wallet_data.json`**: Raw API response from The Graph  
- **Terminal prints**: Feature statistics, model evaluation, score distribution  

