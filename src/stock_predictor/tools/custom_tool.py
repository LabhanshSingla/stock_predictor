# src/stock_predictor/tools/custom_tool.py
from crewai.tools import tool
import base64, pickle
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---- pure helpers you can call from FastAPI ----
def _df_to_cols_dict(df: pd.DataFrame):
    return df.to_dict(orient="list")

def _cols_dict_to_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame(d)

def _pack_model(model) -> str:
    b = pickle.dumps(model)
    return base64.b64encode(b).decode()

def _unpack_model(s: str):
    b = base64.b64decode(s.encode())
    return pickle.loads(b)

def fetch_prices_fn(ticker: str, period: str = "3mo", interval: str = "1d") -> dict:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return {
        "rows": len(df),
        "head": df.head().to_dict(orient="records"),
        "tail": df.tail().to_dict(orient="records"),
        "data": _df_to_cols_dict(df)
    }

def feature_engineer_fn(payload: dict) -> dict:
    df = _cols_dict_to_df(payload["data"])
    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
    df = df.dropna().reset_index(drop=True)
    return {"data": _df_to_cols_dict(df)}

def train_linear_fn(payload: dict) -> dict:
    df = _cols_dict_to_df(payload["data"])
    df["t"] = np.arange(len(df))
    X = df[["t"]].values
    y = df["Close"].values
    model = LinearRegression().fit(X, y)
    return {"model_b64": _pack_model(model), "data": _df_to_cols_dict(df)}

# src/stock_predictor/tools/custom_tool.py

def predict_close_fn(payload: dict, ticker: str) -> dict:
    model = _unpack_model(payload["model_b64"])
    df = _cols_dict_to_df(payload["data"])

    # guard rails
    if df.empty or "Close" not in df.columns:
        raise ValueError("No price data available after preprocessing.")

    # use positional indexing
    last_close = float(df["Close"].iloc[-1])

    # predict the next time step
    next_t = np.array([[len(df)]])
    est = float(model.predict(next_t)[0])

    change = (est - last_close) / last_close * 100.0
    rationale = (
        f"Educational demo using linear trend on {len(df)} points; "
        f"features (SMA/Momentum) provide context. Not financial advice."
    )
    return {
        "ticker": ticker,
        "last_close": round(last_close, 4),
        "predicted_close": round(est, 4),
        "pct_move": round(change, 3),
        "rationale": rationale
    }


# ---- CrewAI tool-wrapped versions for agents (no return_direct!) ----
@tool("fetch_prices")
def fetch_prices_tool(ticker: str, period: str = "3mo", interval: str = "1d") -> dict:
    """Fetch adjusted OHLCV for a ticker."""
    return fetch_prices_fn(ticker, period, interval)

@tool("feature_engineer")
def feature_engineer_tool(payload: dict) -> dict:
    """Add simple features to a price frame."""
    return feature_engineer_fn(payload)

@tool("train_linear")
def train_linear_tool(payload: dict) -> dict:
    """Fit LinearRegression on time index -> Close."""
    return train_linear_fn(payload)

@tool("predict_close")
def predict_close_tool(payload: dict, ticker: str) -> dict:
    """Predict next-day Close and summarise."""
    return predict_close_fn(payload, ticker)
