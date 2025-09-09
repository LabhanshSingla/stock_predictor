# src/stock_predictor/main.py
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Any, Dict, List
import datetime
from fastapi.encoders import jsonable_encoder 
from .tools import fetch_prices_fn, feature_engineer_fn, train_linear_fn, predict_close_fn

def _to_jsonable(v):
    # pandas Timestamp / numpy datetime64 -> ISO string
    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()

    # numpy scalars -> Python scalars
    if isinstance(v, np.generic):
        return v.item()

    # leave Python primitives as-is
    return v
    

# NEW: Bedrock tool (add the files I gave earlier)
try:
    from .tools import LLMPredictTool  # provided by tools/llm_predict_tool.py
except Exception:  # keep server booting even if Bedrock isn't configured yet
    LLMPredictTool = None  # type: ignore

app = FastAPI(title="CrewAI Stock Predictor")

# front-end is inside the package
PKG_DIR = Path(__file__).resolve().parent                    # .../src/stock_predictor
FRONT_DIR = PKG_DIR / "frontend"
FRONT_PATH = FRONT_DIR / "index.html"

# Optional: serve any static assets from the same folder
app.mount("/static", StaticFiles(directory=str(FRONT_DIR)), name="static")

@app.get("/")
def root():
    if FRONT_PATH.exists():
        return FileResponse(str(FRONT_PATH))
    # fallback inline HTML so you’re never blocked
    return HTMLResponse("<h3>Frontend not found. Create src/stock_predictor/frontend/index.html</h3>")

@app.get("/favicon.ico")
def favicon():
    fav = FRONT_DIR / "favicon.ico"
    if fav.exists():
        return FileResponse(str(fav))
    return JSONResponse({}, status_code=204)

# ----- helpers -----
from typing import Any, Dict, List, Iterable

def _as_list(x: Any) -> List[Any] | None:
    """Coerce column-like (list/tuple/np.ndarray/pd.Series) into a list."""
    if x is None:
        return None
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes, dict)):
        try:
            return list(x)
        except Exception:
            return None
    return None

def _pick_first_level_columns_from_multiindex_dict(data: Dict[Any, Any]) -> Dict[str, List[Any] | None]:
    """
    Handle dicts with MultiIndex-like keys, e.g. {('Open','AAPL'): [...], ('Close','AAPL'): [...], ('Date',''): [...] }.
    Returns a lowercase name->list mapping for date/open/high/low/close/volume if present.
    """
    want = {"date": None, "open": None, "high": None, "low": None, "close": None, "volume": None}
    for k, v in data.items():
        # derive first-level key name
        if isinstance(k, tuple) and len(k) >= 1:
            name = str(k[0])
        else:
            name = str(k)
        key = name.strip().lower()
        if key in want:
            want[key] = _as_list(v)
    return want

def _ohlcv_rows_from(collected: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize to list of dicts:
    [{"date": "...", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...]
    Supports:
      A) collected["data"] as a column dict (simple keys)
      B) collected["data"] as a column dict with MultiIndex/tuple keys (your case)
      C) collected["data"] as a pandas.DataFrame
      D) collected["rows"] as a list of row dicts
    """
    norm: List[Dict[str, Any]] = []
    data = collected.get("data")

    # --- Case A/B: column dict under "data"
    if isinstance(data, dict):
        # Try simple keys first
        dates  = _as_list(data.get("Date")  or data.get("date"))
        opens  = _as_list(data.get("Open")  or data.get("open"))
        highs  = _as_list(data.get("High")  or data.get("high"))
        lows   = _as_list(data.get("Low")   or data.get("low"))
        closes = _as_list(data.get("Close") or data.get("close"))
        vols   = _as_list(data.get("Volume") or data.get("volume"))

        # If simple keys failed, try MultiIndex/tuple keys
        if not any([dates, opens, highs, lows, closes, vols]):
            picked = _pick_first_level_columns_from_multiindex_dict(data)
            dates  = dates  or picked.get("date")
            opens  = opens  or picked.get("open")
            highs  = highs  or picked.get("high")
            lows   = lows   or picked.get("low")
            closes = closes or picked.get("close")
            vols   = vols   or picked.get("volume")

        cols = [c for c in [dates, opens, highs, lows, closes, vols] if c is not None]
        if cols and all(isinstance(c, list) and len(c) for c in cols):
            n = min(len(c) for c in cols)
            for i in range(n):
                norm.append({
                    "date":   dates[i]  if dates  else None,
                    "open":   opens[i]  if opens  else None,
                    "high":   highs[i]  if highs  else None,
                    "low":    lows[i]   if lows   else None,
                    "close":  closes[i] if closes else None,
                    "volume": vols[i]   if vols   else None,
                })

    # --- Case C: pandas DataFrame under "data"
    if not norm and data is not None and hasattr(data, "__getitem__") and hasattr(data, "columns"):
        try:
            cols = data.columns
            def col(name):
                # support ('Open','AAPL') style MultiIndex columns in DataFrame too
                if name in cols:
                    return _as_list(data[name])
                # find any MultiIndex level-0 match
                for c in cols:
                    try:
                        lvl0 = c[0] if isinstance(c, tuple) and len(c) > 0 else c
                    except Exception:
                        lvl0 = c
                    if str(lvl0).lower() == name.lower():
                        return _as_list(data[c])
                return None

            dates  = col("Date")
            opens  = col("Open")
            highs  = col("High")
            lows   = col("Low")
            closes = col("Close")
            vols   = col("Volume")
            parts = [x for x in [dates, opens, highs, lows, closes, vols] if x is not None]
            if parts and all(len(x) for x in parts):
                n = min(len(x) for x in parts)
                for i in range(n):
                    norm.append({
                        "date":   dates[i]  if dates  else None,
                        "open":   opens[i]  if opens  else None,
                        "high":   highs[i]  if highs  else None,
                        "low":    lows[i]   if lows   else None,
                        "close":  closes[i] if closes else None,
                        "volume": vols[i]   if vols   else None,
                    })
        except Exception:
            pass

    # --- Case D: row-wise dicts
    if not norm:
        rows = collected.get("rows")
        if isinstance(rows, list):
            for r in rows:
                if isinstance(r, dict):
                    norm.append({
                        "date":   r.get("Date")   or r.get("date"),
                        "open":   r.get("Open")   or r.get("open"),
                        "high":   r.get("High")   or r.get("high"),
                        "low":    r.get("Low")    or r.get("low"),
                        "close":  r.get("Close")  or r.get("close"),
                        "volume": r.get("Volume") or r.get("volume"),
                    })

    if not norm:
        # include a peek at keys to aid debugging if it ever fails again
        peek_keys = []
        if isinstance(data, dict):
            for i, k in enumerate(data.keys()):
                if i >= 6: break
                peek_keys.append(repr(k))
        raise ValueError(
            "Could not normalize OHLCV: expected 'data' columns (supports MultiIndex) or "
            "'rows' as list. Sample data keys: " + ", ".join(peek_keys)
        )

    # Most-recent-first if dates look increasing
    try:
        if len(norm) >= 2 and str(norm[0]["date"]) < str(norm[-1]["date"]):
            norm = list(reversed(norm))
    except Exception:
        pass

    return norm


# ----- existing numeric pipeline -----
@app.get("/predict")
def predict(
    ticker: str = Query(..., min_length=1),
    period: str = "3mo",
    interval: str = "1d",
    # NEW: optionally route to Bedrock LLM from the same endpoint
    use_llm: bool = False,
    horizon: int = 5,
):
    try:
        collected = fetch_prices_fn(ticker=ticker.upper(), period=period, interval=interval)

        if use_llm:
            if LLMPredictTool is None:
                return JSONResponse({"error": "LLM not available. Install and configure Bedrock tool."}, status_code=501)
            llm = LLMPredictTool()
            ohlcv = _ohlcv_rows_from(collected)
            # TODO: plug your news fetcher here. empty list is OK.
            news: List[Dict[str, str]] = []
            llm_pred = llm.run(ticker.upper(), ohlcv, news, horizon_days=horizon)
            return JSONResponse({"source": "bedrock_llm", "ticker": ticker.upper(), "horizon_days": horizon, "prediction": llm_pred})

        # default numeric path
        feats = feature_engineer_fn(payload=collected)
        trained = train_linear_fn(payload=feats)
        result = predict_close_fn(payload=trained, ticker=ticker.upper())
        return JSONResponse({"source": "linear_model", **result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# in src/stock_predictor/main.py

@app.get("/history")
def history(ticker: str, period: str = "3mo", interval: str = "1d"):
    try:
        collected = fetch_prices_fn(ticker=ticker.upper(), period=period, interval=interval)
        rows = _ohlcv_rows_from(collected)

        # sort oldest -> newest for charts
        if len(rows) >= 2 and str(rows[0]["date"]) > str(rows[-1]["date"]):
            rows = list(reversed(rows))

        # sanitize rows to pure Python types
        clean_rows = []
        for r in rows:
            clean_rows.append({
                "date": _to_jsonable(r.get("date")),
                "open": float(r["open"])   if r.get("open")   is not None else None,
                "high": float(r["high"])   if r.get("high")   is not None else None,
                "low":  float(r["low"])    if r.get("low")    is not None else None,
                "close":float(r["close"])  if r.get("close")  is not None else None,
                "volume": int(r["volume"]) if r.get("volume") is not None else None,
            })

        payload = {
            "ticker": ticker.upper(),
            "rows": clean_rows,
            "dates": [cr["date"] for cr in clean_rows],
            "open":  [cr["open"] for cr in clean_rows],
            "high":  [cr["high"] for cr in clean_rows],
            "low":   [cr["low"]  for cr in clean_rows],
            "close": [cr["close"] for cr in clean_rows],
            "volume":[cr["volume"] for cr in clean_rows],
        }

        # ensure all nested types are JSON-safe
        return JSONResponse(content=jsonable_encoder(payload))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# NEW: explicit LLM route (handy for frontend)
@app.get("/llm_predict")
def llm_predict(
    ticker: str = Query(..., min_length=1),
    horizon: int = 5,
    period: str = "3mo",
    interval: str = "1d",
):
    if LLMPredictTool is None:
        return JSONResponse({"error": "LLM not available. Install and configure Bedrock tool."}, status_code=501)
    try:
        collected = fetch_prices_fn(ticker=ticker.upper(), period=period, interval=interval)
        ohlcv = _ohlcv_rows_from(collected)
        news: List[Dict[str, str]] = []  # TODO: supply news items when you add a fetcher
        llm = LLMPredictTool()
        pred = llm.run(ticker.upper(), ohlcv, news, horizon_days=horizon)
        return JSONResponse({"ticker": ticker.upper(), "horizon_days": horizon, **pred})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# Optional simple health
@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True})
@app.get("/debug/shape")
def debug_shape(ticker: str = Query("AAPL"), period: str = "3mo", interval: str = "1d"):
    collected = fetch_prices_fn(ticker=ticker.upper(), period=period, interval=interval)
    info = {
        "top_keys": list(collected.keys()),
        "data_type": type(collected.get("data")).__name__,
        "rows_type": type(collected.get("rows")).__name__,
        "len_rows": (len(collected.get("rows")) if isinstance(collected.get("rows"), list) else str(collected.get("rows"))),
        "data_keys": (list(collected["data"].keys()) if isinstance(collected.get("data"), dict) else None),
        "has_df_columns": (hasattr(collected.get("data"), "columns")),
    }
    return JSONResponse(info)

@app.get("/debug/ohlcv")
def debug_ohlcv(ticker: str = Query("AAPL"), period: str = "3mo", interval: str = "1d", n: int = 5):
    collected = fetch_prices_fn(ticker=ticker.upper(), period=period, interval=interval)
    rows = _ohlcv_rows_from(collected)
    return JSONResponse({"count": len(rows), "head": rows[:max(0, min(n, len(rows)))]})
