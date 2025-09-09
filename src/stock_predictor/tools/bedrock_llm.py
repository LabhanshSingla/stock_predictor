from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional, List, Literal
import boto3
from pydantic import BaseModel, Field, ValidationError

# ----- Response schema we want the LLM to follow -----
class LLMPrediction(BaseModel):
    ticker: str
    horizon_days: int = Field(..., ge=1, le=30)
    expected_move: Literal["up", "down", "flat"]
    confidence_pct: float = Field(..., ge=0, le=100)
    rationale: str
    price_target: Optional[float] = None
    risk_factors: List[str] = Field(default_factory=list)

JSON_INSTRUCTIONS = """\
You are a markets analyst LLM. Return ONLY a minified JSON object with keys:
ticker (str), horizon_days (int), expected_move ("up"|"down"|"flat"),
confidence_pct (0-100), rationale (str), price_target (float|omit), risk_factors (list of str).
Do not include markdown fences or commentary; return raw JSON.
"""

PROMPT_TEMPLATE = """\
{json_instructions}

Inputs:
- Ticker: {ticker}
- Horizon (days): {horizon}
- Recent OHLCV (most recent first, up to 30 rows):
{ohlcv_snippet}

- Recent headlines (ISO date | headline):
{news_snippet}

Task:
Synthesize signals (trend, volatility, volume, catalysts, macro). Provide a **directional** call for the horizon with confidence. If data is sparse, be conservative and say "flat". Keep rationale concise (<=120 words).
"""

def _build_bedrock_client():
    session = boto3.Session(profile_name=os.getenv("AWS_PROFILE", None), region_name=os.getenv("AWS_REGION", None))
    return session.client("bedrock-runtime")

def _default_model_id() -> str:
    # You can switch this to another Bedrock model id later (e.g., Llama 3.1 or Amazon Nova)
    return os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

class BedrockLLM:
    """
    Thin wrapper around Bedrock text models for JSON-structured predictions.
    Works with Anthropic-compatible JSON payloads. If you swap providers,
    adjust the request body accordingly.
    """
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or _default_model_id()
        self.client = _build_bedrock_client()

    def predict_stock(
        self,
        ticker: str,
        ohlcv_rows: List[Dict[str, Any]],
        news_items: List[Dict[str, str]],
        horizon_days: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> LLMPrediction:
        ohlcv_lines = []
        for r in ohlcv_rows[:30]:
            # Expecting dicts like {"date":"2025-08-20","open":..., "high":..., "low":..., "close":..., "volume":...}
            ohlcv_lines.append(f"{r.get('date')}: O={r.get('open')} H={r.get('high')} L={r.get('low')} C={r.get('close')} V={r.get('volume')}")
        ohlcv_snippet = "\n".join(ohlcv_lines) if ohlcv_lines else "N/A"

        news_lines = []
        for n in news_items[:15]:
            news_lines.append(f"{n.get('date','')}: {n.get('headline','')}")
        news_snippet = "\n".join(news_lines) if news_lines else "N/A"

        prompt = PROMPT_TEMPLATE.format(
            json_instructions=JSON_INSTRUCTIONS,
            ticker=ticker,
            horizon=horizon_days,
            ohlcv_snippet=ohlcv_snippet,
            news_snippet=news_snippet,
        )

        # Anthropic format (Bedrock)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp.get("body").read().decode("utf-8"))

        # Anthropic returns a list of content blocks; pick the first text
        text_out = ""
        for block in payload.get("content", []):
            if block.get("type") == "text":
                text_out += block.get("text", "")

        # Try to parse strict JSON (LLM should output raw JSON)
        text_out = text_out.strip()
        try:
            data = json.loads(text_out)
            return LLMPrediction.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            # Fallback: try to salvage JSON substring if model added extra text
            try:
                start = text_out.find("{")
                end = text_out.rfind("}")
                if start != -1 and end != -1:
                    data = json.loads(text_out[start:end+1])
                    return LLMPrediction.model_validate(data)
            except Exception:
                pass
            raise RuntimeError(f"Bedrock LLM returned un-parseable output: {e}\nRaw: {text_out}")
