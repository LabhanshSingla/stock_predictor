from __future__ import annotations
from typing import Any, Dict, List
from .bedrock_llm import BedrockLLM, LLMPrediction

class LLMPredictTool:
    name = "llm_predict"
    description = "Use AWS Bedrock LLM to generate a short-horizon stock direction prediction from OHLCV + headlines."

    def __init__(self, model_id: str | None = None):
        self.llm = BedrockLLM(model_id=model_id)

    def run(self, ticker: str, ohlcv_rows: List[Dict[str, Any]], news_items: List[Dict[str, str]], horizon_days: int = 5) -> Dict[str, Any]:
        pred: LLMPrediction = self.llm.predict_stock(
            ticker=ticker,
            ohlcv_rows=ohlcv_rows,
            news_items=news_items,
            horizon_days=horizon_days,
        )
        return pred.model_dump()
