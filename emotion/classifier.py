# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Optional, List

from .infer import classify as heuristic_classify, EMOTIONS


class HeuristicClassifier:
    def classify(self, text: str, context: Optional[List[str]] = None) -> str:
        try:
            return heuristic_classify(text, context)
        except Exception as e:
            logging.warning(f"HeuristicClassifier error: {e}")
            return "neutral"
