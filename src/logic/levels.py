from __future__ import annotations

from typing import List

import numpy as np

from src.core.types import LevelCandidate


def suggest_tp_sl(
    close: float,
    direction: str,
    last_fractal_high: float | None,
    last_fractal_low: float | None,
    imb_low: float | None,
    imb_high: float | None,
    poc: float | None,
) -> List[LevelCandidate]:
    """
    Кандидаты фиксации:
    - SL: за противоположным фракталом (инвалидация структуры)
    - TP: дальняя граница имбаланса / зона POC (баланс ликвидности)
    """
    out: List[LevelCandidate] = []
    if poc is not None and np.isfinite(poc):
        out.append(LevelCandidate(float(poc), "poc", 1.0, "маркет-профиль / максимум объёма в бинах"))
    if direction == "long":
        if last_fractal_low is not None and np.isfinite(last_fractal_low):
            out.append(
                LevelCandidate(
                    float(last_fractal_low),
                    "sl",
                    1.0,
                    "SL ниже последнего фрактала вниз (инвалидация)",
                )
            )
        if imb_high is not None and np.isfinite(imb_high):
            out.append(
                LevelCandidate(
                    float(imb_high),
                    "tp",
                    0.8,
                    "TP у верхней границы бычьего имбаланса (FVG)",
                )
            )
        if last_fractal_high is not None and np.isfinite(last_fractal_high) and last_fractal_high > close:
            out.append(
                LevelCandidate(
                    float(last_fractal_high),
                    "tp",
                    0.6,
                    "TP у предыдущего фрактала вверх (ликвидность)",
                )
            )
    elif direction == "short":
        if last_fractal_high is not None and np.isfinite(last_fractal_high):
            out.append(
                LevelCandidate(
                    float(last_fractal_high),
                    "sl",
                    1.0,
                    "SL выше последнего фрактала вверх (инвалидация)",
                )
            )
        if imb_low is not None and np.isfinite(imb_low):
            out.append(
                LevelCandidate(
                    float(imb_low),
                    "tp",
                    0.8,
                    "TP у нижней границы медвежьего имбаланса (FVG)",
                )
            )
        if last_fractal_low is not None and np.isfinite(last_fractal_low) and last_fractal_low < close:
            out.append(
                LevelCandidate(
                    float(last_fractal_low),
                    "tp",
                    0.6,
                    "TP у предыдущего фрактала вниз (ликвидность)",
                )
            )
    return out
