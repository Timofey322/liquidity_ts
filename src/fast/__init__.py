from __future__ import annotations

try:
    from src.fast._fractals import fractal_highs, fractal_lows
except ImportError:  # pragma: no cover
    from src.fast.fractals_py import fractal_highs, fractal_lows

try:
    from src.fast._imbalance import detect_fvg_zones
except ImportError:  # pragma: no cover
    from src.fast.imbalance_py import detect_fvg_zones

try:
    from src.fast._footprint import footprint_histogram
except ImportError:  # pragma: no cover
    from src.fast.footprint_py import footprint_histogram
