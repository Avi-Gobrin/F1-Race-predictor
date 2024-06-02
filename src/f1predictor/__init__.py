"""F1 race winner predictor.

A supervised machine-learning pipeline that predicts Formula 1 race winners
from driver and constructor rolling form, qualifying results, track
characteristics, and weather. Models are trained and evaluated walk-forward:
each season is predicted using only the seasons that preceded it.
"""

__version__ = "0.1.0"
