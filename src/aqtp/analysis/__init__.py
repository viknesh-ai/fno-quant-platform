"""Deep market analysis layer.

This package sits between raw features and the strategy/ensemble layers. Where
`features/` computes indicator *values*, this package computes market *state*:
what the order book is doing, where volume has actually traded, whether the price
series is statistically trending or mean-reverting, how dealers are positioned in
the option chain, and how the rest of the market is behaving at the same moment.

Nothing here places an order or produces a signal on its own. Every module returns
a measurement object with a signed score in [-1, 1], a quality/confidence in
[0, 1], and human-readable reasons. `confluence.py` combines them into a single
conviction with an audit trail, which the ensemble treats as evidence — never as
an instruction.
"""

from .confluence import ConfluenceEngine, ConfluenceReport, Dimension
from .crossasset import CrossAssetReport, CrossAssetTracker
from .engine import AnalysisEngine, AnalysisReport
from .microstructure import MicrostructureReport, MicrostructureTracker
from .options_analytics import OptionsAnalyticsReport, analyze_chain
from .statistics import StatisticalReport, analyze_series
from .volume_profile import VolumeProfile, build_volume_profile

__all__ = [
    "AnalysisEngine",
    "AnalysisReport",
    "ConfluenceEngine",
    "ConfluenceReport",
    "CrossAssetReport",
    "CrossAssetTracker",
    "Dimension",
    "MicrostructureReport",
    "MicrostructureTracker",
    "OptionsAnalyticsReport",
    "StatisticalReport",
    "VolumeProfile",
    "analyze_chain",
    "analyze_series",
    "build_volume_profile",
]
