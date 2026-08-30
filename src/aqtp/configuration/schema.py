"""Typed, validated configuration (REQ 58).

Design rules encoded here:
  * Nothing about capital, instruments, strategies, expiries or lot sizes is
    hard-coded — every one of them is a field with a conservative default.
  * `TradingMode.LIVE` is the default — this is a production trading system.
    LIVE still requires three environment interlocks (REQ 43) so that a stray
    `python -c` cannot start a session, but LIVE is the mode the platform is
    built and tuned for. PAPER remains available for dry runs.
  * Validation is total: the bot refuses to start on an inconsistent config rather
    than silently correcting it.
"""

from __future__ import annotations

import os
from datetime import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.types import Product, TradingMode

# The three interlocks that must all be present, with these exact values, before
# LIVE execution is permitted.
LIVE_INTERLOCKS: dict[str, str] = {
    "AQTP_LIVE_CONFIRM_1": "I_UNDERSTAND_REAL_MONEY",
    "AQTP_LIVE_CONFIRM_2": "I_HAVE_REVIEWED_RISK_LIMITS",
    "AQTP_LIVE_CONFIRM_3": "ENABLE_LIVE_TRADING",
}


class StrictModel(BaseModel):
    model_config = {"extra": "forbid", "validate_assignment": True}


# --------------------------------------------------------------------------- #
class BrokerConfig(StrictModel):
    name: Literal["groww", "simulated"] = "groww"
    auth_mode: Literal["token", "approval", "totp"] = "token"
    base_url: str = "https://api.groww.in"
    instruments_url: str = "https://growwapi-assets.groww.in/instruments/instrument.csv"
    api_version: str = "1.0"
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5
    # Documented Groww limits, expressed per category (requests/second).
    rate_limit_orders_per_second: float = 10.0
    rate_limit_live_data_per_second: float = 10.0
    rate_limit_non_trading_per_second: float = 20.0
    rate_limit_auth_per_second: float = 5.0


class CapitalConfig(StrictModel):
    """REQ 23 — capital is user-supplied and cross-checked against the broker."""

    available_capital: float = Field(gt=0, description="Capital the user allocates to this bot")
    max_capital_deployment_pct: float = Field(0.60, gt=0, le=1.0)
    use_broker_equity: bool = True
    # If the broker reports less than this fraction of the configured capital,
    # refuse to start rather than trading on a stale assumption.
    broker_equity_tolerance_pct: float = Field(0.20, ge=0, le=1.0)


class RiskConfig(StrictModel):
    """REQ 25 — every control is configurable and enforced by the RiskEngine."""

    risk_per_trade_pct: float = Field(0.01, gt=0, le=0.1)
    max_daily_loss_pct: float = Field(0.03, gt=0, le=1.0)
    max_portfolio_drawdown_pct: float = Field(0.10, gt=0, le=1.0)
    max_open_positions: int = Field(4, ge=1)
    max_exposure_per_instrument_pct: float = Field(0.20, gt=0, le=1.0)
    max_exposure_per_underlying_pct: float = Field(0.30, gt=0, le=1.0)
    max_correlated_exposure_pct: float = Field(0.40, gt=0, le=1.0)
    max_sector_exposure_pct: float = Field(0.40, gt=0, le=1.0)
    max_margin_utilization_pct: float = Field(0.70, gt=0, le=1.0)
    max_position_size_pct: float = Field(0.25, gt=0, le=1.0)
    max_spread_pct: float = Field(0.015, gt=0, le=1.0)
    max_slippage_pct: float = Field(0.005, gt=0, le=1.0)
    min_liquidity_volume: int = Field(5_000, ge=0)
    min_open_interest: int = Field(1_000, ge=0)
    max_consecutive_losses: int = Field(4, ge=1)
    max_trades_per_day: int = Field(20, ge=1)
    cooldown_minutes_after_loss: int = Field(10, ge=0)
    correlation_threshold: float = Field(0.7, ge=0, le=1.0)

    # Aggregate option-greek caps, expressed per lakh of equity (REQ 29/30).
    max_portfolio_delta_per_lakh: float = Field(150.0, gt=0)
    max_portfolio_gamma_per_lakh: float = Field(10.0, gt=0)
    max_portfolio_vega_per_lakh: float = Field(500.0, gt=0)
    max_portfolio_theta_per_lakh: float = Field(2_000.0, gt=0)

    # Drawdown ladder (REQ 26): fractions of peak equity.
    drawdown_level2_pct: float = Field(0.04, gt=0, le=1.0)
    drawdown_level3_pct: float = Field(0.07, gt=0, le=1.0)
    drawdown_level4_pct: float = Field(0.10, gt=0, le=1.0)
    level2_risk_multiplier: float = Field(0.5, gt=0, le=1.0)

    @model_validator(mode="after")
    def _ladder_is_monotonic(self) -> "RiskConfig":
        if not (self.drawdown_level2_pct < self.drawdown_level3_pct < self.drawdown_level4_pct):
            raise ValueError(
                "drawdown levels must increase: level2 < level3 < level4 "
                f"(got {self.drawdown_level2_pct}, {self.drawdown_level3_pct}, {self.drawdown_level4_pct})"
            )
        if self.max_portfolio_drawdown_pct < self.drawdown_level4_pct:
            raise ValueError(
                "max_portfolio_drawdown_pct must be >= drawdown_level4_pct"
            )
        if self.risk_per_trade_pct > self.max_daily_loss_pct:
            raise ValueError("risk_per_trade_pct cannot exceed max_daily_loss_pct")
        return self


class UniverseConfig(StrictModel):
    """REQ 4 — the universe is discovered, not hard-coded. These are filters."""

    index_underlyings: list[str] = Field(
        default_factory=lambda: ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]
    )
    include_stock_fno: bool = True
    # Empty list = "every F&O stock the broker lists". A non-empty list narrows it.
    stock_underlyings_allowlist: list[str] = Field(default_factory=list)
    stock_underlyings_blocklist: list[str] = Field(default_factory=list)
    include_futures: bool = True
    include_options: bool = True
    max_universe_size: int = Field(120, ge=1)
    refresh_interval_minutes: int = Field(360, ge=1)
    # Contract eligibility
    max_expiries_ahead: int = Field(3, ge=1)
    min_days_to_expiry: int = Field(0, ge=0)
    max_days_to_expiry: int = Field(45, ge=1)


class TimeframeConfig(StrictModel):
    """REQ 9 — the regime/setup/entry mapping is configuration, not code."""

    regime_timeframe: str = "1h"
    setup_timeframe: str = "15m"
    entry_timeframe: str = "5m"
    enabled_timeframes: list[str] = Field(
        default_factory=lambda: ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "4h", "1d"]
    )
    history_bars: int = Field(400, ge=50)


class DataQualityConfig(StrictModel):
    """REQ 8 — thresholds beyond which data is refused."""

    max_quote_age_seconds: float = Field(15.0, gt=0)
    max_greeks_age_seconds: float = Field(60.0, gt=0)
    max_spread_pct: float = Field(0.05, gt=0, le=1.0)
    max_price_jump_pct: float = Field(0.20, gt=0, le=1.0)
    min_price: float = Field(0.05, gt=0)
    duplicate_tick_window_seconds: float = Field(0.0, ge=0)
    require_depth: bool = False
    max_consecutive_failures: int = Field(3, ge=1)


class StrategyConfig(StrictModel):
    """Per-strategy toggle plus a weight prior for the ensemble."""

    enabled: bool = True
    weight: float = Field(1.0, ge=0)
    params: dict = Field(default_factory=dict)


class EnsembleConfig(StrictModel):
    """REQ 13/54."""

    min_strategy_agreement: float = Field(0.5, ge=0, le=1.0)
    min_strategies_agreeing: int = Field(2, ge=1)
    min_ensemble_confidence: float = Field(0.55, ge=0, le=1.0)
    ml_weight: float = Field(0.5, ge=0, le=1.0)
    strategy_weight: float = Field(0.5, ge=0, le=1.0)
    # Strategies whose signals derive from the same underlying evidence get their
    # combined contribution shrunk to avoid double-counting (REQ 13).
    correlation_groups: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "trend_family": ["trend_following", "market_structure", "vwap"],
            "momentum_family": ["momentum", "breakout", "volatility_expansion"],
        }
    )
    correlation_discount: float = Field(0.6, gt=0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum(self) -> "EnsembleConfig":
        total = self.ml_weight + self.strategy_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ml_weight + strategy_weight must equal 1.0 (got {total})")
        return self


class PredictionConfig(StrictModel):
    """REQ 15 — targets are precisely specified, never 'will it go up'."""

    horizons_minutes: list[int] = Field(default_factory=lambda: [15, 30, 60])
    target_atr_multiple: float = Field(1.5, gt=0)
    stop_atr_multiple: float = Field(1.0, gt=0)
    min_movement_atr: float = Field(0.25, gt=0)
    volatility_adjusted: bool = True
    labeling: Literal["triple_barrier", "fixed_horizon"] = "triple_barrier"
    min_model_probability: float = Field(0.58, gt=0.5, le=1.0)
    max_prediction_age_seconds: float = Field(120.0, gt=0)


class ModelConfig(StrictModel):
    """REQ 16/17/20."""

    families: list[str] = Field(
        default_factory=lambda: ["logistic_regression", "random_forest", "gradient_boosting", "lightgbm"]
    )
    calibration: Literal["isotonic", "sigmoid", "none"] = "isotonic"
    walk_forward_train_days: int = Field(120, ge=10)
    walk_forward_validation_days: int = Field(20, ge=5)
    walk_forward_test_days: int = Field(20, ge=5)
    walk_forward_step_days: int = Field(20, ge=1)
    embargo_minutes: int = Field(60, ge=0)
    random_seed: int = 42
    min_training_samples: int = Field(500, ge=50)
    max_brier_score: float = Field(0.26, gt=0, le=1.0)
    min_roc_auc: float = Field(0.53, ge=0.0, le=1.0)
    regime_specific_models: bool = False
    drift_psi_threshold: float = Field(0.25, gt=0)
    drift_check_interval_minutes: int = Field(30, ge=1)


class OptionSelectionConfig(StrictModel):
    """REQ 6/30 — how candidate contracts are scored, all weights configurable."""

    strike_offsets: list[int] = Field(default_factory=lambda: [-2, -1, 0, 1, 2])
    consider_expiries: int = Field(2, ge=1)
    min_delta: float = Field(0.25, gt=0, lt=1.0)
    max_delta: float = Field(0.75, gt=0, le=1.0)
    # Must stay <= risk.max_spread_pct, else the selector proposes contracts the
    # RiskEngine is guaranteed to reject. Cross-checked in loader._post_validate.
    max_spread_pct: float = Field(0.012, gt=0, le=1.0)
    min_volume: int = Field(500, ge=0)
    min_open_interest: int = Field(1_000, ge=0)
    max_theta_to_premium_ratio: float = Field(0.06, gt=0)
    max_iv_percentile: float = Field(0.85, gt=0, le=1.0)
    min_expected_move_to_premium: float = Field(1.2, gt=0)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "liquidity": 0.20,
            "spread": 0.15,
            "greeks": 0.20,
            "moneyness": 0.10,
            "cost": 0.10,
            "risk_reward": 0.25,
        }
    )

    @model_validator(mode="after")
    def _weights_normalized(self) -> "OptionSelectionConfig":
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("option selection weights must sum to a positive value")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"option selection weights must sum to 1.0 (got {total:.4f})")
        return self


class ExecutionConfig(StrictModel):
    """REQ 31/32/33."""

    default_product: Product = Product.NRML
    entry_order_type: Literal["MARKET", "LIMIT"] = "LIMIT"
    limit_price_buffer_pct: float = Field(0.003, ge=0, le=0.05)
    max_acceptable_slippage_pct: float = Field(0.004, gt=0, le=0.05)
    order_fill_timeout_seconds: float = Field(20.0, gt=0)
    order_poll_interval_seconds: float = Field(1.0, gt=0)
    cancel_unfilled_after_timeout: bool = True
    verify_fill_before_recording: bool = True
    max_order_attempts: int = Field(2, ge=1)
    reconcile_interval_seconds: float = Field(60.0, gt=0)
    loop_interval_seconds: float = Field(5.0, gt=0)
    # REQ 33: don't trade merely because a tick arrived.
    min_feature_change_to_reevaluate: float = Field(0.15, ge=0)


class CostConfig(StrictModel):
    """REQ 56 — never assume zero costs. Values are user-verifiable inputs, not
    invented constants; defaults follow published NSE/Groww F&O schedules and must
    be re-checked against the current contract note."""

    brokerage_per_order: float = Field(20.0, ge=0)
    brokerage_pct_cap: float = Field(0.0005, ge=0)
    exchange_txn_charge_pct_futures: float = Field(0.0000173, ge=0)
    exchange_txn_charge_pct_options: float = Field(0.0003503, ge=0)
    stt_pct_futures_sell: float = Field(0.000125, ge=0)
    stt_pct_options_sell: float = Field(0.000625, ge=0)
    stamp_duty_pct_buy_futures: float = Field(0.00002, ge=0)
    stamp_duty_pct_buy_options: float = Field(0.00003, ge=0)
    sebi_charges_pct: float = Field(0.000001, ge=0)
    gst_pct: float = Field(0.18, ge=0)
    slippage_model: Literal["spread_fraction", "fixed_pct"] = "spread_fraction"
    slippage_spread_fraction: float = Field(0.5, ge=0, le=2.0)
    slippage_fixed_pct: float = Field(0.001, ge=0)


class ExpectedValueConfig(StrictModel):
    """REQ 55."""

    min_expected_value_rupees: float = Field(0.0)
    min_expected_value_r_multiple: float = Field(0.15, ge=0)
    min_risk_reward: float = Field(1.4, gt=0)


class SessionConfig(StrictModel):
    trading_start: time = time(9, 20)
    trading_end: time = time(15, 10)
    no_new_entries_after: time = time(14, 45)
    square_off_time: time = time(15, 15)
    avoid_first_minutes: int = Field(5, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "SessionConfig":
        if not (self.trading_start < self.no_new_entries_after <= self.trading_end <= self.square_off_time):
            raise ValueError(
                "session times must satisfy: start < no_new_entries_after <= end <= square_off"
            )
        return self


class EventRiskConfig(StrictModel):
    """REQ 37 — event data arrives through a controlled source, never scraping."""

    enabled: bool = True
    source: Literal["file", "none"] = "file"
    calendar_path: str = "config/event_calendar.yaml"
    blackout_minutes_before: int = Field(30, ge=0)
    blackout_minutes_after: int = Field(15, ge=0)
    high_impact_action: Literal["block_new", "reduce_risk", "close_positions", "continue"] = "block_new"
    reduced_risk_multiplier: float = Field(0.5, gt=0, le=1.0)


class StrategyHealthConfig(StrictModel):
    """REQ 27/28/53."""

    lookback_trades: int = Field(30, ge=5)
    min_trades_for_judgement: int = Field(12, ge=5)
    reduce_below_expectancy_r: float = -0.05
    pause_below_expectancy_r: float = -0.20
    disable_below_expectancy_r: float = -0.40
    reduced_multiplier: float = Field(0.5, gt=0, le=1.0)
    pause_duration_minutes: int = Field(120, ge=1)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> "StrategyHealthConfig":
        if not (
            self.disable_below_expectancy_r
            < self.pause_below_expectancy_r
            < self.reduce_below_expectancy_r
        ):
            raise ValueError("health thresholds must satisfy disable < pause < reduce")
        return self


class MonitoringConfig(StrictModel):
    dashboard_enabled: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = Field(8787, ge=1024, le=65535)
    alerts_enabled: bool = True
    alert_channels: list[Literal["log", "file", "webhook"]] = Field(default_factory=lambda: ["log", "file"])
    alert_events: list[str] = Field(
        default_factory=lambda: [
            "trade_entry", "trade_exit", "rejected_trade", "risk_limit", "drawdown_limit",
            "broker_error", "api_disconnect", "strategy_disabled", "model_drift",
            "emergency_shutdown",
        ]
    )


class PathsConfig(StrictModel):
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    journal_db: str = "data/journal/journal.sqlite"
    model_dir: str = "data/models"
    experiment_dir: str = "data/experiments"
    log_dir: str = "data/logs"
    state_file: str = "data/state.json"


class AnalysisConfig(StrictModel):
    """Deep-analysis layer (`aqtp.analysis`).

    Every dimension can be switched off, and the weights say how much each one
    counts toward the combined conviction. They are normalised over whichever
    dimensions actually reported, so they need not sum to 1.
    """

    enabled: bool = True
    benchmark: str = "NIFTY"
    vix_symbol: str = "INDIAVIX"

    # --- dimension weights -------------------------------------------------
    dimension_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "trend": 0.16,
            "momentum": 0.14,
            "structure": 0.14,
            "orderflow": 0.16,
            "positioning": 0.14,
            "statistical": 0.10,
            "volatility": 0.08,
            "crossasset": 0.08,
        }
    )
    min_dimensions: int = Field(4, ge=1, le=8)
    conflict_penalty: float = Field(0.5, ge=0.0, le=1.0)

    # --- how the conviction is used ---------------------------------------
    min_conviction: float = Field(0.20, ge=0.0, le=1.0)
    # How much the analysis is allowed to move an ensemble score, as a fraction.
    # 0.0 makes the layer advisory only; 1.0 lets it halve or double a signal.
    influence: float = Field(0.50, ge=0.0, le=1.0)
    veto_enabled: bool = True
    require_direction_agreement: bool = True

    # --- sub-module settings ----------------------------------------------
    volume_profile_bins: int = Field(60, ge=10, le=200)
    value_area_pct: float = Field(0.70, gt=0.0, lt=1.0)
    microstructure_history: int = Field(120, ge=10, le=2000)
    statistical_lookback_bars: int = Field(240, ge=60)

    # --- veto thresholds ---------------------------------------------------
    max_spread_pct: float = Field(0.02, gt=0, le=1.0)
    max_jump_sigmas: float = Field(5.0, gt=0)
    min_tradability: float = Field(0.15, ge=0.0, le=1.0)
    max_pin_risk: float = Field(0.75, ge=0.0, le=1.0)
    max_quote_age_seconds: float = Field(20.0, gt=0)

    @model_validator(mode="after")
    def _weights_are_sane(self) -> "AnalysisConfig":
        known = {
            "trend", "momentum", "structure", "orderflow",
            "positioning", "statistical", "volatility", "crossasset",
        }
        unknown = set(self.dimension_weights) - known
        if unknown:
            raise ValueError(f"unknown analysis dimension(s): {sorted(unknown)}")
        if any(w < 0 for w in self.dimension_weights.values()):
            raise ValueError("analysis dimension weights cannot be negative")
        if sum(self.dimension_weights.values()) <= 0:
            raise ValueError("at least one analysis dimension must carry weight")
        return self


class RuntimeConfig(StrictModel):
    """Production process behaviour (REQ 41/63/64).

    These are the settings that decide whether the bot survives a bad afternoon:
    how fast it loops, how long it tolerates a dead broker session, and what it
    does to open positions when the process is told to stop.
    """

    # --- loop pacing --------------------------------------------------------
    fast_loop_seconds: float = Field(2.0, gt=0, le=60)
    idle_loop_seconds: float = Field(15.0, gt=0, le=300)
    # Positions are re-evaluated at the fast interval regardless of the scan
    # interval: a stop must not wait behind a forty-symbol scan.
    position_check_seconds: float = Field(2.0, gt=0, le=60)
    heartbeat_seconds: float = Field(30.0, gt=0, le=600)

    # --- resilience ---------------------------------------------------------
    max_consecutive_cycle_errors: int = Field(10, ge=1)
    reconnect_backoff_seconds: float = Field(2.0, gt=0)
    reconnect_max_backoff_seconds: float = Field(60.0, gt=0)
    max_reconnect_attempts: int = Field(20, ge=1)
    # If no cycle completes within this many seconds the watchdog trips.
    watchdog_timeout_seconds: float = Field(180.0, gt=0)
    watchdog_action: Literal["alert", "halt", "flatten"] = "halt"

    # --- shutdown -----------------------------------------------------------
    square_off_on_shutdown: bool = False
    cancel_open_orders_on_shutdown: bool = True
    shutdown_grace_seconds: float = Field(20.0, ge=0)

    # --- process ------------------------------------------------------------
    pid_file: str = "data/aqtp.pid"
    state_snapshot_seconds: float = Field(60.0, gt=0)
    auto_square_off_before_close: bool = True

    @model_validator(mode="after")
    def _pacing_is_consistent(self) -> "RuntimeConfig":
        if self.idle_loop_seconds < self.fast_loop_seconds:
            raise ValueError("idle_loop_seconds must be >= fast_loop_seconds")
        if self.reconnect_max_backoff_seconds < self.reconnect_backoff_seconds:
            raise ValueError("reconnect_max_backoff_seconds must be >= reconnect_backoff_seconds")
        return self


class ScannerConfig(StrictModel):
    """REQ 22."""

    min_opportunity_score: float = Field(0.60, ge=0, le=1.0)
    max_opportunities_per_cycle: int = Field(5, ge=1)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "strategy": 0.20,
            "ml": 0.20,
            "analysis": 0.20,
            "regime_fit": 0.12,
            "liquidity": 0.12,
            "expected_return": 0.08,
            "execution_quality": 0.08,
        }
    )

    @model_validator(mode="after")
    def _weights_normalized(self) -> "ScannerConfig":
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scanner weights must sum to 1.0 (got {total:.4f})")
        return self


class BacktestConfig(StrictModel):
    """REQ 38/40."""

    start_date: str | None = None
    end_date: str | None = None
    initial_capital: float = Field(500_000.0, gt=0)
    latency_ms: int = Field(250, ge=0)
    fill_model: Literal["next_bar_open", "current_close", "mid_plus_slippage"] = "next_bar_open"
    monte_carlo_runs: int = Field(500, ge=0)
    parameter_perturbation_pct: float = Field(0.15, ge=0, le=1.0)
    cost_sensitivity_multipliers: list[float] = Field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    slippage_sensitivity_multipliers: list[float] = Field(default_factory=lambda: [0.0, 1.0, 2.0, 3.0])
    entry_delay_bars: list[int] = Field(default_factory=lambda: [0, 1, 2])


# --------------------------------------------------------------------------- #
class AppConfig(StrictModel):
    """Root configuration object. One instance is threaded through the system."""

    mode: TradingMode = TradingMode.LIVE
    run_name: str = "default"
    log_level: str = "INFO"
    json_logs: bool = False

    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    capital: CapitalConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    timeframes: TimeframeConfig = Field(default_factory=TimeframeConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    strategies: dict[str, StrategyConfig] = Field(default_factory=dict)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    option_selection: OptionSelectionConfig = Field(default_factory=OptionSelectionConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    expected_value: ExpectedValueConfig = Field(default_factory=ExpectedValueConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    event_risk: EventRiskConfig = Field(default_factory=EventRiskConfig)
    strategy_health: StrategyHealthConfig = Field(default_factory=StrategyHealthConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, value: object) -> object:
        if isinstance(value, str):
            return TradingMode(value.strip().upper())
        return value

    @model_validator(mode="after")
    def _validate_timeframes(self) -> "AppConfig":
        from ..core.types import Timeframe

        valid = {tf.value for tf in Timeframe}
        for name, tf in (
            ("regime_timeframe", self.timeframes.regime_timeframe),
            ("setup_timeframe", self.timeframes.setup_timeframe),
            ("entry_timeframe", self.timeframes.entry_timeframe),
        ):
            if tf not in valid:
                raise ValueError(f"{name}={tf!r} is not a supported timeframe {sorted(valid)}")
            if tf not in self.timeframes.enabled_timeframes:
                raise ValueError(f"{name}={tf!r} must also appear in enabled_timeframes")

        minutes = {tf: Timeframe(tf).minutes for tf in valid}
        if not (
            minutes[self.timeframes.regime_timeframe]
            >= minutes[self.timeframes.setup_timeframe]
            >= minutes[self.timeframes.entry_timeframe]
        ):
            raise ValueError(
                "timeframe hierarchy must be regime >= setup >= entry "
                f"(got {self.timeframes.regime_timeframe}/{self.timeframes.setup_timeframe}/"
                f"{self.timeframes.entry_timeframe})"
            )
        return self

    @model_validator(mode="after")
    def _validate_universe(self) -> "AppConfig":
        if not self.universe.include_futures and not self.universe.include_options:
            raise ValueError("universe must include futures, options, or both")
        if not self.universe.index_underlyings and not self.universe.include_stock_fno:
            raise ValueError("universe would be empty: no index underlyings and stock F&O disabled")
        overlap = set(self.universe.stock_underlyings_allowlist) & set(
            self.universe.stock_underlyings_blocklist
        )
        if overlap:
            raise ValueError(f"symbols in both allowlist and blocklist: {sorted(overlap)}")
        if self.universe.min_days_to_expiry >= self.universe.max_days_to_expiry:
            raise ValueError("min_days_to_expiry must be < max_days_to_expiry")
        return self

    @model_validator(mode="after")
    def _validate_mode_interlocks(self) -> "AppConfig":
        """REQ 43 — LIVE requires three independent environment confirmations."""
        if self.mode is not TradingMode.LIVE:
            return self
        missing = [
            key for key, expected in LIVE_INTERLOCKS.items() if os.environ.get(key, "") != expected
        ]
        if missing:
            raise ValueError(
                "LIVE mode refused. The following environment interlocks are missing or "
                f"incorrect: {missing}. Set each to its documented value (see .env.example), "
                "or run `aqtp arm` to print them. LIVE is the configured default, but it "
                "cannot be entered from a config file alone."
            )
        if self.broker.name == "simulated":
            raise ValueError("LIVE mode cannot run against the simulated broker")
        return self

    @model_validator(mode="after")
    def _validate_broker_for_mode(self) -> "AppConfig":
        if self.mode in (TradingMode.PAPER, TradingMode.BACKTEST):
            # Paper may read real data through the Groww adapter, but execution is
            # forced to the simulator by the ExecutionEngine — see execution/engine.py.
            pass
        return self

    @property
    def is_live(self) -> bool:
        return self.mode is TradingMode.LIVE

    @property
    def submits_real_orders(self) -> bool:
        """The single predicate the execution layer consults before touching a
        broker order endpoint (REQ 41)."""
        return self.mode in (TradingMode.LIVE, TradingMode.DEMO)

    def mode_banner(self) -> str:
        """REQ 42/43 — the mode is displayed prominently at startup."""
        return f"TRADING MODE: {self.mode.value}"
