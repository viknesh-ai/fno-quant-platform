# AQTP Architecture

> Maintained per REQ 67.3. Update this document whenever a layer boundary changes.

## 1. Governing principle

The platform is a **quantitative research and execution system**, not a trading
script. Its six intelligence layers are kept independent, and the only permitted
data flow between them is one direction:

```
Market Data → Feature Engine → Deep Analysis → Market Regime → Strategy Analysis
   → ML Prediction → Signal Aggregation → Opportunity Ranking → Risk Engine
   → Position Sizing → Execution Engine → Broker API
```

Two invariants are enforced *structurally* rather than by convention:

| Invariant | How it is enforced |
|---|---|
| A strategy can never place an order (REQ 67.23) | `StrategyContext` is a frozen dataclass with no broker handle. `StrategySignal` has no `quantity` or `order_type` field. Asserted by `test_pipeline.py::TestStrategyCannotReachTheBroker`. |
| The RiskEngine cannot be bypassed (REQ 67.22) | `ExecutionEngine.execute()` takes a `RiskApproval` as a **required** argument and refuses any approval where `approved is False`. |
| PAPER/BACKTEST cannot submit real orders (REQ 41) | `brokers/factory.py::build_execution_broker` returns a `SimulatedBroker` based on mode alone — no config switch can override it. |
| Analysis cannot create a trade | `SignalEnsemble.combine` consults the analysis *after* the agreement, regime and ML gates have already passed. It may veto, refuse a direction, or scale a score within `analysis.influence`; there is no path by which it raises a rejected signal. Asserted by `test_analysis_gate.py::TestAnalysisCannotCreateTrades`. |
| Manual orders cannot bypass risk | `execution/manual.py::ManualTrader` calls the same `RiskEngine.evaluate` and `ExecutionEngine.execute`. `--force` replaces the *sizing* only; kill switches and the exchange freeze limit still apply. Asserted by `test_manual_trading.py`. |
| Look-ahead bias (REQ 19) | Features read only `MultiTimeframeStore.closed()`; higher timeframes are joined with `merge_asof` shifted by the bar length; labels drop the trailing horizon; splits carry an embargo. |

## 2. Module map

```
src/aqtp/
├── core/              vocabulary shared by every layer; imports nothing else
│   ├── types.py       Instrument, Quote, Order, Position, enums
│   ├── clock.py       Clock protocol — why backtest and live share code paths
│   ├── errors.py      retryable vs non-retryable distinction
│   ├── ids.py         deterministic idempotency keys
│   └── logging.py     structured logging + credential redaction
│
├── configuration/     typed config; LIVE interlocks live here
├── brokers/           BrokerAdapter ABC │ GrowwAdapter │ SimulatedBroker │ factory
├── data/              market data engine, quality gate, candles, instrument discovery
├── features/          causal indicators, multi-timeframe FeatureEngine, option features
├── analysis/          eight independent market reads + confluence
│   ├── statistics.py       Hurst, variance ratio, entropy, GARCH, jump detection
│   ├── volume_profile.py   POC, value area, HVN/LVN, initial balance, VWAP bands
│   ├── microstructure.py   book imbalance, micro-price, OFI, toxicity, tape delta
│   ├── options_analytics.py dealer gamma, flip level, walls, max pain, skew, vanna/charm
│   ├── crossasset.py       correlation, breadth, dispersion, relative strength, VIX
│   ├── confluence.py       combines dimensions; conflict and vetoes are explicit
│   └── engine.py           runs all of it for one underlying
├── regime/            probabilistic MarketRegimeEngine + regime/strategy fit table
├── strategies/        Strategy ABC, 9 families, registry, health monitor
├── options/           Black-Scholes pricing/greeks/IV, contract selector
├── ml/                labeling, leakage-safe datasets, models, walk-forward trainer,
│                      registry, drift detection, prediction engine
├── signals/           SignalEnsemble, expected value
├── scanner/           OpportunityScanner (ranking)
├── risk/              RiskEngine, sizing, portfolio greeks, drawdown, kill switches, costs
├── execution/         ExecutionEngine (13 steps), position manager, exits, reconciliation
│   └── manual.py      operator-initiated orders, through the same risk pipeline
├── events/            EventRisk interface + file-backed calendar
├── journal/           SQLite trade journal + explainability
├── backtest/          event-driven engine, metrics, walk-forward, robustness
├── monitoring/        dashboard, alerts
├── runtime/           process supervisor: signals, reconnect, watchdog, square-off
├── research/          experiment versioning
├── cli/               trading commands + `trade` (manual orders) + research commands
└── orchestrator.py    sequences the pipeline; contains no trading logic itself
```

## 3. Layer contracts

### Market data (REQ 7/8)
`MarketDataEngine` is the **only** component that calls the broker for prices.
Every record passes `DataQualityGate` before storage; a failure produces a recorded
rejection reason, never an exception into the trading loop. If data is unreliable
the result is `NO_TRADE`.

### Features (REQ 10)
One `FeatureEngine` serves both training and live inference — deliberately, so a
model is never scored on a distribution it did not see. `FEATURE_VERSION` is
recorded with every model; `ModelRegistry.load()` refuses a version mismatch.

**Warm-up cost:** the 200-period EMA on the entry timeframe means roughly 205
closed bars are needed before any signal. On 5-minute bars that is ~17 sessions of
1-minute history, and the multi-timeframe join pushes the usable dataset start
later still. This is expected, not a defect.

### Deep analysis
Eight dimensions, each measuring something the others do not, each returning a
signed score in [-1, 1] with a confidence in [0, 1] and its own reasoning. The
combination in `confluence.py` is deliberately *not* a weighted mean:

* **Coverage and data quality scale the conviction**, so a reading built from
  three dimensions on thin data cannot outscore one built from eight on good data.
* **Conflict is penalised, not cancelled.** Dimensions pulling opposite ways
  produce a low-conviction report, never a confident neutral one.
* **Vetoes are returned separately from the score**, because "the book cannot
  fill this" and "the evidence is weak" are different statements that call for
  different responses.

Two dimensions — `statistical` and `volatility` — are not directional on their
own. They are scored against a *reference drift* derived from price structure, so
they read as "does this support the move already in progress", and mean-reverting
statistics correctly score *against* a momentum entry.

The layer's authority is bounded in config: `analysis.influence` caps how far it
can move a score, `analysis.min_conviction` sets the floor below which it blocks,
and `analysis.veto_enabled` / `require_direction_agreement` can each be switched
off to make it purely advisory.

### Regime (REQ 11)
Output is a probability distribution plus entropy, not a label. Low confidence or
high entropy collapses to `UNCERTAIN`, which blocks trading. Strategy compatibility
is weighted across the whole distribution, so a marginal regime call does not fully
switch a strategy off.

### Strategies (REQ 12)
Nine independent families. Each returns a `StrategySignal` carrying direction,
confidence, expected move, entry/stop/target, holding period, regime fit and
reasoning. A strategy never raises into the loop — bad input yields a FLAT signal
with the reason recorded.

Mean reversion is locked out of trending regimes **twice** (regime-fit table and an
independent in-strategy ADX/slope guard), because fading a real trend is the fastest
way to lose money here.

### Ensemble (REQ 13/54)
Correlated strategies are combined sub-additively: within a correlation family the
strongest contribution counts fully and each additional one is multiplied by
`correlation_discount`. Confidence and probability are separate quantities.

### ML (REQ 14–21)
Long and short are **separate models on separate labels** — "target before stop for
a long" is not the complement of the same question for a short. Every model is
calibrated on a held-out validation fold, never on training predictions. Selection
is on walk-forward out-of-sample expectancy, never accuracy.

### Risk (REQ 25)
`RiskEngine.evaluate()` runs **every** control and returns the full checklist, pass
or fail, even after the first failure — so an operator can see whether one limit or
five were breached. Limits are checked against the *projected* portfolio (post-fill),
not the current one.

### Execution (REQ 32/45)
Thirteen steps, with the requirement's central rule enforced structurally: the
engine records an order **intent** before submitting, submits without retry, then
**queries the broker** to learn what actually happened. An ambiguous submit raises
`OrderStateAmbiguous` and is resolved by reference lookup — never by resubmitting.

### Manual trading
`execution/manual.py` resolves a symbol specification (underlying + type + strike
+ expiry, or an exact trading symbol) to an instrument, then runs it through the
identical path an automated signal takes: expected value, `RiskEngine.evaluate`,
`ExecutionEngine.execute`, journal, `PositionManager`. A manual position has a
stop and a target and is squared off with everything else.

An explicit `--lots`/`--quantity` is honoured only if it is no larger than the
sizer allows; a larger request is refused rather than silently trimmed. `--force`
substitutes an operator-supplied size for the sizer's and is journalled as a
forced order. It does not touch kill switches or the exchange freeze limit.

### Runtime supervision
`runtime/supervisor.py` wraps the orchestrator loop with the five things a live
session needs that a bare loop does not have: signal handling that stops at a
cycle boundary, broker reconnection with exponential backoff followed by a
re-reconciliation, a watchdog on cycle completion, adaptive loop pacing inside
and outside the session, and an enforced square-off. Every one of them writes to
the journal, so a post-mortem does not depend on the log file surviving.

A process that exits holding positions raises a CRITICAL alert and says so on
stdout. That is the one state this system will not fail quietly in.

## 4. Modes

| Mode | Market data | Execution | Guard |
|---|---|---|---|
| BACKTEST | historical | `SimulatedBroker` | — |
| PAPER | **real** | `SimulatedBroker` | mode-based routing |
| DEMO | real | broker demo env | CLI confirmation |
| **LIVE (default)** | real | **real broker** | 3 env interlocks + typed phrase + banner |

`config/default.yaml` ships with `mode: LIVE` — this is a production system and
the configuration says so. That is not the same as LIVE being easy to enter: all
three of `AQTP_LIVE_CONFIRM_1/2/3` must still be set to their exact documented
values or `load_config` raises, and the CLI additionally requires the operator to
type `I ACCEPT THE RISK`. The `--yes` flag skips the typed phrase for systemd and
cron; it cannot substitute for the interlocks.

The split is deliberate. The config declares *intent* — what this deployment is
for. The environment declares *consent* — that a human decided to run it today.

## 5. Failure handling

| Failure | Response |
|---|---|
| Timeout on order submit | `OrderStateAmbiguous` → reference lookup → halt entries if unresolved |
| Duplicate client order id | Refused locally by `IdempotencyGuard` before any network call |
| Broker unreachable at reconciliation | `safe_to_trade = False`; entries blocked |
| Crash between intent and ack | Startup reconciliation resolves the intent from the broker |
| Stale/impossible/frozen market data | `DataQualityGate` rejects; symbol circuit-breaks after N failures |
| Drawdown level 4 | Emergency shutdown; persists across restart |
| Broker session drops mid-session | Supervisor re-authenticates with backoff, then re-reconciles; entries stay blocked until the reconciliation is clean |
| A cycle wedges | Watchdog trips after `runtime.watchdog_timeout_seconds` → alert / halt / flatten |
| Repeated cycle exceptions | Halts after `runtime.max_consecutive_cycle_errors` rather than trading blind |
| SIGTERM (systemd, `docker stop`) | Stops at a cycle boundary, cancels resting orders, optionally squares off |
| Toxic order flow / post-jump features | Analysis veto — the trade is refused, not resized |

## 6. Known limitations

These are real and deliberate, not oversights:

1. **Short-option margin is not modelled in simulation.** `SimulatedBroker.get_required_margin`
   approximates premium outlay only. SPAN/exposure for short options needs exchange
   risk arrays, so short-premium strategies must be sized against the live broker.
2. **Transaction cost rates are defaults, not verified constants.** They follow
   published NSE/SEBI/Groww schedules but change by circular. Re-verify against a
   current contract note before LIVE.
3. **No streaming feed.** The loop polls; Groww's documented REST endpoints are the
   only data source used. The architecture is event-driven enough to accept a feed,
   but none is invented. This bounds the microstructure dimension in particular:
   order-flow imbalance is computed from successive quote *snapshots*, not from a
   tick feed, and trade side is inferred by the tick rule rather than observed.
   The measurements are real; their resolution is limited by the data source.
4. **The correlation/sector map is a starting point**, not an exhaustive
   classification of the F&O universe. The cross-asset dimension computes live
   correlations to compensate, but only over symbols the scanner has actually
   sampled this session.
5. **Dealer positioning rests on a stated assumption.** Gamma exposure is signed
   on the convention that dealers are long call gamma and short put gamma. That is
   the standard convention and it is usually right, but it is an assumption about
   who is on the other side, not a measurement. `analysis/options_analytics.py`
   states it in its module docstring rather than burying it.
6. **Max pain is a measurement, not a forecast.** It is only meaningful near
   expiry, which is why `pin_risk` is zero more than two days out.
7. **No strategy in this repository has demonstrated an edge.** The reference
   EMA-cross source in the CLI exists to exercise the engine, and its own robustness
   report marks it NOT ROBUST. See `docs/TRACEABILITY.md` §68.
