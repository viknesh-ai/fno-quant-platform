# AQTP Architecture

> Maintained per REQ 67.3. Update this document whenever a layer boundary changes.

## 1. Governing principle

The platform is a **quantitative research and execution system**, not a trading
script. Its five intelligence layers are kept independent, and the only permitted
data flow between them is one direction:

```
Market Data → Feature Engine → Market Regime → Strategy Analysis → ML Prediction
   → Signal Aggregation → Opportunity Ranking → Risk Engine → Position Sizing
   → Execution Engine → Broker API
```

Two invariants are enforced *structurally* rather than by convention:

| Invariant | How it is enforced |
|---|---|
| A strategy can never place an order (REQ 67.23) | `StrategyContext` is a frozen dataclass with no broker handle. `StrategySignal` has no `quantity` or `order_type` field. Asserted by `test_pipeline.py::TestStrategyCannotReachTheBroker`. |
| The RiskEngine cannot be bypassed (REQ 67.22) | `ExecutionEngine.execute()` takes a `RiskApproval` as a **required** argument and refuses any approval where `approved is False`. |
| PAPER/BACKTEST cannot submit real orders (REQ 41) | `brokers/factory.py::build_execution_broker` returns a `SimulatedBroker` based on mode alone — no config switch can override it. |
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
├── regime/            probabilistic MarketRegimeEngine + regime/strategy fit table
├── strategies/        Strategy ABC, 9 families, registry, health monitor
├── options/           Black-Scholes pricing/greeks/IV, contract selector
├── ml/                labeling, leakage-safe datasets, models, walk-forward trainer,
│                      registry, drift detection, prediction engine
├── signals/           SignalEnsemble, expected value
├── scanner/           OpportunityScanner (ranking)
├── risk/              RiskEngine, sizing, portfolio greeks, drawdown, kill switches, costs
├── execution/         ExecutionEngine (13 steps), position manager, exits, reconciliation
├── events/            EventRisk interface + file-backed calendar
├── journal/           SQLite trade journal + explainability
├── backtest/          event-driven engine, metrics, walk-forward, robustness
├── monitoring/        dashboard, alerts
├── research/          experiment versioning
├── cli/               trading commands + research commands
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

## 4. Modes

| Mode | Market data | Execution | Guard |
|---|---|---|---|
| BACKTEST | historical | `SimulatedBroker` | — |
| PAPER (default) | **real** | `SimulatedBroker` | mode-based routing |
| DEMO | real | broker demo env | CLI confirmation |
| LIVE | real | **real broker** | 3 env interlocks + typed phrase + banner |

LIVE cannot be enabled by a config file. All three of `AQTP_LIVE_CONFIRM_1/2/3`
must be set to their exact documented values, or `load_config` raises.

## 5. Failure handling

| Failure | Response |
|---|---|
| Timeout on order submit | `OrderStateAmbiguous` → reference lookup → halt entries if unresolved |
| Duplicate client order id | Refused locally by `IdempotencyGuard` before any network call |
| Broker unreachable at reconciliation | `safe_to_trade = False`; entries blocked |
| Crash between intent and ack | Startup reconciliation resolves the intent from the broker |
| Stale/impossible/frozen market data | `DataQualityGate` rejects; symbol circuit-breaks after N failures |
| Drawdown level 4 | Emergency shutdown; persists across restart |

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
   but none is invented.
4. **The correlation/sector map is a starting point**, not an exhaustive
   classification of the F&O universe.
5. **No strategy in this repository has demonstrated an edge.** The reference
   EMA-cross source in the CLI exists to exercise the engine, and its own robustness
   report marks it NOT ROBUST. See `docs/TRACEABILITY.md` §68.
