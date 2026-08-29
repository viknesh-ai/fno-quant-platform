# AI-Native Indian F&O Autonomous Trading Platform

## 1. Project Objective

Build a production-grade, AI-assisted, algorithmic trading platform for Indian exchange-traded markets.

The system must be capable of:

- Continuously monitoring eligible Indian F&O instruments.
- Monitoring NIFTY 50 and other index derivatives.
- Monitoring stock futures and stock options.
- Monitoring option chains and Greeks.
- Detecting market regimes.
- Generating multiple independent trading hypotheses.
- Using machine-learning models to estimate future outcomes probabilistically.
- Combining quantitative strategies and ML predictions.
- Selecting the most appropriate instrument and contract dynamically.
- Selecting appropriate option strikes and expiries dynamically.
- Calculating position size from user-configured capital and risk parameters.
- Automatically entering trades.
- Managing open positions.
- Exiting trades automatically.
- Continuously reassessing existing positions.
- Enforcing strict risk controls.
- Recording every decision and every trade.
- Backtesting strategies.
- Performing walk-forward validation.
- Performing out-of-sample evaluation.
- Paper trading.
- Demo/simulation trading.
- Live trading through the configured broker API.

The system must NOT be designed around one fixed strategy, one fixed indicator, one fixed instrument, one fixed option strike, or one fixed expiry.

The system must adapt to changing market conditions.

---

# 2. Core Design Philosophy

The platform must be treated as a quantitative research and execution system rather than a simple trading script.

The system has five major intelligence layers:

1. Market Understanding
2. Strategy Generation
3. Predictive Modeling
4. Risk Management
5. Execution

The architecture must keep these layers independent.

A strategy must never directly place a broker order.

The flow must always be:

Market Data
→ Feature Engine
→ Market Regime
→ Strategy Analysis
→ ML Prediction
→ Signal Aggregation
→ Opportunity Ranking
→ Risk Engine
→ Position Sizing
→ Execution Engine
→ Broker API

---

# 3. Broker Integration

Initial broker:

Groww Trading API.

The broker integration must be implemented as an abstraction.

Create:

BrokerAdapter
├── GrowwAdapter
└── FutureBrokerAdapter

The rest of the application must not depend directly on Groww-specific API code.

The Groww adapter must support, where available:

- Authentication
- Instrument lookup
- Market data
- LTP
- OHLC
- Market depth
- Option chain
- Greeks
- Order placement
- Order modification
- Order cancellation
- Order status
- Trade status
- Positions
- Holdings
- Margin
- Historical data

Use the official Groww API documentation as the source of truth for current endpoints, authentication, limits and request schemas.

Never invent undocumented endpoints.

---

# 4. Market Universe

The platform must support dynamically discovering eligible instruments.

Initial universe:

## Index derivatives

- NIFTY
- BANKNIFTY
- FINNIFTY
- MIDCPNIFTY
- NIFTYNXT50
- Any additional exchange-supported index derivatives available through the broker

## Stock derivatives

The system must be capable of scanning all currently eligible F&O stocks available through the broker.

Do not hard-code a static list.

Instrument discovery must be refreshed according to a configurable schedule.

The system must automatically account for:

- Newly introduced contracts
- Expired contracts
- Expiry changes
- Lot-size changes
- Strike changes
- Trading-status changes
- Illiquid contracts
- Suspended instruments

---

# 5. Instruments

Support:

- Index futures
- Index calls
- Index puts
- Stock futures
- Stock calls
- Stock puts

The architecture must allow future support for:

- Currency derivatives
- Commodity derivatives
- ETFs
- Other exchange-supported derivatives

Do not implement unsupported products merely by assuming their API format.

---

# 6. Dynamic Contract Selection

The system must NOT permanently trade:

"nearest expiry ATM CE"

or:

"one fixed strike."

Instead, when the underlying presents an opportunity, evaluate available contracts.

Candidate contracts may include:

- ATM
- Slightly ITM
- Slightly OTM
- Nearby strikes
- Different expiries where appropriate

Score each candidate based on:

- Liquidity
- Bid/ask spread
- Volume
- Open interest
- Change in open interest
- Implied volatility
- Delta
- Gamma
- Theta
- Vega
- Moneyness
- Distance from underlying
- Time to expiry
- Expected movement
- Estimated slippage
- Capital requirement
- Risk/reward
- Strategy compatibility

The system must select the contract with the highest expected risk-adjusted opportunity rather than simply selecting the nearest strike.

---

# 7. Market Data Engine

Create a normalized market-data layer.

Required data categories:

## Underlying

- LTP
- OHLC
- returns
- volatility
- volume where available
- market depth where available
- VWAP where available

## Futures

- futures price
- spot/futures spread
- basis
- volume
- open interest
- OI change

## Options

- LTP
- bid
- ask
- spread
- volume
- open interest
- OI change
- IV
- delta
- gamma
- theta
- vega
- rho
- intrinsic value
- time value
- moneyness
- expiry

## Market Context

- previous day high
- previous day low
- previous close
- opening price
- session high
- session low
- gap
- intraday range
- volatility regime
- market breadth where available

---

# 8. Data Quality Layer

Every incoming market-data record must be validated.

Detect:

- stale prices
- missing prices
- timestamp anomalies
- duplicate ticks
- impossible values
- abnormal spreads
- missing option-chain contracts
- stale Greeks
- broker/API failures

Invalid data must never reach the strategy engine.

If market data becomes unreliable:

NO_TRADE.

---

# 9. Multi-Timeframe Engine

Support configurable timeframes.

Minimum:

- Tick / live updates where available
- 1 minute
- 3 minute
- 5 minute
- 10 minute
- 15 minute
- 30 minute
- 1 hour
- 4 hour
- Daily

Do not hard-code one timeframe.

Different strategies may operate on different timeframes.

Example architecture:

Higher timeframe:
Market regime

Middle timeframe:
Setup

Lower timeframe:
Entry

The exact mapping must be configurable and empirically validated.

---

# 10. Feature Engineering Engine

Create a reusable feature engine.

Features must include several categories.

## Price Features

- returns
- log returns
- momentum
- acceleration
- price distance from moving averages
- candle body
- candle range
- upper wick
- lower wick
- gap
- rolling highs
- rolling lows

## Trend Features

- EMA
- SMA
- ADX
- directional movement
- trend slope
- market structure

## Momentum

- RSI
- MACD
- ROC
- stochastic
- momentum acceleration

## Volatility

- ATR
- realized volatility
- rolling volatility
- volatility percentile
- volatility expansion
- volatility compression

## Volume

- volume change
- relative volume
- volume acceleration
- volume percentile

## VWAP

- price/VWAP distance
- VWAP slope
- VWAP deviation
- session VWAP

## Options

- IV
- IV percentile
- IV change
- delta
- gamma
- theta
- vega
- OI
- OI change
- volume/OI relationship
- put/call relationships
- strike-wise positioning

## Time Features

- time of day
- minutes since market open
- minutes until close
- day of week
- expiry proximity
- expiry day
- monthly/weekly expiry state

---

# 11. Market Regime Detection

Build a dedicated MarketRegimeEngine.

Supported regimes:

- TRENDING_UP
- TRENDING_DOWN
- RANGE
- BREAKOUT
- HIGH_VOLATILITY
- LOW_VOLATILITY
- VOLATILITY_EXPANSION
- VOLATILITY_COMPRESSION
- EXPIRY_DRIVEN
- EVENT_DRIVEN
- UNCERTAIN

The regime engine may use:

- deterministic rules
- statistical models
- machine learning
- hidden-state models
- clustering
- volatility classification

The regime output must contain probabilities.

Example conceptual output:

regime:
TRENDING_UP: 0.74
RANGE: 0.08
BREAKOUT: 0.13
UNCERTAIN: 0.05

Do not trade when regime uncertainty is too high.

---

# 12. Strategy Engine

Implement independent, pluggable strategies.

Minimum strategy families:

## 12.1 Trend Following

Use combinations of:

- EMA structure
- ADX
- trend slope
- market structure
- VWAP
- higher-high/lower-low analysis

The strategy must work for both bullish and bearish markets.

---

## 12.2 Momentum

Features may include:

- RSI
- MACD
- ROC
- price acceleration
- volume acceleration
- breakout momentum

---

## 12.3 Breakout

Detect:

- opening range breakout
- previous day high/low
- consolidation breakout
- volatility breakout
- session breakout
- range expansion

Confirm with:

- volume
- volatility
- momentum
- market structure

---

## 12.4 Mean Reversion

Use:

- VWAP deviation
- statistical deviation
- RSI extremes
- volatility
- range regime

Mean reversion must automatically disable itself during strong directional trends.

---

## 12.5 Volatility Expansion

Detect:

- compression
- ATR expansion
- volatility percentile changes
- range expansion
- sudden volume expansion

---

## 12.6 Market Structure

Detect:

- swing highs
- swing lows
- higher highs
- higher lows
- lower highs
- lower lows
- break of structure
- change of character
- rejection
- failed breakout
- liquidity sweep patterns

Do not treat discretionary terminology as truth.

Every market-structure concept must be converted into explicit measurable rules.

---

## 12.7 VWAP Strategy

Analyze:

- VWAP trend
- distance from VWAP
- VWAP reclaim
- VWAP rejection
- VWAP breakout
- VWAP mean reversion

---

## 12.8 Options Flow / Positioning Strategy

Analyze:

- OI concentration
- OI change
- volume
- put/call positioning
- strike-wise changes
- IV changes
- unusual volume
- changes in option positioning

Do not assume OI alone predicts direction.

---

## 12.9 Expiry Strategy

Create a dedicated expiry regime.

Consider:

- time to expiry
- gamma
- theta
- IV
- option liquidity
- underlying movement
- rapid premium decay
- volatility expansion

The system must be able to decide:

TRADE EXPIRY
or
AVOID EXPIRY

based on measured conditions.

---

# 13. Strategy Ensemble

Never rely on one strategy.

Each strategy produces:

- direction
- confidence
- expected move
- proposed entry
- proposed invalidation
- proposed target
- expected holding period
- regime compatibility
- reasoning

The ensemble engine combines strategies.

Example conceptual output:

Trend:
BUY 0.82

Momentum:
BUY 0.77

Breakout:
BUY 0.69

Mean Reversion:
SELL 0.31

Options positioning:
BUY 0.74

Regime:
TRENDING_UP 0.81

The system should combine independent evidence while avoiding double-counting correlated indicators.

---

# 14. AI Prediction Engine

This is a core component.

Do not use an LLM as the primary real-time predictor.

Build dedicated machine-learning prediction models.

The model must make measurable probabilistic predictions.

Possible targets:

## Directional Prediction

Probability that underlying price moves upward/downward within a defined horizon.

## Barrier Prediction

Probability that:

TARGET

is reached before:

STOP

within a specified future window.

## Return Prediction

Expected future return distribution.

## Volatility Prediction

Expected future volatility.

## Trade Quality Prediction

Probability that a generated strategy signal will result in a favorable risk-adjusted outcome.

## Option Premium Prediction

Probability distribution of future option-premium movement.

---

# 15. Prediction Target Design

Never define the target simply as:

"Will NIFTY go up?"

Use precise targets.

The prediction configuration must define:

- prediction horizon
- target threshold
- stop threshold
- labeling methodology
- minimum movement
- volatility adjustment

The prediction system must support multiple horizons.

Examples of horizons:

- very short-term
- intraday
- multi-hour

Do not hard-code exact horizons.

They must be configurable and tested.

---

# 16. ML Model Architecture

The platform must support multiple model families.

Start with strong tabular models before introducing unnecessarily complex deep learning.

Candidate models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost/LightGBM where permitted
- CatBoost where appropriate
- temporal neural networks
- sequence models

The architecture must allow model comparison.

Models must be evaluated using:

- Log Loss
- Brier Score
- calibration
- precision
- recall
- ROC-AUC where meaningful
- PR-AUC where meaningful
- expected trading value
- drawdown impact

Do not optimize solely for classification accuracy.

---

# 17. Model Ensemble

Support multiple predictive models.

Example:

Direction Model
Volatility Model
Regime Model
Trade Quality Model

Then combine outputs into a meta-model or calibrated probability layer.

The final output must be:

- probability
- confidence
- expected movement
- uncertainty

---

# 18. AI Must Not Hallucinate Market Data

The ML model must only consume validated numerical/time-series features.

No fabricated data.

No missing-data assumptions without explicit handling.

No future information may leak into training features.

---

# 19. Avoid Data Leakage

This is mandatory.

The system must ensure:

- no future candles in features
- no future option-chain state
- no future IV
- no future OI
- no future price
- no look-ahead bias
- proper chronological splitting

All transformations must be fit only on training data where appropriate.

---

# 20. Walk-Forward Training

Implement walk-forward validation.

Example conceptual pipeline:

Training Window
→ Validation Window
→ Test Window

Then move forward.

Repeat across multiple historical periods.

Store performance by period.

Do not report one aggregate backtest as sufficient evidence.

---

# 21. Regime-Specific Models

Where statistically justified, support separate models for:

- trending
- ranging
- high volatility
- low volatility
- expiry
- event-driven periods

Do not create separate models merely because it sounds sophisticated.

Only retain them if out-of-sample performance demonstrates value.

---

# 22. Opportunity Scanner

The system must continuously scan the available F&O universe.

For each eligible instrument:

Calculate:

- liquidity score
- spread score
- volatility score
- strategy score
- ML probability
- expected return
- expected risk
- regime compatibility
- execution quality
- capital requirement

Then rank opportunities.

Conceptual output:

Instrument A
Score: 0.84

Instrument B
Score: 0.77

Instrument C
Score: 0.41

The system should trade only opportunities above configurable thresholds.

---

# 23. Capital Management

Do not hard-code account size.

The user must be able to specify:

- available capital
- maximum capital allocation
- risk per trade
- maximum daily loss
- maximum portfolio loss
- maximum position size
- maximum number of simultaneous positions

The bot must retrieve actual account equity and available margin from the broker where possible.

User configuration overrides must be validated.

Never assume a particular account balance.

---

# 24. Position Sizing

Position size must be dynamically calculated.

Inputs:

- account equity
- configured risk percentage
- stop distance
- contract lot size
- option premium
- margin
- tick size
- tick value
- liquidity
- broker constraints

Position sizing must respect:

- lot size
- quantity freeze
- margin availability
- maximum exposure
- portfolio risk

Never use arbitrary fixed quantities.

---

# 25. Risk Engine

The RiskEngine is the final authority before execution.

Every proposed trade must pass all risk checks.

Minimum controls:

- maximum risk per trade
- maximum daily loss
- maximum portfolio drawdown
- maximum open positions
- maximum exposure per instrument
- maximum exposure per underlying
- maximum correlated exposure
- maximum sector exposure
- maximum margin utilization
- maximum spread
- maximum slippage
- minimum liquidity
- maximum position size
- maximum consecutive losses
- maximum trades per period
- cooldown after losses
- emergency shutdown

If any control fails:

NO_TRADE.

---

# 26. Drawdown Protection

Implement multiple levels.

LEVEL 1:
Normal operation.

LEVEL 2:
Reduced risk.

LEVEL 3:
Trading paused.

LEVEL 4:
Emergency shutdown.

Thresholds must be configurable.

Do not automatically increase risk after winning streaks.

---

# 27. Loss-Streak Protection

Track:

- consecutive losing trades
- rolling losses
- strategy-specific losses
- instrument-specific losses
- regime-specific losses

If a strategy begins underperforming:

reduce exposure or disable the strategy.

Do not blindly continue trading.

---

# 28. Strategy Health Monitoring

Every strategy receives a live health score.

Track:

- recent expectancy
- win rate
- average R
- drawdown
- performance by regime
- performance by time
- performance by instrument

A strategy may be:

ACTIVE
REDUCED
PAUSED
DISABLED

based on configurable statistical rules.

---

# 29. Portfolio-Level Intelligence

The bot must not evaluate trades independently.

Before entering a trade, calculate:

- current portfolio exposure
- correlation
- underlying overlap
- sector overlap
- directional exposure
- option Greeks
- aggregate delta
- aggregate gamma
- aggregate theta
- aggregate vega

Reject trades that create excessive concentration.

---

# 30. Option-Specific Risk

For options, calculate:

- delta exposure
- gamma exposure
- theta exposure
- vega exposure
- IV exposure
- expiry concentration

The system must understand that a high-probability directional prediction does not automatically imply a good option trade.

A contract may be rejected because:

- spread too wide
- theta too high
- IV too expensive
- insufficient liquidity
- poor risk/reward
- excessive gamma risk
- insufficient expected movement

---

# 31. Entry Engine

The EntryEngine determines:

- whether to enter
- direction
- instrument
- contract
- strike
- expiry
- order type
- entry price
- maximum acceptable price
- stop
- target
- position size

The system must support:

- market orders
- limit orders
- stop orders
- order modification
- cancellation

Use only order types supported by the broker/exchange for the selected instrument.

---

# 32. Execution Engine

The ExecutionEngine must:

1. Validate signal.
2. Validate risk.
3. Validate instrument.
4. Validate liquidity.
5. Validate market status.
6. Validate quantity.
7. Validate margin.
8. Create order.
9. Submit order.
10. Verify broker response.
11. Verify actual execution.
12. Reconcile local state.
13. Record the trade.

Never assume:

"API returned success = position exists."

Always verify actual order/trade state.

---

# 33. Fast Trading

The system must support short-interval trading.

The architecture must be event-driven rather than dependent on slow polling wherever the broker/API supports suitable real-time feeds.

However:

Do not trade merely because a new tick arrived.

The system should recompute relevant features and only generate a trade when conditions materially change.

Execution latency must be measured.

Record:

- market-data timestamp
- signal timestamp
- risk-check timestamp
- order-submit timestamp
- broker acknowledgement timestamp
- fill timestamp

Calculate:

- signal latency
- API latency
- execution latency
- total decision-to-fill latency

---

# 34. Position Management

After entry, continuously monitor:

- P&L
- underlying movement
- option Greeks
- volatility
- spread
- regime
- strategy validity
- ML probability
- target
- stop
- time in position

Possible actions:

- HOLD
- PARTIAL_EXIT
- MOVE_STOP
- TRAIL_STOP
- EXIT
- EMERGENCY_EXIT

---

# 35. Dynamic Exit Engine

Do not use one fixed exit rule.

Possible exit reasons:

- target reached
- stop reached
- prediction invalidated
- regime changed
- strategy invalidated
- volatility changed
- time limit reached
- option liquidity deteriorated
- risk limit reached
- portfolio risk changed
- end-of-session rule
- emergency shutdown

Every exit must have a recorded reason.

---

# 36. No-Trade Engine

NO_TRADE must be an explicit outcome.

Reject opportunities when:

- market uncertain
- spread excessive
- liquidity poor
- prediction confidence insufficient
- expected value insufficient
- risk/reward insufficient
- data stale
- API unstable
- portfolio exposure too high
- volatility unsuitable
- strategy regime mismatch
- execution quality poor

---

# 37. News/Event Awareness

Create an EventRisk interface.

The system must be designed to optionally consume structured event data.

Examples:

- RBI events
- Fed events
- inflation releases
- major economic releases
- corporate events
- major index events

Do not scrape random websites inside the execution loop.

Event data must come through a controlled data source.

When a major event is approaching, the RiskEngine may:

- reduce risk
- disable new positions
- widen required confidence
- switch strategy
- close positions
- continue normal trading

based on configurable rules.

---

# 38. Backtesting Engine

Build a realistic event-driven backtester.

It must simulate:

- historical prices
- option chains
- spreads
- slippage
- commissions
- order execution
- position sizing
- stops
- targets
- latency assumptions
- trading hours
- expiry
- contract rollover
- liquidity constraints

Do not use future information.

---

# 39. Backtesting Metrics

Report:

- total return
- CAGR where applicable
- total trades
- winning trades
- losing trades
- win rate
- average win
- average loss
- expectancy
- profit factor
- maximum drawdown
- maximum adverse excursion
- maximum favorable excursion
- Sharpe
- Sortino
- Calmar
- average holding time
- consecutive losses
- consecutive wins
- monthly performance
- daily performance
- strategy contribution
- instrument contribution
- regime contribution
- transaction costs
- slippage impact

---

# 40. Robustness Testing

Implement:

- walk-forward analysis
- Monte Carlo trade-order randomization
- parameter perturbation
- transaction-cost sensitivity
- slippage sensitivity
- delayed-entry testing
- delayed-exit testing
- missing-data simulation
- spread widening simulation
- execution-failure simulation

A strategy must demonstrate robustness rather than only one beautiful backtest.

---

# 41. Paper Trading

Create a complete PAPER mode.

Paper mode must use:

REAL market data
+
SIMULATED execution

It must record:

- hypothetical orders
- simulated fills
- simulated P&L
- slippage
- latency
- risk metrics

No real broker order may be submitted in PAPER mode.

---

# 42. Demo Mode

Create a DEMO mode where supported by the broker environment.

The application must make the mode explicit at startup.

Example:

MODE=DEMO

The application must display the mode prominently.

---

# 43. Live Mode

LIVE mode must require explicit configuration.

LIVE must never be the default.

The application must require multiple independent confirmations before enabling live execution.

At startup:

TRADING MODE: LIVE

must be clearly visible.

---

# 44. Kill Switch

Implement:

- application kill switch
- strategy kill switch
- instrument kill switch
- portfolio kill switch
- broker kill switch
- global emergency shutdown

Emergency shutdown must:

- stop opening new positions
- optionally close managed positions according to configured emergency policy
- cancel outstanding orders where appropriate
- persist state
- alert the operator

---

# 45. Broker/API Failure Handling

Handle:

- timeout
- authentication failure
- rate limit
- HTTP failure
- invalid request
- duplicate request
- network failure
- broker rejection
- partial fill
- order status ambiguity
- disconnected feed

Never blindly retry an order without idempotency protection.

Prevent duplicate orders.

---

# 46. State Reconciliation

At startup and periodically:

Compare:

LOCAL STATE

against:

BROKER STATE

Detect:

- missing positions
- unexpected positions
- missing orders
- unexpected orders
- quantity mismatch
- price mismatch

The broker must be treated as the source of truth for actual execution state.

---

# 47. Trade Journal

Every signal and trade must be stored.

Required fields:

- timestamp
- underlying
- instrument
- expiry
- strike
- option type
- strategy
- regime
- ML probability
- expected return
- expected risk
- entry
- stop
- target
- position size
- broker order ID
- fill
- exit
- P&L
- reason
- latency
- spread
- slippage
- portfolio state

---

# 48. Explainability

Every trade must answer:

WHY DID WE ENTER?

WHY THIS INSTRUMENT?

WHY THIS STRIKE?

WHY THIS EXPIRY?

WHY THIS SIZE?

WHY THIS STOP?

WHY THIS TARGET?

WHY DID WE EXIT?

The system must produce a structured decision report.

---

# 49. AI Research Assistant Integration

Claude Code is a DEVELOPMENT AND RESEARCH ASSISTANT.

It must NOT be required for every live trade.

The production trading engine must run without Anthropic API access.

Claude Code may be used to:

- create code
- refactor code
- analyze backtests
- inspect logs
- analyze losing trades
- compare strategies
- generate research reports
- identify bugs
- improve tests
- investigate model degradation
- propose experiments
- inspect model performance
- explain system behavior

The production trading loop must remain deterministic and autonomous.

---

# 50. Automated Research Loop

Create tooling that allows Claude Code to analyze generated results.

Pipeline:

DATA
→ FEATURES
→ TRAIN
→ BACKTEST
→ WALK-FORWARD
→ PAPER TRADE
→ ANALYZE
→ EXPERIMENT
→ RETRAIN
→ VALIDATE

Every experiment must be versioned.

Store:

- dataset version
- feature version
- strategy version
- model version
- configuration
- random seed
- backtest period
- results

---

# 51. Model Registry

Every trained ML model must have:

- unique model ID
- training date
- feature version
- training dataset
- validation results
- out-of-sample results
- calibration metrics
- trading metrics
- deployment status

Possible status:

RESEARCH
VALIDATION
PAPER
ACTIVE
RETIRED

---

# 52. Model Drift Detection

Monitor whether the live feature distribution differs significantly from training data.

Detect:

- feature drift
- prediction drift
- calibration drift
- regime drift
- performance degradation

If drift becomes significant:

reduce exposure or pause the model.

---

# 53. Strategy Competition

Allow strategies to compete.

Track performance by:

- market
- instrument
- regime
- time
- volatility
- expiry state

The system may dynamically allocate more opportunity to strategies demonstrating robust out-of-sample performance.

Do not dynamically optimize parameters from tiny recent samples.

Avoid overfitting.

---

# 54. Ensemble Confidence

The final trade score must consider:

- strategy agreement
- ML probability
- regime confidence
- expected return
- expected risk
- liquidity
- spread
- execution quality
- portfolio exposure

Do not equate:

confidence = probability.

Confidence must be calibrated.

---

# 55. Expected Value

Every candidate trade should calculate an estimated expected value.

Conceptually:

Expected Value =
Probability of favorable outcome × expected gain
minus
Probability of unfavorable outcome × expected loss
minus
transaction costs
minus
estimated slippage

Only trades above the configured minimum expected value should reach the RiskEngine.

---

# 56. Transaction Cost Awareness

The system must model:

- brokerage
- exchange charges
- taxes/levies applicable to the transaction
- slippage
- spread
- other applicable costs

Use current broker/exchange specifications where available.

Never assume zero transaction costs.

---

# 57. Capital Allocation

Support:

- single-trade allocation
- strategy allocation
- instrument allocation
- portfolio allocation

Capital allocation must be dynamically constrained by:

- available margin
- risk
- correlation
- liquidity
- current exposure

Never hard-code a particular capital amount.

---

# 58. User Configuration

The user must be able to configure through a CLI/configuration interface:

- available capital
- maximum capital deployment
- risk per trade
- daily loss limit
- drawdown limit
- maximum positions
- instruments
- strategies
- trading hours
- allowed expiries
- maximum spread
- minimum liquidity
- minimum model probability
- minimum expected value
- trading mode

The bot must validate configuration before starting.

---

# 59. Interactive Trading Command Interface

Create commands such as:

status

portfolio

positions

orders

signals

opportunities

strategies

models

risk

performance

backtest

paper-start

paper-stop

demo-start

demo-stop

live-status

kill

resume

config

The interface must clearly show the current trading mode.

---

# 60. Monitoring Dashboard

Create a local dashboard.

Show:

- market regime
- active opportunities
- active positions
- P&L
- drawdown
- risk utilization
- strategy health
- ML probabilities
- model health
- API health
- execution latency
- open orders
- recent decisions

---

# 61. Alerting

Support configurable alerts for:

- trade entry
- trade exit
- rejected trade
- risk limit
- drawdown limit
- broker error
- API disconnect
- strategy disabled
- model drift
- emergency shutdown

---

# 62. Security

Never store:

- broker password
- API token
- API secret
- client credentials

inside source code.

Use:

- environment variables
- .env excluded from Git
- OS credential store
- secure secrets management where appropriate

Never log credentials.

---

# 63. Reliability

The bot must survive:

- application restart
- network interruption
- broker API interruption
- market-data interruption
- system clock issues
- partial execution
- process crash

On restart:

1. Load persisted state.
2. Connect to broker.
3. Reconcile broker state.
4. Detect unmanaged positions.
5. Detect outstanding orders.
6. Restore monitoring.
7. Only then allow new signals.

---

# 64. Testing

Required:

## Unit tests

For:

- indicators
- features
- strategies
- regime detection
- ML inference
- option selection
- position sizing
- risk management
- order generation

## Integration tests

For:

- broker adapter
- market data
- order lifecycle
- state reconciliation

## Simulation tests

For:

- flash volatility
- spread widening
- API failure
- delayed data
- partial fill
- duplicate response
- order rejection
- sudden market reversal

---

# 65. Paper Trading Acceptance Criteria

Before demo/live:

The system must demonstrate:

- correct market-data ingestion
- correct option selection
- correct signal generation
- correct position sizing
- correct risk controls
- correct simulated execution
- correct P&L accounting
- correct state reconciliation

No live order execution should be enabled during this stage.

---

# 66. Development Phases

Claude Code must implement the project incrementally.

## Phase 1

Architecture and repository.

## Phase 2

Configuration system.

## Phase 3

Broker abstraction.

## Phase 4

Groww adapter.

## Phase 5

Market-data engine.

## Phase 6

Instrument discovery.

## Phase 7

Feature engine.

## Phase 8

Regime engine.

## Phase 9

Strategy framework.

## Phase 10

Initial strategies.

## Phase 11

Option analytics.

## Phase 12

ML training pipeline.

## Phase 13

Prediction engine.

## Phase 14

Signal ensemble.

## Phase 15

Risk engine.

## Phase 16

Backtesting engine.

## Phase 17

Walk-forward testing.

## Phase 18

Paper trading.

## Phase 19

Monitoring.

## Phase 20

Demo execution.

## Phase 21

Live execution infrastructure.

Do not skip phases.

---

# 67. Claude Code Development Rules

Claude Code must:

1. Read this entire REQUIREMENTS.md before implementation.
2. Inspect the existing repository before creating files.
3. Maintain an architecture document.
4. Maintain a requirements traceability matrix.
5. Never silently remove requirements.
6. Ask for clarification only when a requirement is genuinely ambiguous.
7. Prefer configurable implementations.
8. Never hard-code account capital.
9. Never hard-code a single instrument.
10. Never hard-code a single strategy.
11. Never hard-code a single expiry.
12. Never hard-code lot size.
13. Never hard-code broker-specific assumptions outside the adapter.
14. Write tests for every critical component.
15. Run tests after major changes.
16. Fix test failures before moving forward.
17. Never claim a strategy is profitable without out-of-sample evidence.
18. Never optimize directly against final test data.
19. Never introduce look-ahead bias.
20. Never use future market information.
21. Never use an LLM in the critical execution path.
22. Never bypass RiskEngine.
23. Never allow strategies to directly place broker orders.
24. Never use martingale.
25. Never implement uncontrolled averaging-down.
26. Never implement uncontrolled grid trading.
27. Never duplicate orders after API failures.
28. Always reconcile broker state.
29. Keep paper/demo/live modes isolated.
30. LIVE must never be the default.

---

# 68. Research Quality Standard

The objective is not to create a backtest with an attractive equity curve.

The objective is to determine whether a strategy demonstrates a statistically and economically meaningful edge after:

- transaction costs
- slippage
- realistic execution
- regime changes
- out-of-sample testing
- walk-forward testing
- parameter perturbation
- Monte Carlo analysis

If evidence does not support an edge:

mark the strategy as FAILED.

Do not manipulate the strategy until the backtest becomes profitable.

---

# 69. Final System Architecture

The target architecture is:

USER
│
├── Capital configuration
├── Risk configuration
├── Strategy configuration
└── Trading-mode configuration
│
↓
TRADING ORCHESTRATOR
│
├── Market Data Engine
├── Instrument Discovery
├── Feature Engine
├── Regime Engine
├── Strategy Engine
├── ML Prediction Engine
├── Opportunity Scanner
├── Option Selector
├── Signal Ensemble
├── Risk Engine
├── Portfolio Manager
├── Position Manager
├── Execution Engine
├── Broker Adapter
├── Trade Journal
├── Monitoring
└── Alerting
│
↓
GROWW API
│
↓
NSE/BSE F&O

---

# 70. Definition of Done

The system is considered complete only when it can:

1. Discover eligible F&O instruments.
2. Consume validated market data.
3. Construct multi-timeframe features.
4. Detect market regimes.
5. Run multiple strategies.
6. Generate probabilistic ML predictions.
7. Rank opportunities.
8. Select suitable option contracts.
9. Calculate dynamic position size.
10. Apply portfolio-level risk.
11. Generate BUY/SELL/NO_TRADE.
12. Execute through the broker adapter.
13. Verify execution.
14. Manage open positions.
15. Exit positions according to defined logic.
16. Persist all decisions.
17. Backtest historically.
18. Perform walk-forward validation.
19. Perform out-of-sample testing.
20. Run paper trading.
21. Monitor live system health.
22. Recover from failures.
23. Reconcile broker state.
24. Provide complete trade explanations.
25. Operate without requiring Claude/Anthropic API access during live execution.

The final system must behave as an adaptive quantitative trading platform rather than a collection of technical indicators.

END OF REQUIREMENTS