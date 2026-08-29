# AQTP — Adaptive Quantitative Trading Platform

An AI-assisted algorithmic trading platform for Indian exchange-traded F&O markets,
built to the specification in `REQUIREMENTS.md`.

**Default mode is PAPER. LIVE cannot be enabled from a config file.**

---

## What this is

A quantitative research and execution system with five independent layers — market
understanding, strategy generation, predictive modelling, risk management and
execution — wired into one fixed pipeline:

```
Market Data → Features → Regime → Strategies → ML Prediction → Ensemble
  → Ranking → Risk Engine → Sizing → Execution → Broker
```

A strategy cannot place an order. Execution cannot happen without a `RiskApproval`.
Paper and backtest modes cannot reach a real order endpoint. These are enforced by
the type signatures, not by convention — see `docs/ARCHITECTURE.md` §1.

## What this is not

**It contains no strategy that has been shown to make money.** The reference
EMA-cross source exists to exercise the backtester, and its own robustness report
marks it `NOT ROBUST`. Finding an edge is the research work this platform is built
to support — it is not something the platform ships with.

---

## Install

```bash
cd fno-quant-platform
python3 -m pip install -e ".[ml,dashboard,dev]"
cp .env.example .env          # then fill in your Groww credentials
```

Requires Python 3.10+. Core dependencies are numpy, pandas, scikit-learn, pydantic,
typer and requests; LightGBM/XGBoost/CatBoost are optional and register themselves
only if installed.

## Configure

Everything lives in `config/default.yaml` and is validated at startup — a typo is a
hard error, not a silent default. The only value you must set is your capital:

```yaml
capital:
  available_capital: 500000     # the bot cross-checks this against broker margin
risk:
  risk_per_trade_pct: 0.01      # 1% of equity at risk per trade
  max_daily_loss_pct: 0.03
  max_open_positions: 4
```

Check it before running anything:

```bash
aqtp config --validate
aqtp doctor                     # credentials + broker connectivity
```

> **Sizing reality check.** With ₹5 lakh and 1% risk per trade, a single NIFTY
> option lot (75 units) will often exceed the risk budget, and the sizer will
> correctly return zero rather than over-risk. If you see `position_sizing` in your
> NO_TRADE reasons, that is the system working — either raise capital, widen
> `risk_per_trade_pct`, or trade smaller-lot underlyings.

## Run

```bash
# Paper trading: real market data, simulated execution, no broker orders
aqtp paper-start

# Inspect
aqtp status
aqtp opportunities              # run one scan and rank candidates
aqtp signals                    # recent decisions, including why we did NOT trade
aqtp positions
aqtp performance --days 30
aqtp explain D<decision-id>     # the full 8-question decision report

# Control
aqtp kill --scope global --reason "investigating"
aqtp resume
```

The dashboard runs at `http://127.0.0.1:8787` while a session is live.

## Research

```bash
aqtp research backtest NIFTY26JUNFUT --days 180
aqtp research walkforward NIFTY26JUNFUT --days 365
aqtp research train NIFTY --days 365 --horizon 30
aqtp research experiments
aqtp research promote <model-id> VALIDATION
```

Models climb `RESEARCH → VALIDATION → PAPER → ACTIVE → RETIRED` one rung at a time,
and a model whose walk-forward verdict was `FAILED` cannot be activated at all.

## Going live

LIVE requires **three independent environment interlocks** plus a typed confirmation
phrase. Check your state first:

```bash
aqtp live-status
```

Then, only if you genuinely intend to trade real money:

```bash
export AQTP_LIVE_CONFIRM_1=I_UNDERSTAND_REAL_MONEY
export AQTP_LIVE_CONFIRM_2=I_HAVE_REVIEWED_RISK_LIMITS
export AQTP_LIVE_CONFIRM_3=ENABLE_LIVE_TRADING
aqtp live-start
```

Before you do, the requirements themselves ask for evidence you should insist on:

- a strategy that survives walk-forward validation and the robustness suite,
- a model whose out-of-sample verdict is `PASSED`,
- a real paper-trading period reviewed in the journal (REQ 65),
- transaction cost rates re-verified against a current contract note.

## Testing

```bash
pytest                          # 256 tests
pytest tests/unit/test_leakage.py -v   # the ones that matter most
```

The leakage suite is the highest-value part of the test set: it asserts that a
feature value at bar *t* does not change when future bars arrive, that
higher-timeframe bars are not visible before they close, that unresolvable labels
are dropped, and that scalers and calibrators never see across a split boundary.

## Layout

```
src/aqtp/       core · configuration · brokers · data · features · regime
                strategies · options · ml · signals · scanner · risk
                execution · events · journal · backtest · monitoring · cli
config/         default.yaml
docs/           ARCHITECTURE.md · TRACEABILITY.md
tests/          unit · integration · simulation
```

- `docs/ARCHITECTURE.md` — layer contracts, invariants, and known limitations
- `docs/TRACEABILITY.md` — every requirement mapped to code and to its test

## Safety

Credentials are read only from the environment; `.env` is git-ignored and a
redaction filter scrubs tokens from every log record. Kill switches persist across
restarts. On restart the system loads state, reconciles against the broker, adopts
or flags unmanaged positions, and only then accepts new signals.

The broker is always treated as the source of truth for execution state. An order
that times out mid-flight is resolved by querying, never by resending.
