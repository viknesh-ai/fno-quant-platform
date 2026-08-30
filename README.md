# AQTP — Adaptive Quantitative Trading Platform

A production algorithmic trading system for Indian exchange-traded F&O markets,
built to the specification in `REQUIREMENTS.md`.

**Default mode is LIVE.** This is a real-money system: it places real orders
through the Groww Trading API, manages them in real time, and squares off before
the close. PAPER remains available for dry runs, but it is not the default.

---

## What this is

Six independent layers — market understanding, deep analysis, strategy
generation, predictive modelling, risk management and execution — wired into one
fixed pipeline:

```
Market Data → Features → Deep Analysis → Regime → Strategies → ML Prediction
  → Ensemble → Ranking → Risk Engine → Sizing → Execution → Broker
```

A strategy cannot place an order. Execution cannot happen without a
`RiskApproval`. Paper and backtest modes cannot reach a real order endpoint.
These are enforced by the type signatures, not by convention — see
`docs/ARCHITECTURE.md` §1.

## What this is not

**It contains no strategy that has been shown to make money.** The reference
EMA-cross source exists to exercise the backtester, and its own robustness report
marks it `NOT ROBUST`. Finding an edge is the research work this platform is
built to support — it is not something the platform ships with. The analysis
layer below makes the platform better at *evaluating* an edge. It does not
create one.

---

## The analysis layer

Most retail bots stack six momentum indicators and call the agreement
confirmation. Six indicators reading the same phenomenon is one opinion voting
six times. `src/aqtp/analysis/` measures eight things that are genuinely
independent, and only agreement *between* them counts:

| Dimension | What it measures | Module |
|---|---|---|
| **trend** | ADX/DI, EMA structure, VWAP distance, multi-timeframe alignment | `features/` |
| **momentum** | RSI, MACD, ROC, price acceleration, OBV slope | `features/` |
| **structure** | Volume-at-price: POC, value area, HVN/LVN, initial balance, break of structure | `analysis/volume_profile.py` |
| **orderflow** | Book imbalance, micro-price, depth slope, order-flow imbalance, VPIN-style toxicity, absorption, tape delta | `analysis/microstructure.py` |
| **positioning** | Dealer gamma exposure, gamma flip level, OI walls, max pain, 25Δ risk reversal, vanna/charm, OI buildup | `analysis/options_analytics.py` |
| **statistical** | Hurst, Lo–MacKinlay variance ratio, permutation entropy, efficiency ratio, autocorrelation, jump detection | `analysis/statistics.py` |
| **volatility** | Expansion/contraction, EWMA and GARCH(1,1) forecast, squeeze | `analysis/statistics.py` |
| **crossasset** | Pairwise correlation, breadth, dispersion, relative strength, India VIX | `analysis/crossasset.py` |

`analysis/confluence.py` combines them into a single conviction, and it does
three things most scoring code does not:

- **Conflict reduces conviction; it does not average away.** Four dimensions at
  +0.8 and four at −0.8 is not a confident zero, it is a low-conviction mess, and
  the score says so.
- **A veto is not a low score.** Toxic order flow, a 5σ jump that just
  invalidated every feature, an unfillable book, or high pin risk on expiry day
  mean *do not trade* — not *trade smaller*.
- **Every number carries its reasoning.** `aqtp analyze NIFTY` prints the whole
  audit trail, and it is stored on the decision in the journal.

The analysis layer has exactly three powers over the pipeline, all bounded by
`analysis.influence` in the config: it can veto, it can refuse a direction it
actively opposes, and it can scale a score. **It cannot create a trade** the
strategies and the model did not already agree on — `tests/integration/test_analysis_gate.py`
pins that.

---

## Install

```bash
cd fno-quant-platform
python3 -m pip install -e ".[ml,dashboard,dev]"
cp .env.example .env && chmod 600 .env    # then fill in your Groww credentials
```

Requires Python 3.10+. Core dependencies are numpy, pandas, scipy, scikit-learn,
pydantic, typer and requests; LightGBM/XGBoost/CatBoost are optional and register
themselves only if installed.

## Configure

Everything lives in `config/default.yaml` and is validated at startup — a typo is
a hard error, not a silent default. The only value you must set is your capital:

```yaml
mode: LIVE
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
aqtp doctor                     # credentials + broker connectivity + margin
```

> **Sizing reality check.** With ₹5 lakh and 1% risk per trade, a single NIFTY
> option lot (75 units) will often exceed the risk budget, and the sizer will
> correctly return zero rather than over-risk. If you see `position_sizing` in
> your NO_TRADE reasons, that is the system working — either raise capital, widen
> `risk_per_trade_pct`, or trade smaller-lot underlyings.

---

## Going live

LIVE requires three environment interlocks. The mode in the config says what the
platform is for; the interlocks say a human decided to run it today.

```bash
aqtp live-status                # what is armed right now
aqtp arm                        # prints the exports; run them yourself
```

```bash
export AQTP_LIVE_CONFIRM_1=I_UNDERSTAND_REAL_MONEY
export AQTP_LIVE_CONFIRM_2=I_HAVE_REVIEWED_RISK_LIMITS
export AQTP_LIVE_CONFIRM_3=ENABLE_LIVE_TRADING

aqtp doctor && aqtp start
```

`aqtp start` shows your capital, risk budget, daily loss cap and square-off time,
then asks you to type `I ACCEPT THE RISK`. Pass `--yes` to skip the prompt for
systemd or cron — the interlocks still apply, so `--yes` cannot start a live
session on its own.

Before you do, the requirements themselves ask for evidence you should insist on:

- a strategy that survives walk-forward validation and the robustness suite,
- a model whose out-of-sample verdict is `PASSED`,
- a real paper-trading period reviewed in the journal (REQ 65),
- transaction cost rates re-verified against a current contract note.

---

## Running the engine

```bash
aqtp start                            # LIVE (or whatever the config says)
aqtp start --mode PAPER               # dry run: real data, simulated fills
aqtp start --yes --no-dashboard       # unattended
aqtp start --cycles 100               # stop after N cycles

aqtp stop                             # graceful: stop at the next cycle boundary
aqtp stop --close-positions           # ...and flatten everything first
```

The process runs under a supervisor (`src/aqtp/runtime/supervisor.py`) that
handles what a live session actually needs:

- **SIGTERM/SIGINT** stop at a cycle boundary, never mid-order.
- **A dropped broker session** is re-authenticated with exponential backoff, and
  entries stay blocked until reconciliation confirms the state.
- **A wedged cycle** trips a watchdog after `runtime.watchdog_timeout_seconds`,
  which alerts, halts entries, or flattens — your choice.
- **Repeated cycle failures** halt the process rather than trade blind.
- **Square-off** happens at `session.square_off_time` regardless of what any
  strategy thinks.
- **Exiting with open positions** raises a CRITICAL alert, because a process that
  exits holding risk is a fact you need before you close the terminal.

The dashboard runs at `http://127.0.0.1:8787` while a session is live.

---

## Executing trades by hand

Every manual order goes through the **same** risk engine, sizer, execution engine
and journal as the automated pipeline. A manual position gets a stop, a target,
and is managed and squared off like any other. It is not a back door.

### Buy

```bash
# ATM call on the nearest expiry, sized by the risk engine
aqtp trade buy NIFTY --option CE

# A specific strike and expiry, one lot
aqtp trade buy NIFTY --option PE --strike 23800 --expiry 2026-06-25 --lots 1

# An exact contract by trading symbol
aqtp trade buy NIFTY26JUN24000CE --lots 2

# The future instead of an option
aqtp trade buy BANKNIFTY --future --lots 1

# With your own levels and a limit price
aqtp trade buy NIFTY --option CE --lots 1 \
    --limit 128.50 --stop 95 --target 220

# Levels as a fraction of premium instead of absolute prices
aqtp trade buy NIFTY --option CE --lots 1 --stop-pct 0.25 --target-pct 0.75
```

### Sell

```bash
aqtp trade sell NIFTY --option CE --strike 24200 --lots 1
aqtp trade sell NIFTY26JUNFUT --lots 1
```

### Check before you send

```bash
# Resolve, quote, and run every risk check — without sending anything
aqtp trade buy NIFTY --option CE --lots 1 --dry-run

# Live quote, spread, depth ladder and microstructure read
aqtp trade quote NIFTY --option CE --strike 24000
```

`--dry-run` prints the full risk checklist — kill switches, drawdown ladder,
exposure caps, margin, spread, liquidity, expected value, sizing — with PASS /
FAIL / SKIP against each, then tells you exactly what it *would* send.

### Close and manage

```bash
aqtp positions                        # position ids live here
aqtp trade exit P<position-id>                    # close at market
aqtp trade exit P<position-id> --quantity 75      # partial exit
aqtp trade close-all                              # flatten everything
aqtp trade close-all --yes                        # ...without the prompt

aqtp trade set-stop P<position-id> --stop 95 --target 250   # move the levels

aqtp orders                           # recent orders
aqtp trade modify <broker-order-id> --price 130 --quantity 75
aqtp trade cancel <broker-order-id>
aqtp trade cancel --all
```

### Sizing, and `--force`

By default the sizer decides the size. `--lots` and `--quantity` are honoured
only if they are **no larger** than what the risk budget allows — asking for more
is a rejection, not a silent trim:

```
✗ BUY refused: requested 15000 units but risk sizing allows 75
  (risk_per_trade). Reduce the size, or pass --force to override deliberately.
```

`--force` overrides the sizing and scoring checks, requires an explicit size, and
is journalled as a forced order so it shows up in any review:

```bash
aqtp trade buy NIFTY --option CE --lots 5 --force
```

`--force` does **not** bypass kill switches, and it does **not** bypass the
exchange freeze limit. Both are asserted in `tests/integration/test_manual_trading.py`.

---

## Analysis and inspection

```bash
aqtp analyze NIFTY                    # the full eight-dimension report
aqtp analyze NIFTY --dimensions       # just the confluence table
aqtp analyze NIFTY --json             # the ~50 analysis features, as JSON

aqtp market                           # breadth, correlation, dispersion, VIX
aqtp opportunities                    # one scan, ranked candidates
aqtp universe --filter BANK           # what is tradable

aqtp status                           # equity, positions, kill switches
aqtp portfolio                        # exposure and aggregate greeks
aqtp positions
aqtp orders
aqtp signals                          # recent decisions, including why NOT
aqtp performance --days 30
aqtp risk                             # limits and current utilisation
aqtp strategies
aqtp models
aqtp explain D<decision-id>           # the full 8-question decision report
```

`aqtp analyze` output looks like this:

```
dimension      score  confidence  weight  reading
trend          +0.79        1.00    0.16  ADX 48.4 with DI spread +26.0
structure      +0.32        1.00    0.14  price above value (POC 24,120.00)
momentum       +0.26        1.00    0.14  RSI 71.3
positioning    -0.18        0.87    0.14  dealers are LONG gamma — moves get damped
orderflow      +0.04        0.15    0.16  tape pressure +0.16
statistical    -0.05        0.40    0.10  Hurst 0.52 — random-walk-like

╭──────────────────────────────────────────────────────────────╮
│ NIFTY — bias LONG                                            │
│ net score +0.372 · conviction 0.275 · alignment 100%         │
│ agreeing: trend, momentum, structure                         │
│ conflicting: positioning                                     │
╰──────────────────────────────────────────────────────────────╯
```

## Control

```bash
aqtp kill --scope global --reason "investigating"
aqtp kill --scope global --close-positions --reason "emergency"
aqtp kill --scope strategy --target momentum --reason "expectancy collapsed"
aqtp kill --scope instrument --target NIFTY26JUN24000CE
aqtp resume                           # release all
aqtp resume --scope strategy --target momentum --note "root cause fixed"
```

Kill switches persist across restarts. A global switch engaged from a previous
run refuses to let the process start until you clear it.

## Research

```bash
aqtp research backtest NIFTY26JUNFUT --days 180
aqtp research walkforward NIFTY26JUNFUT --days 365
aqtp research train NIFTY --days 365 --horizon 30
aqtp research experiments
aqtp research promote <model-id> VALIDATION
```

Models climb `RESEARCH → VALIDATION → PAPER → ACTIVE → RETIRED` one rung at a
time, and a model whose walk-forward verdict was `FAILED` cannot be activated at
all.

---

## Deployment

### systemd

```bash
sudo useradd -r -s /usr/sbin/nologin aqtp
sudo install -d -o aqtp -g aqtp /opt/aqtp /etc/aqtp
sudo install -m 600 -o aqtp .env /etc/aqtp/aqtp.env    # credentials + interlocks
sudo cp deploy/aqtp.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now aqtp
journalctl -u aqtp -f
```

`TimeoutStopSec=120` gives the supervisor time to cancel orders and square off on
`systemctl stop`. Do not lower it below `runtime.shutdown_grace_seconds`.

### Docker

```bash
docker build -f deploy/Dockerfile -t aqtp:latest .
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml exec aqtp aqtp status
```

The image pins `TZ=Asia/Kolkata`. Every session boundary, expiry and square-off
time in this system is IST, and a container running in UTC would silently trade
the wrong window.

### Daily operations

The Groww access token expires at 06:00 IST. With `auth_mode: token` you must
refresh `GROWW_ACCESS_TOKEN` and restart daily; with `approval` or `totp` the
token is minted at startup, so a restart after 06:00 is enough.

```bash
0 15 9 * * 1-5  systemctl restart aqtp     # 09:15 IST, weekdays
```

---

## Testing

```bash
pytest                                      # 388 tests
pytest tests/unit/test_leakage.py -v        # the ones that matter most
pytest tests/unit/test_analysis.py -v       # the analysis layer
pytest tests/integration/test_analysis_gate.py -v    # what analysis may/may not do
pytest tests/integration/test_manual_trading.py -v   # manual orders
pytest tests/simulation/test_supervisor.py -v        # failure paths
```

The leakage suite is the highest-value part of the test set: it asserts that a
feature value at bar *t* does not change when future bars arrive, that
higher-timeframe bars are not visible before they close, that unresolvable labels
are dropped, and that scalers and calibrators never see across a split boundary.

## Layout

```
src/aqtp/       core · configuration · brokers · data · features · analysis
                regime · strategies · options · ml · signals · scanner · risk
                execution · events · journal · backtest · monitoring · runtime · cli
config/         default.yaml
deploy/         aqtp.service · Dockerfile · docker-compose.yml
docs/           ARCHITECTURE.md · TRACEABILITY.md
tests/          unit · integration · simulation
```

- `docs/ARCHITECTURE.md` — layer contracts, invariants, and known limitations
- `docs/TRACEABILITY.md` — every requirement mapped to code and to its test

## Safety

Credentials are read only from the environment; `.env` is git-ignored and a
redaction filter scrubs tokens from every log record. Kill switches persist
across restarts. On restart the system loads state, reconciles against the
broker, adopts or flags unmanaged positions, and only then accepts new signals.

The broker is always treated as the source of truth for execution state. An order
that times out mid-flight is resolved by querying, never by resending.
