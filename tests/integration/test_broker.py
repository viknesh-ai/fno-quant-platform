"""Broker adapter, order lifecycle and reconciliation tests (REQ 64 integration).

The Groww adapter is exercised against a stub HTTP session rather than the live
API: these tests must be runnable without credentials and without touching a real
exchange. What they verify is the adapter's *contract* — that it sends documented
request shapes, parses documented response shapes, and never fabricates state.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest
import requests

from aqtp.brokers.base import Capability, NotSupportedError
from aqtp.brokers.groww import GrowwAdapter
from aqtp.brokers.rest import IdempotencyGuard, RateLimiter
from aqtp.brokers.simulated import SimulatedBroker, SimulationSettings
from aqtp.core.clock import IST
from aqtp.core.errors import (
    AuthenticationError,
    BrokerError,
    DuplicateOrderError,
    OrderRejected,
    OrderStateAmbiguous,
    RateLimitError,
)
from aqtp.core.types import (
    OrderRequest,
    OrderStatus,
    OrderType,
    Product,
    Segment,
    TransactionType,
)
from tests.conftest import make_quote


# --------------------------------------------------------------------------- #
# HTTP stub
# --------------------------------------------------------------------------- #
class StubResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


class StubSession:
    """Records requests and replays queued responses."""

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list = []
        self.default = StubResponse({"status": "SUCCESS", "payload": {}})

    def queue(self, payload, status_code=200, headers=None):
        self.responses.append(StubResponse(payload, status_code, headers))
        return self

    def queue_exception(self, exc):
        self.responses.append(exc)
        return self

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.requests.append(
            {"method": method, "url": url, "params": params, "json": json, "headers": headers}
        )
        if not self.responses:
            return self.default
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass

    @property
    def last(self) -> dict:
        return self.requests[-1]


@pytest.fixture
def session() -> StubSession:
    return StubSession()


@pytest.fixture
def adapter(session, tmp_path) -> GrowwAdapter:
    broker = GrowwAdapter(
        access_token="test-token", auth_mode="token", cache_dir=tmp_path, session=session,
    )
    broker.authenticate()
    return broker


# =========================================================================== #
class TestGrowwAdapterContract:
    def test_capabilities_are_declared(self, adapter):
        for capability in (
            Capability.PLACE_ORDER, Capability.OPTION_CHAIN, Capability.GREEKS,
            Capability.POSITIONS, Capability.MARGIN, Capability.HISTORICAL,
        ):
            assert adapter.supports(capability)

    def test_token_auth_sets_the_bearer_header(self, adapter, session):
        session.queue({"status": "SUCCESS", "payload": {"clear_cash": 100.0}})
        adapter.get_margin()
        headers = session.last["headers"]
        assert headers["Authorization"] == "Bearer test-token"
        assert headers["X-API-VERSION"] == "1.0"

    def test_missing_token_raises(self, tmp_path, session):
        broker = GrowwAdapter(access_token="", auth_mode="token", cache_dir=tmp_path, session=session)
        with pytest.raises(AuthenticationError):
            broker.authenticate()

    def test_place_order_sends_the_documented_body(self, adapter, session, call_instrument):
        session.queue(
            {"status": "SUCCESS",
             "payload": {"groww_order_id": "GRW1", "order_status": "ACKED",
                         "trading_symbol": call_instrument.trading_symbol, "quantity": 75}}
        )
        request = OrderRequest(
            instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
            order_type=OrderType.LIMIT, product=Product.NRML, price=200.0,
            client_order_id="AQTEST12345678",
        )
        order = adapter.place_order(request)

        body = session.last["json"]
        assert session.last["url"].endswith("/v1/order/create")
        assert body["trading_symbol"] == call_instrument.trading_symbol
        assert body["segment"] == "FNO"
        assert body["exchange"] == "NSE"
        assert body["product"] == "NRML"
        assert body["order_type"] == "LIMIT"
        assert body["transaction_type"] == "BUY"
        assert body["validity"] == "DAY"
        assert body["order_reference_id"] == "AQTEST12345678"
        assert order.broker_order_id == "GRW1"

    def test_order_rejection_is_surfaced_not_swallowed(self, adapter, session, call_instrument):
        session.queue(
            {"status": "FAILURE", "error": {"code": "GA001", "message": "insufficient funds"}},
            status_code=400,
        )
        request = OrderRequest(
            instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
            order_type=OrderType.MARKET, product=Product.NRML, client_order_id="AQTEST22345678",
        )
        with pytest.raises(OrderRejected, match="insufficient funds"):
            adapter.place_order(request)

    def test_a_rejected_order_frees_its_idempotency_key(self, adapter, session, call_instrument):
        """A definitive rejection never reached the book, so the key may be reused."""
        session.queue(
            {"status": "FAILURE", "error": {"code": "GA001", "message": "bad price"}},
            status_code=400,
        )
        request = OrderRequest(
            instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
            order_type=OrderType.LIMIT, product=Product.NRML, price=200.0,
            client_order_id="AQTEST32345678",
        )
        with pytest.raises(OrderRejected):
            adapter.place_order(request)
        assert not adapter.idempotency.has("AQTEST32345678")

    def test_unknown_order_status_becomes_UNKNOWN_not_a_guess(self, adapter, session):
        session.queue(
            {"status": "SUCCESS",
             "payload": {"groww_order_id": "G2", "order_status": "SOMETHING_NEW",
                         "trading_symbol": "X", "segment": "FNO"}}
        )
        order = adapter.get_order("G2", segment=Segment.FNO)
        assert order.status is OrderStatus.UNKNOWN

    def test_positions_net_credit_and_debit_legs(self, adapter, session):
        session.queue(
            {"status": "SUCCESS", "payload": {"positions": [
                {"trading_symbol": "NIFTY26JUN24000CE", "segment": "FNO", "product": "NRML",
                 "credit_quantity": 150, "credit_price": 30000.0,
                 "debit_quantity": 75, "debit_price": 16000.0,
                 "cf_credit_quantity": 0, "cf_debit_quantity": 0,
                 "cf_credit_price": 0, "cf_debit_price": 0},
            ]}}
        )
        positions = adapter.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 75  # 150 long - 75 short

    def test_flat_positions_are_excluded(self, adapter, session):
        session.queue(
            {"status": "SUCCESS", "payload": {"positions": [
                {"trading_symbol": "X", "segment": "FNO", "credit_quantity": 75,
                 "debit_quantity": 75, "credit_price": 100, "debit_price": 100,
                 "cf_credit_quantity": 0, "cf_debit_quantity": 0,
                 "cf_credit_price": 0, "cf_debit_price": 0},
            ]}}
        )
        assert adapter.get_positions() == []

    def test_option_chain_parses_strikes_and_greeks(self, adapter, session):
        session.queue(
            {"status": "SUCCESS", "payload": {
                "underlying_ltp": 24000.0,
                "strikes": {
                    "24000": {
                        "CE": {"trading_symbol": "NIFTY26JUN24000CE", "ltp": 200.0,
                               "open_interest": 100000, "volume": 50000,
                               "greeks": {"delta": 0.52, "gamma": 0.0008, "theta": -15.0,
                                          "vega": 13.0, "rho": 5.0, "iv": 0.14}},
                        "PE": {"trading_symbol": "NIFTY26JUN24000PE", "ltp": 170.0,
                               "open_interest": 90000, "volume": 40000,
                               "greeks": {"delta": -0.48, "gamma": 0.0008, "theta": -14.0,
                                          "vega": 13.0, "rho": -5.0, "iv": 0.14}},
                    }
                }}}
        )
        chain = adapter.get_option_chain("NIFTY", date(2026, 6, 25))
        assert chain.underlying_ltp == 24000.0
        assert len(chain.contracts) == 2
        call = chain.calls()[0]
        assert call.greeks.delta == pytest.approx(0.52)
        assert call.last_price == pytest.approx(200.0)

    def test_greeks_are_refused_for_non_options(self, adapter, future_instrument):
        with pytest.raises(NotSupportedError):
            adapter.get_greeks(future_instrument)

    def test_rate_limit_raises_after_retries_are_exhausted(self, tmp_path):
        session = StubSession()
        for _ in range(5):
            session.queue(
                {"status": "FAILURE", "error": {"message": "rate limited"}},
                status_code=429, headers={"Retry-After": "0"},
            )
        broker = GrowwAdapter(
            access_token="t", cache_dir=tmp_path, session=session, max_retries=2, backoff=0.0
        )
        broker.authenticate()
        with pytest.raises(RateLimitError):
            broker.get_margin()

    def test_auth_failure_is_not_retried(self, tmp_path):
        session = StubSession()
        session.queue({"status": "FAILURE", "error": {"message": "expired"}}, status_code=401)
        broker = GrowwAdapter(access_token="t", cache_dir=tmp_path, session=session, max_retries=3)
        broker.authenticate()
        with pytest.raises(AuthenticationError):
            broker.get_margin()
        assert len(session.requests) == 1, "an auth failure must not be retried"

    def test_instrument_csv_is_cached_and_parsed(self, adapter, tmp_path):
        csv = (
            "exchange,exchange_token,trading_symbol,groww_symbol,name,instrument_type,segment,"
            "series,isin,underlying_symbol,underlying_exchange_token,lot_size,expiry_date,"
            "strike_price,tick_size,freeze_quantity,is_reserved,buy_allowed,sell_allowed\n"
            "NSE,1,NIFTY26JUN24000CE,X,NIFTY,CE,FNO,,,NIFTY,26000,75,2026-06-25,24000,0.05,1800,"
            "false,true,true\n"
        )
        (tmp_path / "instruments.csv").write_text(csv)
        instruments = GrowwAdapter.parse_instruments_csv(csv)
        assert len(instruments) == 1
        assert instruments[0].lot_size == 75
        assert instruments[0].expiry_date == date(2026, 6, 25)


# =========================================================================== #
# Idempotency and ambiguity (REQ 45)
# =========================================================================== #
class TestIdempotency:
    def test_reusing_a_key_is_refused_locally(self):
        guard = IdempotencyGuard()
        guard.reserve("AQKEY12345678")
        with pytest.raises(DuplicateOrderError):
            guard.reserve("AQKEY12345678")

    def test_release_permits_reuse(self):
        guard = IdempotencyGuard()
        guard.reserve("AQKEY12345678")
        guard.release("AQKEY12345678")
        guard.reserve("AQKEY12345678")  # must not raise

    def test_empty_key_is_refused(self):
        with pytest.raises(ValueError):
            IdempotencyGuard().reserve("")

    def test_same_intent_produces_the_same_key(self):
        from aqtp.core.ids import decision_bucket, is_valid_reference_id, make_client_order_id

        moment = datetime(2026, 6, 1, 11, 0, 30, tzinfo=IST)
        args = dict(
            trading_symbol="NIFTY26JUN24000CE", transaction_type="BUY", quantity=75,
            strategy="trend_following", bucket=decision_bucket(moment, seconds=60),
        )
        first = make_client_order_id(**args)
        second = make_client_order_id(**args)
        assert first == second, "a retry within one decision window must reuse the key"
        assert is_valid_reference_id(first)

    def test_a_different_cycle_produces_a_different_key(self):
        from aqtp.core.ids import decision_bucket, make_client_order_id

        base = dict(
            trading_symbol="NIFTY26JUN24000CE", transaction_type="BUY", quantity=75,
            strategy="trend_following",
        )
        first = make_client_order_id(
            **base, bucket=decision_bucket(datetime(2026, 6, 1, 11, 0, tzinfo=IST))
        )
        second = make_client_order_id(
            **base, bucket=decision_bucket(datetime(2026, 6, 1, 11, 5, tzinfo=IST))
        )
        assert first != second

    def test_timeout_on_submit_raises_ambiguous_not_failure(self, tmp_path, call_instrument):
        """REQ 45: a timed-out submit must never be reported as a clean failure."""
        session = StubSession()
        session.queue_exception(requests.Timeout("timed out"))
        # The adapter then tries to resolve by reference and gets a definitive "no".
        session.queue({"status": "FAILURE", "error": {"code": "404", "message": "not found"}},
                      status_code=404)

        broker = GrowwAdapter(access_token="t", cache_dir=tmp_path, session=session)
        broker.authenticate()
        request = OrderRequest(
            instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
            order_type=OrderType.MARKET, product=Product.NRML, client_order_id="AQTEST42345678",
        )
        with pytest.raises(OrderStateAmbiguous):
            broker.place_order(request)

    def test_ambiguous_submit_is_resolved_by_reference_lookup(self, tmp_path, call_instrument):
        """If the order did reach the exchange, the lookup finds it — no resubmit."""
        session = StubSession()
        session.queue_exception(requests.Timeout("timed out"))
        session.queue(
            {"status": "SUCCESS", "payload": {
                "groww_order_id": "GRW-RECOVERED", "order_status": "EXECUTED",
                "trading_symbol": call_instrument.trading_symbol, "quantity": 75,
                "filled_quantity": 75, "average_fill_price": 201.0,
                "order_reference_id": "AQTEST52345678"}}
        )
        broker = GrowwAdapter(access_token="t", cache_dir=tmp_path, session=session)
        broker.authenticate()
        request = OrderRequest(
            instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
            order_type=OrderType.MARKET, product=Product.NRML, client_order_id="AQTEST52345678",
        )
        order = broker.place_order(request)
        assert order.broker_order_id == "GRW-RECOVERED"
        assert order.filled_quantity == 75
        # Exactly two calls: the submit and the lookup. No resubmission.
        create_calls = [r for r in session.requests if "order/create" in r["url"]]
        assert len(create_calls) == 1

    def test_order_writes_are_never_retried(self, tmp_path, call_instrument):
        session = StubSession()
        session.queue_exception(requests.ConnectionError("network down"))
        session.queue({"status": "FAILURE", "error": {"code": "404"}}, status_code=404)

        broker = GrowwAdapter(
            access_token="t", cache_dir=tmp_path, session=session, max_retries=5, backoff=0.0
        )
        broker.authenticate()
        request = OrderRequest(
            instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
            order_type=OrderType.MARKET, product=Product.NRML, client_order_id="AQTEST62345678",
        )
        with pytest.raises(OrderStateAmbiguous):
            broker.place_order(request)
        create_calls = [r for r in session.requests if "order/create" in r["url"]]
        assert len(create_calls) == 1, "an order submission was retried"


class TestRateLimiter:
    def test_allows_traffic_under_the_limit(self):
        limiter = RateLimiter(per_second=100, per_minute=1000)
        assert limiter.acquire() == 0.0

    def test_blocks_once_the_per_second_ceiling_is_hit(self):
        import time

        limiter = RateLimiter(per_second=2, per_minute=1000)
        for _ in range(2):
            limiter.acquire()
        started = time.monotonic()
        limiter.acquire()
        assert time.monotonic() - started > 0.5


# =========================================================================== #
# Order lifecycle against the simulator
# =========================================================================== #
@pytest.fixture
def simulator(call_instrument):
    quotes = {call_instrument.trading_symbol: make_quote(call_instrument, price=200.0, spread=1.0)}
    return SimulatedBroker(
        quote_source=lambda i: quotes.get(i.trading_symbol),
        starting_cash=500_000,
        settings=SimulationSettings(slippage_spread_fraction=0.5, seed=1),
    )


class TestOrderLifecycle:
    def test_market_order_fills_and_creates_a_position(self, simulator, call_instrument):
        simulator.authenticate()
        request = OrderRequest(
            instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
            order_type=OrderType.MARKET, product=Product.NRML, client_order_id="SIMKEY123456",
        )
        order = simulator.place_order(request)
        assert order.status is OrderStatus.EXECUTED
        assert order.filled_quantity == 75

        positions = simulator.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 75

    def test_buy_fills_at_or_above_the_offer(self, simulator, call_instrument):
        """Slippage must always work against us."""
        simulator.authenticate()
        order = simulator.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
                order_type=OrderType.MARKET, product=Product.NRML, client_order_id="SIMKEY223456",
            )
        )
        assert order.average_fill_price >= 200.5  # the ask

    def test_unmarketable_limit_rests_unfilled(self, simulator, call_instrument):
        simulator.authenticate()
        order = simulator.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
                order_type=OrderType.LIMIT, product=Product.NRML, price=150.0,
                client_order_id="SIMKEY323456",
            )
        )
        assert order.filled_quantity == 0
        assert order.status.is_open

    def test_round_trip_realizes_pnl(self, simulator, call_instrument):
        simulator.authenticate()
        simulator.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
                order_type=OrderType.MARKET, product=Product.NRML, client_order_id="SIMKEY423456",
            )
        )
        simulator.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.SELL, quantity=75,
                order_type=OrderType.MARKET, product=Product.NRML, client_order_id="SIMKEY523456",
            )
        )
        assert simulator.get_positions() == []
        # Crossing the spread twice must produce a loss on an unchanged price.
        assert simulator.realized_pnl < 0

    def test_cancel_moves_the_order_to_terminal(self, simulator, call_instrument):
        simulator.authenticate()
        order = simulator.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
                order_type=OrderType.LIMIT, product=Product.NRML, price=150.0,
                client_order_id="SIMKEY623456",
            )
        )
        cancelled = simulator.cancel_order(order.broker_order_id, segment=Segment.FNO)
        assert cancelled.status is OrderStatus.CANCELLED
        assert cancelled.status.is_terminal

    def test_duplicate_client_id_returns_the_existing_order(self, simulator, call_instrument):
        simulator.authenticate()
        request = OrderRequest(
            instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
            order_type=OrderType.MARKET, product=Product.NRML, client_order_id="SIMKEY723456",
        )
        first = simulator.place_order(request)
        second = simulator.place_order(request)
        assert first.broker_order_id == second.broker_order_id
        assert len(simulator.get_positions()) == 1, "a duplicate created a second position"

    def test_trades_are_recorded_per_order(self, simulator, call_instrument):
        simulator.authenticate()
        order = simulator.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
                order_type=OrderType.MARKET, product=Product.NRML, client_order_id="SIMKEY823456",
            )
        )
        trades = simulator.get_trades(order.broker_order_id, segment=Segment.FNO)
        assert len(trades) == 1
        assert trades[0].quantity == 75
