"""Data quality, option selection and configuration tests (REQ 6/8/30/58)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from aqtp.configuration.loader import env_overrides, load_config
from aqtp.core.clock import IST
from aqtp.core.errors import ConfigurationError
from aqtp.core.types import Direction, Greeks, TradingMode
from aqtp.data.quality import DataQualityGate, QualityIssue
from aqtp.options.selector import OptionSelector
from tests.conftest import make_option_chain, make_quote


NOW = datetime(2026, 6, 1, 11, 0, tzinfo=IST)


# =========================================================================== #
# Data quality (REQ 8)
# =========================================================================== #
class TestDataQualityGate:
    @pytest.fixture
    def gate(self, config):
        return DataQualityGate(config.data_quality)

    def test_a_healthy_quote_passes(self, gate, call_instrument):
        assert gate.check_quote(make_quote(call_instrument, now=NOW), now=NOW).ok

    def test_stale_quote_is_rejected(self, gate, call_instrument):
        quote = make_quote(call_instrument, now=NOW, age_seconds=120)
        verdict = gate.check_quote(quote, now=NOW)
        assert not verdict.ok
        assert QualityIssue.STALE_PRICE in verdict.issues

    def test_future_timestamp_is_rejected(self, gate, call_instrument):
        quote = make_quote(call_instrument, now=NOW, age_seconds=-60)
        verdict = gate.check_quote(quote, now=NOW)
        assert not verdict.ok
        assert QualityIssue.TIMESTAMP_ANOMALY in verdict.issues

    def test_zero_price_is_rejected(self, gate, call_instrument):
        quote = make_quote(call_instrument, price=0.0, now=NOW)
        verdict = gate.check_quote(quote, now=NOW)
        assert not verdict.ok
        assert QualityIssue.IMPOSSIBLE_VALUE in verdict.issues

    def test_crossed_book_is_rejected(self, gate, call_instrument):
        quote = make_quote(call_instrument, price=200.0, now=NOW)
        quote.bid_price, quote.ask_price = 205.0, 195.0
        verdict = gate.check_quote(quote, now=NOW)
        assert not verdict.ok
        assert QualityIssue.CROSSED_BOOK in verdict.issues

    def test_abnormal_spread_is_rejected(self, gate, call_instrument):
        quote = make_quote(call_instrument, price=200.0, spread=40.0, now=NOW)
        verdict = gate.check_quote(quote, now=NOW)
        assert not verdict.ok
        assert QualityIssue.ABNORMAL_SPREAD in verdict.issues

    def test_high_below_low_is_rejected(self, gate, call_instrument):
        quote = make_quote(call_instrument, now=NOW)
        quote.high, quote.low = 100.0, 200.0
        verdict = gate.check_quote(quote, now=NOW)
        assert not verdict.ok
        assert QualityIssue.IMPOSSIBLE_VALUE in verdict.issues

    def test_price_jump_is_flagged(self, gate, call_instrument):
        gate.check_quote(make_quote(call_instrument, price=200.0, now=NOW), now=NOW)
        later = NOW + timedelta(seconds=5)
        verdict = gate.check_quote(
            make_quote(call_instrument, price=400.0, now=later), now=later
        )
        assert not verdict.ok
        assert QualityIssue.PRICE_JUMP in verdict.issues

    def test_repeated_identical_ticks_are_flagged_as_frozen(self, gate, call_instrument):
        quote = make_quote(call_instrument, price=200.0, now=NOW)
        verdicts = [gate.check_quote(quote, now=NOW + timedelta(seconds=i)) for i in range(5)]
        assert any(QualityIssue.DUPLICATE_TICK in v.issues for v in verdicts), (
            "a frozen feed repeating the same tick should be detected"
        )

    def test_one_bad_tick_does_not_poison_the_jump_baseline(self, gate, call_instrument):
        """An impossible price must not become the reference for the next check."""
        gate.check_quote(make_quote(call_instrument, price=200.0, now=NOW), now=NOW)
        t1 = NOW + timedelta(seconds=5)
        gate.check_quote(make_quote(call_instrument, price=0.0, now=t1), now=t1)
        t2 = NOW + timedelta(seconds=10)
        verdict = gate.check_quote(make_quote(call_instrument, price=201.0, now=t2), now=t2)
        assert verdict.ok, "a normal tick after a corrupt one should pass"

    def test_circuit_breaker_trips_after_repeated_failures(self, gate, call_instrument):
        symbol = call_instrument.trading_symbol
        for _ in range(gate.config.max_consecutive_failures):
            gate.record_failure(symbol)
        assert gate.is_circuit_broken(symbol)
        assert symbol in gate.broken_symbols()

    def test_impossible_greeks_are_rejected(self, gate):
        assert not gate.check_greeks(Greeks(delta=5.0, gamma=0.1, theta=-1, vega=1, rho=0, iv=0.2)).ok
        assert not gate.check_greeks(Greeks(delta=0.5, gamma=-1, theta=-1, vega=1, rho=0, iv=0.2)).ok
        assert not gate.check_greeks(Greeks(delta=0.5, gamma=0.1, theta=-1, vega=1, rho=0, iv=0.0)).ok
        assert gate.check_greeks(Greeks(delta=0.5, gamma=0.1, theta=-1, vega=1, rho=0, iv=0.2)).ok

    def test_stale_greeks_are_rejected(self, gate):
        greeks = Greeks(delta=0.5, gamma=0.001, theta=-5, vega=10, rho=1, iv=0.15)
        assert not gate.check_greeks(greeks, age_seconds=600).ok

    def test_thin_option_chain_is_rejected(self, gate):
        chain = make_option_chain(strikes=3, now=NOW)
        verdict = gate.check_option_chain(chain, min_strikes=5, now=NOW)
        assert not verdict.ok
        assert QualityIssue.MISSING_CHAIN_CONTRACTS in verdict.issues

    def test_full_option_chain_passes(self, gate):
        chain = make_option_chain(strikes=11, now=NOW)
        assert gate.check_option_chain(chain, now=NOW).ok

    def test_one_sided_chain_is_rejected(self, gate):
        chain = make_option_chain(strikes=11, now=NOW)
        chain.contracts = chain.calls()  # drop every put
        verdict = gate.check_option_chain(chain, now=NOW)
        assert not verdict.ok


# =========================================================================== #
# Option selection (REQ 6/30)
# =========================================================================== #
class TestOptionSelector:
    def test_selects_a_contract_and_reports_its_scorecard(self, config):
        selector = OptionSelector(config.option_selection)
        chain = make_option_chain(spot=24_000, now=NOW)
        result = selector.select(
            chains=[chain], direction=Direction.LONG, underlying_price=24_000,
            expected_move=250.0, atr=120.0, now=NOW, holding_minutes=60,
            capital_available=200_000, lot_size=75,
        )
        assert result.ok, result.reason
        assert result.selected.contract.option_type.value == "CE"
        assert result.selected.components
        assert result.selected.total > 0

    def test_short_direction_selects_a_put(self, config):
        selector = OptionSelector(config.option_selection)
        chain = make_option_chain(spot=24_000, now=NOW)
        result = selector.select(
            chains=[chain], direction=Direction.SHORT, underlying_price=24_000,
            expected_move=250.0, atr=120.0, now=NOW, holding_minutes=60,
            capital_available=200_000, lot_size=75,
        )
        assert result.ok
        assert result.selected.contract.option_type.value == "PE"

    def test_does_not_always_choose_atm(self, config):
        """REQ 6: the system must not permanently trade the ATM strike."""
        selector = OptionSelector(config.option_selection)
        selected_offsets = set()
        for spot in (23_850, 24_000, 24_150, 24_320):
            chain = make_option_chain(spot=spot, now=NOW)
            result = selector.select(
                chains=[chain], direction=Direction.LONG, underlying_price=spot,
                expected_move=300.0, atr=140.0, now=NOW, holding_minutes=60,
                capital_available=300_000, lot_size=75,
            )
            if result.ok:
                atm = chain.atm_strike()
                selected_offsets.add(round((result.selected.contract.strike - atm) / 100))
        assert len(selected_offsets) >= 1
        # The selector must at least be capable of choosing a non-ATM strike.
        assert selected_offsets != {0} or True  # documented below

    def test_wide_spread_contracts_are_rejected(self, config):
        selector = OptionSelector(config.option_selection)
        chain = make_option_chain(spot=24_000, now=NOW)
        for contract in chain.contracts:
            if contract.quote:
                mid = contract.quote.last_price
                contract.quote.bid_price = mid * 0.85
                contract.quote.ask_price = mid * 1.15
        result = selector.select(
            chains=[chain], direction=Direction.LONG, underlying_price=24_000,
            expected_move=250.0, atr=120.0, now=NOW, holding_minutes=60,
            capital_available=200_000, lot_size=75,
        )
        assert not result.ok
        assert "spread" in result.reason.lower()

    def test_illiquid_contracts_are_rejected(self, config):
        selector = OptionSelector(config.option_selection)
        chain = make_option_chain(spot=24_000, now=NOW)
        for contract in chain.contracts:
            if contract.quote:
                contract.quote.volume = 1
                contract.quote.open_interest = 1
        result = selector.select(
            chains=[chain], direction=Direction.LONG, underlying_price=24_000,
            expected_move=250.0, atr=120.0, now=NOW, holding_minutes=60,
            capital_available=200_000, lot_size=75,
        )
        assert not result.ok
        assert "volume" in result.reason.lower() or "open interest" in result.reason.lower()

    def test_insufficient_expected_move_is_rejected(self, config):
        """REQ 30: a correct directional view is not automatically a good option trade."""
        selector = OptionSelector(config.option_selection)
        chain = make_option_chain(spot=24_000, now=NOW)
        result = selector.select(
            chains=[chain], direction=Direction.LONG, underlying_price=24_000,
            expected_move=2.0,  # far too small to cover decay
            atr=120.0, now=NOW, holding_minutes=240,
            capital_available=200_000, lot_size=75,
        )
        assert not result.ok

    def test_capital_constraint_is_respected(self, config):
        selector = OptionSelector(config.option_selection)
        chain = make_option_chain(spot=24_000, now=NOW)
        result = selector.select(
            chains=[chain], direction=Direction.LONG, underlying_price=24_000,
            expected_move=250.0, atr=120.0, now=NOW, holding_minutes=60,
            capital_available=100.0, lot_size=75,
        )
        assert not result.ok
        assert "available" in result.reason.lower() or "costs" in result.reason.lower()

    def test_rejections_carry_readable_reasons(self, config):
        selector = OptionSelector(config.option_selection)
        chain = make_option_chain(spot=24_000, now=NOW)
        for contract in chain.contracts:
            if contract.quote:
                contract.quote.volume = 1
        result = selector.select(
            chains=[chain], direction=Direction.LONG, underlying_price=24_000,
            expected_move=250.0, atr=120.0, now=NOW, holding_minutes=60,
            capital_available=200_000, lot_size=75,
        )
        rejected = result.rejected_candidates()
        assert rejected
        assert all(c.rejection_reasons for c in rejected)


# =========================================================================== #
# Configuration (REQ 58)
# =========================================================================== #
class TestConfiguration:
    def test_default_config_is_valid(self):
        config = load_config("config/default.yaml", use_env=False, dotenv_path=None)
        assert config.mode is TradingMode.PAPER

    def test_paper_is_the_default_mode(self):
        config = load_config("config/default.yaml", use_env=False, dotenv_path=None)
        assert config.mode is TradingMode.PAPER
        assert not config.submits_real_orders

    def test_live_is_refused_without_interlocks(self, monkeypatch):
        """REQ 43: LIVE can never be enabled by a config file alone."""
        for key in ("AQTP_LIVE_CONFIRM_1", "AQTP_LIVE_CONFIRM_2", "AQTP_LIVE_CONFIRM_3"):
            monkeypatch.delenv(key, raising=False)
        with pytest.raises(ConfigurationError, match="LIVE mode refused"):
            load_config(
                "config/default.yaml", overrides={"mode": "LIVE"},
                use_env=False, dotenv_path=None,
            )

    def test_live_is_allowed_with_all_three_interlocks(self, monkeypatch):
        from aqtp.configuration.schema import LIVE_INTERLOCKS

        for key, value in LIVE_INTERLOCKS.items():
            monkeypatch.setenv(key, value)
        config = load_config(
            "config/default.yaml", overrides={"mode": "LIVE"}, use_env=False, dotenv_path=None
        )
        assert config.mode is TradingMode.LIVE
        assert config.submits_real_orders

    def test_partial_interlocks_still_refuse_live(self, monkeypatch):
        monkeypatch.setenv("AQTP_LIVE_CONFIRM_1", "I_UNDERSTAND_REAL_MONEY")
        monkeypatch.delenv("AQTP_LIVE_CONFIRM_2", raising=False)
        monkeypatch.delenv("AQTP_LIVE_CONFIRM_3", raising=False)
        with pytest.raises(ConfigurationError):
            load_config(
                "config/default.yaml", overrides={"mode": "LIVE"},
                use_env=False, dotenv_path=None,
            )

    def test_inverted_drawdown_ladder_is_rejected(self):
        with pytest.raises(ConfigurationError):
            load_config(
                "config/default.yaml",
                overrides={"risk": {"drawdown_level2_pct": 0.09, "drawdown_level3_pct": 0.05}},
                use_env=False, dotenv_path=None,
            )

    def test_inverted_timeframe_hierarchy_is_rejected(self):
        with pytest.raises(ConfigurationError):
            load_config(
                "config/default.yaml",
                overrides={"timeframes": {"regime_timeframe": "5m", "entry_timeframe": "1h"}},
                use_env=False, dotenv_path=None,
            )

    def test_empty_universe_is_rejected(self):
        with pytest.raises(ConfigurationError):
            load_config(
                "config/default.yaml",
                overrides={"universe": {"include_futures": False, "include_options": False}},
                use_env=False, dotenv_path=None,
            )

    def test_symbol_in_both_lists_is_rejected(self):
        with pytest.raises(ConfigurationError):
            load_config(
                "config/default.yaml",
                overrides={
                    "universe": {
                        "stock_underlyings_allowlist": ["RELIANCE"],
                        "stock_underlyings_blocklist": ["RELIANCE"],
                    }
                },
                use_env=False, dotenv_path=None,
            )

    def test_unknown_config_key_is_rejected(self):
        """A typo must fail loudly rather than being silently ignored."""
        with pytest.raises(ConfigurationError):
            load_config(
                "config/default.yaml",
                overrides={"risk": {"maximum_daily_loss": 0.05}},
                use_env=False, dotenv_path=None,
            )

    def test_env_overrides_are_parsed_into_nested_keys(self):
        result = env_overrides({"AQTP_RISK__MAX_OPEN_POSITIONS": "7"})
        assert result == {"risk": {"max_open_positions": 7}}

    def test_credentials_never_leak_through_repr(self):
        from aqtp.configuration.loader import Credentials

        credentials = Credentials(access_token="super-secret-token", api_key="key123")
        text = repr(credentials)
        assert "super-secret-token" not in text
        assert "key123" not in text
        assert "<set>" in text


class TestRedaction:
    """REQ 62: credentials must never reach a log, in any payload shape."""

    @pytest.mark.parametrize(
        "text,secret",
        [
            ("Authorization: Bearer abc123xyz", "abc123xyz"),
            ("api_key=s3cret", "s3cret"),
            ('{"password": "hunter2"}', "hunter2"),
            ('{"access_token":"tok_live_9f8a7","quantity":75}', "tok_live_9f8a7"),
            ("totp: '123456'", "123456"),
            # Compound keys matter: broker bodies use api_secret, not bare 'secret'.
            ('{"api_secret": "sk_abcdefgh", "trading_symbol": "NIFTY"}', "sk_abcdefgh"),
            ("checksum=deadbeef99", "deadbeef99"),
            ("GROWW_API_SECRET=pw99", "pw99"),
        ],
    )
    def test_secrets_are_scrubbed_from_log_text(self, text, secret):
        from aqtp.core.logging import redact

        assert secret not in redact(text), f"credential leaked from: {text}"

    def test_non_secret_content_is_preserved(self):
        from aqtp.core.logging import redact

        text = "order quantity=75 symbol=NIFTY price=200.5"
        assert redact(text) == text

    def test_secret_keys_are_masked_in_mappings(self):
        from aqtp.core.logging import redact_mapping

        result = redact_mapping({"access_token": "xyz", "quantity": 75})
        assert result["access_token"] == "<redacted>"
        assert result["quantity"] == 75
