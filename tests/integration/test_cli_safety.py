"""CLI-level guarantees around LIVE mode and the trade commands.

Two things are worth pinning here. First, the shipped config declares LIVE but
the environment still decides: nothing that places an order loads without the
interlocks. Second, the commands you need *when you are not armed* — live-status,
risk, status — must still run, or the operator is locked out of their own
diagnostics by the safety mechanism.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from aqtp.cli.main import app
from aqtp.configuration.schema import LIVE_INTERLOCKS

runner = CliRunner()

READ_ONLY_COMMANDS = ["live-status", "risk", "strategies", "arm"]


@pytest.fixture
def disarmed(monkeypatch):
    for key in LIVE_INTERLOCKS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def armed(monkeypatch):
    for key, value in LIVE_INTERLOCKS.items():
        monkeypatch.setenv(key, value)


class TestLiveGating:
    def test_the_shipped_config_declares_live(self):
        import yaml

        with open("config/default.yaml") as handle:
            assert yaml.safe_load(handle)["mode"] == "LIVE"

    def test_validating_the_config_fails_when_disarmed(self, disarmed):
        result = runner.invoke(app, ["config", "--validate"])
        assert result.exit_code != 0
        assert "interlock" in result.output

    def test_validating_the_config_succeeds_when_armed(self, armed):
        result = runner.invoke(app, ["config", "--validate"])
        assert result.exit_code == 0
        assert "LIVE" in result.output

    def test_paper_is_reachable_without_interlocks(self, disarmed):
        result = runner.invoke(app, ["config", "--validate", "--mode", "PAPER"])
        assert result.exit_code == 0
        assert "PAPER" in result.output

    def test_live_status_reports_missing_interlocks(self, disarmed):
        result = runner.invoke(app, ["live-status"])
        assert result.exit_code == 0
        assert "blocked" in result.output
        for key in LIVE_INTERLOCKS:
            assert key in result.output

    def test_live_status_reports_armed(self, armed):
        result = runner.invoke(app, ["live-status"])
        assert result.exit_code == 0
        assert "ARMED" in result.output

    def test_arm_prints_every_interlock(self, disarmed):
        result = runner.invoke(app, ["arm"])
        assert result.exit_code == 0
        for key, value in LIVE_INTERLOCKS.items():
            assert key in result.output
            assert value in result.output

    def test_arm_does_not_set_anything_itself(self, disarmed, monkeypatch):
        import os

        runner.invoke(app, ["arm"])
        # It prints the exports; it must not perform them.
        assert not any(os.environ.get(key) for key in LIVE_INTERLOCKS)


class TestDisarmedDiagnostics:
    """Being unarmed must not lock the operator out of read-only commands."""

    @pytest.mark.parametrize("command", READ_ONLY_COMMANDS)
    def test_read_only_commands_run_when_disarmed(self, disarmed, command):
        result = runner.invoke(app, [command])
        assert result.exit_code == 0, f"{command} failed: {result.output}"

    def test_risk_shows_the_configured_limits_when_disarmed(self, disarmed):
        result = runner.invoke(app, ["risk"])
        assert result.exit_code == 0
        assert "risk per trade" in result.output


class TestCommandSurface:
    def test_every_documented_trade_command_exists(self):
        result = runner.invoke(app, ["trade", "--help"])
        assert result.exit_code == 0
        for command in ("buy", "sell", "exit", "close-all", "cancel", "modify",
                        "set-stop", "quote"):
            assert command in result.output

    def test_the_production_entry_points_exist(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("start", "stop", "arm", "analyze", "market", "universe",
                        "trade", "research"):
            assert command in result.output

    def test_buy_exposes_the_documented_options(self):
        result = runner.invoke(app, ["trade", "buy", "--help"])
        assert result.exit_code == 0
        for option in ("--option", "--strike", "--expiry", "--lots", "--quantity",
                       "--stop", "--target", "--force", "--dry-run", "--yes"):
            assert option in result.output
