"""Tests for the achievements evaluator.

Each test covers one condition type with mock stats — verifies both the
locked and unlocked paths. Plus end-to-end evaluation against the sample
audit-log fixture.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pythia_observability.achievements import (
    AchievementsEvaluator,
    eval_brier_score,
    eval_drawdown_pct,
    eval_realized_pnl_usd,
    eval_trade_count,
    eval_win_count,
    eval_win_streak,
    eval_wins_in_category,
)
from pythia_observability.audit_reader import AuditLogReader
from pythia_observability.types import AchievementCondition

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_LOG = FIXTURES / "sample_audit.jsonl"
SAMPLE_ACH = FIXTURES / "sample_achievements.toml"

def _cond(**kwargs) -> AchievementCondition:
    """Build an AchievementCondition with sensible defaults."""
    defaults = {"type": "trade_count", "op": ">=", "value": 1}
    defaults.update(kwargs)
    return AchievementCondition.model_validate(defaults)

# --------------------------------------------------------------------------- #
# Per-condition evaluators.
# --------------------------------------------------------------------------- #

class TestEvalTradeCount:
    def test_unlocked_when_meeting_threshold(self) -> None:
        cond = _cond(type="trade_count", op=">=", value=10)
        unlocked, val = eval_trade_count({"total_trades": 15}, cond)
        assert unlocked is True
        assert val == 15

    def test_locked_when_below_threshold(self) -> None:
        cond = _cond(type="trade_count", op=">=", value=10)
        unlocked, val = eval_trade_count({"total_trades": 7}, cond)
        assert unlocked is False
        assert val == 7

    def test_strict_greater_than(self) -> None:
        cond = _cond(type="trade_count", op=">", value=10)
        assert eval_trade_count({"total_trades": 10}, cond)[0] is False
        assert eval_trade_count({"total_trades": 11}, cond)[0] is True

class TestEvalWinCount:
    def test_unlocked(self) -> None:
        cond = _cond(type="win_count", op=">=", value=1)
        unlocked, val = eval_win_count({"winning_trades": 3}, cond)
        assert unlocked is True
        assert val == 3

    def test_locked_at_zero(self) -> None:
        cond = _cond(type="win_count", op=">=", value=1)
        unlocked, val = eval_win_count({"winning_trades": 0}, cond)
        assert unlocked is False
        assert val == 0

class TestEvalWinStreak:
    def _entry(self, pnl: float | None) -> SimpleNamespace:
        return SimpleNamespace(realized_pnl_usd=pnl)

    def test_unlocked_long_streak(self) -> None:
        cond = _cond(type="win_streak", op=">=", value=3)
        entries = [
            self._entry(10.0), self._entry(20.0), self._entry(30.0),
            self._entry(-5.0), self._entry(15.0),
        ]
        unlocked, val = eval_win_streak({"_entries": entries}, cond)
        assert unlocked is True
        assert val == 3

    def test_locked_short_streak(self) -> None:
        cond = _cond(type="win_streak", op=">=", value=3)
        entries = [self._entry(10.0), self._entry(-5.0), self._entry(20.0)]
        unlocked, val = eval_win_streak({"_entries": entries}, cond)
        assert unlocked is False
        assert val == 1

    def test_skip_unsettled_in_streak(self) -> None:
        """Unsettled trades (pnl=None) should not break or extend the streak."""
        cond = _cond(type="win_streak", op=">=", value=2)
        entries = [self._entry(10.0), self._entry(None), self._entry(20.0)]
        unlocked, val = eval_win_streak({"_entries": entries}, cond)
        assert unlocked is True
        assert val == 2

    def test_empty_entries(self) -> None:
        cond = _cond(type="win_streak", op=">=", value=1)
        unlocked, val = eval_win_streak({"_entries": []}, cond)
        assert unlocked is False
        assert val == 0

    def test_missing_entries_key(self) -> None:
        cond = _cond(type="win_streak", op=">=", value=1)
        unlocked, val = eval_win_streak({}, cond)
        assert unlocked is False
        assert val == 0

class TestEvalRealizedPnL:
    def test_unlocked_at_threshold(self) -> None:
        cond = _cond(type="realized_pnl_usd", op=">=", value=50)
        unlocked, val = eval_realized_pnl_usd({"total_realized_pnl_usd": 50.0}, cond)
        assert unlocked is True
        assert val == 50.0

    def test_unlocked_above_threshold(self) -> None:
        cond = _cond(type="realized_pnl_usd", op=">=", value=50)
        unlocked, val = eval_realized_pnl_usd({"total_realized_pnl_usd": 127.42}, cond)
        assert unlocked is True
        assert val == 127.42

    def test_locked_below_threshold(self) -> None:
        cond = _cond(type="realized_pnl_usd", op=">=", value=100)
        unlocked, val = eval_realized_pnl_usd({"total_realized_pnl_usd": 42.0}, cond)
        assert unlocked is False
        assert val == 42.0

    def test_negative_pnl(self) -> None:
        cond = _cond(type="realized_pnl_usd", op=">=", value=0)
        unlocked, val = eval_realized_pnl_usd({"total_realized_pnl_usd": -15.0}, cond)
        assert unlocked is False
        assert val == -15.0

class TestEvalBrierScore:
    def test_unlocked_under_threshold_with_min_trades_met(self) -> None:
        cond = _cond(type="brier_score", op="<=", value=0.20, min_trades=5)
        stats = {"settled_trades": 10, "per_analyst_brier": {"a": 0.15, "b": 0.25}}
        unlocked, val = eval_brier_score(stats, cond)
        assert unlocked is True
        assert val == 0.15  # best (lowest)

    def test_locked_when_below_min_trades(self) -> None:
        cond = _cond(type="brier_score", op="<=", value=0.20, min_trades=20)
        stats = {"settled_trades": 5, "per_analyst_brier": {"a": 0.15}}
        unlocked, val = eval_brier_score(stats, cond)
        assert unlocked is False
        assert val is None

    def test_locked_when_score_above_threshold(self) -> None:
        cond = _cond(type="brier_score", op="<=", value=0.20)
        stats = {"settled_trades": 10, "per_analyst_brier": {"a": 0.35}}
        unlocked, val = eval_brier_score(stats, cond)
        assert unlocked is False
        assert val == 0.35

    def test_locked_when_no_brier_scores(self) -> None:
        cond = _cond(type="brier_score", op="<=", value=0.20)
        unlocked, val = eval_brier_score({"settled_trades": 0, "per_analyst_brier": {}}, cond)
        assert unlocked is False
        assert val is None

    def test_no_min_trades_constraint(self) -> None:
        """min_trades defaults to None -> 0, so any settled trade can unlock."""
        cond = _cond(type="brier_score", op="<=", value=0.20)
        stats = {"settled_trades": 1, "per_analyst_brier": {"a": 0.10}}
        unlocked, val = eval_brier_score(stats, cond)
        assert unlocked is True
        assert val == 0.10

class TestEvalDrawdownPct:
    def test_unlocked_when_under_threshold(self) -> None:
        cond = _cond(type="drawdown_pct", op="<=", value=5)
        unlocked, val = eval_drawdown_pct({"current_drawdown_pct": 2.5}, cond)
        assert unlocked is True
        assert val == 2.5

    def test_locked_when_above_threshold(self) -> None:
        cond = _cond(type="drawdown_pct", op="<=", value=5)
        unlocked, val = eval_drawdown_pct({"current_drawdown_pct": 8.0}, cond)
        assert unlocked is False
        assert val == 8.0

    def test_unlocked_at_zero_drawdown(self) -> None:
        cond = _cond(type="drawdown_pct", op="<=", value=5)
        unlocked, val = eval_drawdown_pct({"current_drawdown_pct": 0.0}, cond)
        assert unlocked is True
        assert val == 0.0

class TestEvalWinsInCategory:
    def _entry(self, pnl: float | None, category: str | None) -> SimpleNamespace:
        return SimpleNamespace(realized_pnl_usd=pnl, category=category)

    def test_unlocked_with_enough_wins(self) -> None:
        cond = _cond(type="wins_in_category", op=">=", value=3, category="subjective")
        entries = [
            self._entry(10.0, "subjective"),
            self._entry(20.0, "subjective"),
            self._entry(30.0, "subjective"),
            self._entry(-5.0, "politics"),  # losing, different cat
        ]
        unlocked, val = eval_wins_in_category({"_entries": entries}, cond)
        assert unlocked is True
        assert val == 3

    def test_locked_with_too_few_wins(self) -> None:
        cond = _cond(type="wins_in_category", op=">=", value=3, category="subjective")
        entries = [
            self._entry(10.0, "subjective"),
            self._entry(20.0, "politics"),
        ]
        unlocked, val = eval_wins_in_category({"_entries": entries}, cond)
        assert unlocked is False
        assert val == 1

    def test_locked_when_no_category_on_condition(self) -> None:
        cond = _cond(type="wins_in_category", op=">=", value=1)  # no category
        unlocked, val = eval_wins_in_category(
            {"_entries": [self._entry(10.0, "subjective")]}, cond
        )
        assert unlocked is False
        assert val == 0

    def test_case_insensitive_category_match(self) -> None:
        cond = _cond(type="wins_in_category", op=">=", value=1, category="Subjective")
        entries = [self._entry(10.0, "SUBJECTIVE")]
        unlocked, val = eval_wins_in_category({"_entries": entries}, cond)
        assert unlocked is True
        assert val == 1

    def test_losing_trades_excluded(self) -> None:
        cond = _cond(type="wins_in_category", op=">=", value=1, category="subjective")
        entries = [self._entry(-5.0, "subjective")]
        unlocked, val = eval_wins_in_category({"_entries": entries}, cond)
        assert unlocked is False
        assert val == 0

class TestUnsupportedOperator:
    def test_raises_on_unknown_op(self) -> None:
        cond = _cond(type="trade_count", op="~=", value=10)
        with pytest.raises(ValueError, match="Unsupported operator"):
            eval_trade_count({"total_trades": 10}, cond)

    def test_raises_on_non_numeric_threshold(self) -> None:
        cond = _cond(type="trade_count", op=">=", value="ten")
        with pytest.raises(ValueError, match="not numeric"):
            eval_trade_count({"total_trades": 10}, cond)

# --------------------------------------------------------------------------- #
# AchievementsEvaluator (end-to-end against the fixture).
# --------------------------------------------------------------------------- #

class TestAchievementsEvaluator:
    def test_loads_all_achievements_from_toml(self) -> None:
        ev = AchievementsEvaluator(SAMPLE_ACH)
        # The sample achievements.toml has 9 entries.
        assert len(ev.achievements) == 9
        ids = {a.id for a in ev.achievements}
        assert "first_trade" in ids
        assert "calibrated" in ids
        assert "niche_master" in ids

    def test_evaluate_returns_achievement_objects(self) -> None:
        ev = AchievementsEvaluator(SAMPLE_ACH)
        reader = AuditLogReader(SAMPLE_LOG)
        results = ev.evaluate(reader)
        assert len(results) == 9
        assert all(hasattr(a, "unlocked_at") for a in results)

    def test_evaluate_does_not_mutate_internal_state(self) -> None:
        """Calling evaluate() twice should return equivalent results
        (the evaluator's internal achievements list isn't mutated)."""
        ev = AchievementsEvaluator(SAMPLE_ACH)
        reader = AuditLogReader(SAMPLE_LOG)
        r1 = ev.evaluate(reader)
        r2 = ev.evaluate(reader)
        assert len(r1) == len(r2)
        # The unlocked_at fields should match (both set to "now", which may
        # differ by microseconds — compare just the unlocked state).
        for a, b in zip(r1, r2):
            assert (a.unlocked_at is not None) == (b.unlocked_at is not None)

    def test_first_trade_unlocked(self) -> None:
        ev = AchievementsEvaluator(SAMPLE_ACH)
        reader = AuditLogReader(SAMPLE_LOG)
        results = {a.id: a for a in ev.evaluate(reader)}
        # 8 entries in the sample log; threshold is 1.
        assert results["first_trade"].unlocked_at is not None
        assert results["first_trade"].unlocked_value == 8

    def test_ten_trades_locked(self) -> None:
        ev = AchievementsEvaluator(SAMPLE_ACH)
        reader = AuditLogReader(SAMPLE_LOG)
        results = {a.id: a for a in ev.evaluate(reader)}
        # Only 8 entries; threshold is 10.
        assert results["ten_trades"].unlocked_at is None

    def test_first_win_unlocked(self) -> None:
        ev = AchievementsEvaluator(SAMPLE_ACH)
        reader = AuditLogReader(SAMPLE_LOG)
        results = {a.id: a for a in ev.evaluate(reader)}
        assert results["first_win"].unlocked_at is not None
        assert results["first_win"].unlocked_value == 4

    def test_streak_3_locked(self) -> None:
        """Sample log's longest win streak is 2 (entries 5, 6 — but entry 5 is
        paper, which still counts as settled). Threshold is 3 → locked."""
        ev = AchievementsEvaluator(SAMPLE_ACH)
        reader = AuditLogReader(SAMPLE_LOG)
        results = {a.id: a for a in ev.evaluate(reader)}
        assert results["streak_3"].unlocked_at is None
        # Streak of 2: entries 1, 2 won, entry 3 lost, then entries 5, 6 won.
        assert results["streak_3"].unlocked_value == 2

    def test_pnl_plus_50_unlocked(self) -> None:
        ev = AchievementsEvaluator(SAMPLE_ACH)
        reader = AuditLogReader(SAMPLE_LOG)
        results = {a.id: a for a in ev.evaluate(reader)}
        # Total realized P&L is ~80.41.
        assert results["pnl_plus_50"].unlocked_at is not None
        assert results["pnl_plus_50"].unlocked_value == pytest.approx(80.41, abs=0.01)

    def test_pnl_plus_100_locked(self) -> None:
        ev = AchievementsEvaluator(SAMPLE_ACH)
        reader = AuditLogReader(SAMPLE_LOG)
        results = {a.id: a for a in ev.evaluate(reader)}
        assert results["pnl_plus_100"].unlocked_at is None

    def test_calibrated_locked_due_to_min_trades(self) -> None:
        """Brier threshold 0.20 with min_trades=20. Sample has only 5 settled
        trades → locked via the min_trades gate, regardless of Brier."""
        ev = AchievementsEvaluator(SAMPLE_ACH)
        reader = AuditLogReader(SAMPLE_LOG)
        results = {a.id: a for a in ev.evaluate(reader)}
        assert results["calibrated"].unlocked_at is None

    def test_niche_master_locked(self) -> None:
        """wins_in_category >= 3 for 'subjective'. Sample has 2 subjective wins
        (entries 5 and 6)."""
        ev = AchievementsEvaluator(SAMPLE_ACH)
        reader = AuditLogReader(SAMPLE_LOG)
        results = {a.id: a for a in ev.evaluate(reader)}
        assert results["niche_master"].unlocked_at is None
        assert results["niche_master"].unlocked_value == 2

    def test_unknown_condition_type_leaves_locked(self, tmp_path: Path) -> None:
        p = tmp_path / "weird.toml"
        p.write_text(
            '[[achievement]]\n'
            'id = "weird"\n'
            'name = "Weird"\n'
            'description = "Unknown condition type."\n'
            'condition = { type = "nonexistent_type", op = ">=", value = 1 }\n',
            encoding="utf-8",
        )
        ev = AchievementsEvaluator(p)
        reader = AuditLogReader(SAMPLE_LOG)
        results = ev.evaluate(reader)
        assert len(results) == 1
        assert results[0].unlocked_at is None

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AchievementsEvaluator(tmp_path / "does_not_exist.toml")

    def test_malformed_config_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.toml"
        # `achievement` is a string, not an array of tables.
        p.write_text('achievement = "not a table"\n', encoding="utf-8")
        with pytest.raises(TypeError, match="must be an array"):
            AchievementsEvaluator(p)

    def test_missing_condition_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "no_cond.toml"
        p.write_text(
            '[[achievement]]\n'
            'id = "x"\n'
            'name = "X"\n'
            'description = "No condition."\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing `condition`"):
            AchievementsEvaluator(p)
