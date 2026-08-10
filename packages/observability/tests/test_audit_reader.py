"""Tests for the audit-log reader.

Uses the fixture at `tests/fixtures/sample_audit.jsonl` — 8 example audit
entries covering: a winning YES trade, a winning NO trade, a losing trade, a
skipped trade (agreement below threshold), a paper trade, a settled
subjective-category win, an open (executed but not yet settled) trade, and a
stub-signed skipped trade (drawdown breaker).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pythia_observability.audit_reader import AuditLogReader
from pythia_observability.types import AuditEntry, PnLMilestone

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_LOG = FIXTURES / "sample_audit.jsonl"

# --------------------------------------------------------------------------- #
# Fixture: a temp JSONL file built from a list of dicts.
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_reader() -> AuditLogReader:
    return AuditLogReader(SAMPLE_LOG)

@pytest.fixture
def tmp_audit_log(tmp_path: Path) -> Path:
    """Write a small JSONL file from inline dicts; return its path."""
    entries = [
        {
            "timestamp": "2026-01-10T09:00:00Z",
            "market_id": "mkt_a",
            "estimates": [{"analyst_id": "politics", "probability": 0.7, "confidence": 0.7,
                           "rationale": "ok", "evidence": []}],
            "decision": {"market_id": "mkt_a", "consensus_prob": 0.7, "agreement_score": 0.85,
                         "gate": "trade", "contributor_ids": ["politics"], "method": "logit-mean",
                         "market_category": "politics"},
            "plan": {"market_id": "mkt_a", "side": "YES", "size_usd": 50.0,
                     "limit_price": 0.6, "decision": "APPROVE", "bankroll_before": 1000.0},
            "receipt": {"market_id": "mkt_a", "side": "YES", "size_usd": 50.0,
                        "fill_price": 0.6, "att_order_id": "ord_1", "signed_by": "k",
                        "mode": "live",
                        "settlement": {"outcome": "YES", "realized_pnl_usd": 33.33}},
            "skipped_reason": None,
            "signature": "ed25519:deadbeef",
        },
        {
            "timestamp": "2026-01-11T09:00:00Z",
            "market_id": "mkt_b",
            "estimates": [],
            "decision": {"gate": "skip", "market_category": "crypto"},
            "plan": {"side": "NO", "size_usd": 0.0, "decision": "REJECT"},
            "receipt": None,
            "skipped_reason": "agreement_below_threshold",
            "signature": "stub:sha256:abc",
        },
    ]
    p = tmp_path / "audit.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return p

# --------------------------------------------------------------------------- #
# iter_entries / read_all.
# --------------------------------------------------------------------------- #

class TestIterEntries:
    def test_iter_returns_audit_entry_instances(self, sample_reader: AuditLogReader) -> None:
        entries = list(sample_reader.iter_entries())
        assert len(entries) == 8
        assert all(isinstance(e, AuditEntry) for e in entries)

    def test_iter_is_lazy_generator(self, sample_reader: AuditLogReader) -> None:
        gen = sample_reader.iter_entries()
        first = next(gen)
        assert isinstance(first, AuditEntry)
        assert first.market_id == "dphi_01J7Q-sample-winner"

    def test_iter_skips_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "with_blanks.jsonl"
        p.write_text(
            json.dumps({"timestamp": "2026-01-10T09:00:00Z", "market_id": "a",
                        "estimates": [], "decision": {}, "plan": {}}) + "\n\n\n"
            + json.dumps({"timestamp": "2026-01-11T09:00:00Z", "market_id": "b",
                          "estimates": [], "decision": {}, "plan": {}}) + "\n",
            encoding="utf-8",
        )
        reader = AuditLogReader(p)
        entries = list(reader.iter_entries())
        assert len(entries) == 2

    def test_iter_skips_malformed_json(self, tmp_path: Path) -> None:
        p = tmp_path / "malformed.jsonl"
        p.write_text(
            json.dumps({"timestamp": "2026-01-10T09:00:00Z", "market_id": "a",
                        "estimates": [], "decision": {}, "plan": {}}) + "\n"
            + "not valid json\n"
            + json.dumps({"timestamp": "2026-01-11T09:00:00Z", "market_id": "b",
                          "estimates": [], "decision": {}, "plan": {}}) + "\n",
            encoding="utf-8",
        )
        reader = AuditLogReader(p)
        entries = list(reader.iter_entries())
        assert len(entries) == 2  # malformed middle line skipped

class TestReadAll:
    def test_read_all_returns_list(self, sample_reader: AuditLogReader) -> None:
        entries = sample_reader.read_all()
        assert isinstance(entries, list)
        assert len(entries) == 8

    def test_read_all_caches(self, sample_reader: AuditLogReader) -> None:
        e1 = sample_reader.read_all()
        e2 = sample_reader.read_all()
        # Cache should return the *same* list object on the second call.
        assert e1 is e2

    def test_invalidate_cache(self, sample_reader: AuditLogReader) -> None:
        e1 = sample_reader.read_all()
        sample_reader.invalidate_cache()
        e2 = sample_reader.read_all()
        assert e1 is not e2

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AuditLogReader(tmp_path / "does_not_exist.jsonl")

# --------------------------------------------------------------------------- #
# Slicing.
# --------------------------------------------------------------------------- #

class TestSlicing:
    def test_get_by_market_returns_only_matching(self, sample_reader: AuditLogReader) -> None:
        entries = sample_reader.get_by_market("dphi_01J7Q-sample-winner")
        assert len(entries) == 1
        assert entries[0].market_id == "dphi_01J7Q-sample-winner"

    def test_get_by_market_unknown_returns_empty(self, sample_reader: AuditLogReader) -> None:
        assert sample_reader.get_by_market("does_not_exist") == []

    def test_filter_by_time_inclusive(self, sample_reader: AuditLogReader) -> None:
        start = datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 1, 15, 23, 59, 59, tzinfo=UTC)
        entries = sample_reader.filter_by_time(start, end)
        # Entries 3, 4, 5 fall in [Jan 12 00:00, Jan 15 23:59].
        ids = [e.market_id for e in entries]
        assert "dphi_01J9L-lakers-playoffs" in ids  # Jan 12
        assert "dphi_01JA3-fed-cuts-q1" in ids       # Jan 13
        assert "dphi_01JB5-best-album-2026" in ids   # Jan 14
        assert "dphi_01JC7-meme-coin-flips-first" in ids  # Jan 15
        assert len(entries) == 4

    def test_filter_by_time_naive_datetime_treated_as_utc(
        self, sample_reader: AuditLogReader
    ) -> None:
        # Naive datetime should be treated as UTC, not raise.
        start = datetime(2026, 1, 10, 0, 0, 0)  # noqa: DTZ001 - deliberately naive for the test
        end = datetime(2026, 1, 11, 23, 59, 59)  # noqa: DTZ001 - deliberately naive for the test
        entries = sample_reader.filter_by_time(start, end)
        assert len(entries) == 2  # Jan 10 + Jan 11

    def test_filter_by_time_no_matches(self, sample_reader: AuditLogReader) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 1, 31, tzinfo=UTC)
        assert sample_reader.filter_by_time(start, end) == []

# --------------------------------------------------------------------------- #
# compute_pnl_series.
# --------------------------------------------------------------------------- #

class TestPnLSeries:
    def test_pnl_series_returns_milestones(self, sample_reader: AuditLogReader) -> None:
        series = sample_reader.compute_pnl_series()
        assert len(series) > 0
        assert all(isinstance(m, PnLMilestone) for m in series)
        # Sample has 5 settled trades (entries 1, 2, 3, 5, 6 — entry 7 has no
        # settlement, entries 4 and 8 were skipped).
        assert len(series) == 5

    def test_pnl_series_starts_at_initial_bankroll(
        self, sample_reader: AuditLogReader
    ) -> None:
        series = sample_reader.compute_pnl_series()
        # First entry's plan.bankroll_before is 1000.0.
        assert series[0].bankroll_usd == pytest.approx(1000.0 + 26.92, abs=0.01)

    def test_pnl_series_cumulative_monotonic_on_pnl(
        self, sample_reader: AuditLogReader
    ) -> None:
        series = sample_reader.compute_pnl_series()
        # cumulative realized_pnl should equal the sum of settled trades so far.
        expected_cumulative = [26.92, 26.92 + 26.67, 26.92 + 26.67 - 35.00,
                               26.92 + 26.67 - 35.00 + 25.00,
                               26.92 + 26.67 - 35.00 + 25.00 + 36.82]
        for actual, expected in zip(series, expected_cumulative):
            assert actual.realized_pnl_usd == pytest.approx(expected, abs=0.01)

    def test_pnl_series_bankroll_tracks_cumulative_pnl(
        self, sample_reader: AuditLogReader
    ) -> None:
        series = sample_reader.compute_pnl_series()
        for m in series:
            # bankroll = 1000 + cumulative_pnl (no fees modelled).
            assert m.bankroll_usd == pytest.approx(1000.0 + m.realized_pnl_usd, abs=0.01)

    def test_pnl_series_drawdown_nonnegative(self, sample_reader: AuditLogReader) -> None:
        series = sample_reader.compute_pnl_series()
        for m in series:
            assert m.drawdown_pct >= 0.0

    def test_pnl_series_drawdown_after_loss(self, sample_reader: AuditLogReader) -> None:
        series = sample_reader.compute_pnl_series()
        # After the -$35 loss (3rd milestone), bankroll drops from peak 1053.59
        # to 1018.59. Drawdown = (1053.59 - 1018.59) / 1053.59 * 100 ≈ 3.32%.
        assert series[2].drawdown_pct == pytest.approx(3.32, abs=0.1)

    def test_pnl_series_empty_log(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert AuditLogReader(p).compute_pnl_series() == []

# --------------------------------------------------------------------------- #
# compute_brier_scores.
# --------------------------------------------------------------------------- #

class TestBrierScores:
    def test_brier_returns_dict(self, sample_reader: AuditLogReader) -> None:
        scores = sample_reader.compute_brier_scores()
        assert isinstance(scores, dict)
        # Sample fixture has politics, crypto, niche, sports analysts.
        assert "politics" in scores
        assert "crypto" in scores
        assert "niche" in scores

    def test_brier_scores_in_valid_range(self, sample_reader: AuditLogReader) -> None:
        scores = sample_reader.compute_brier_scores()
        for analyst_id, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{analyst_id}: {score} out of [0, 1]"

    def test_brier_perfect_analyst_zero(self, tmp_path: Path) -> None:
        """An analyst who always predicts the correct outcome with p=1.0
        should get Brier = 0.0."""
        p = tmp_path / "perfect.jsonl"
        entries = []
        for i, outcome in enumerate(["YES", "NO", "YES"]):
            entries.append({
                "timestamp": f"2026-01-1{i}T09:00:00Z",
                "market_id": f"mkt_{i}",
                "estimates": [{"analyst_id": "oracle", "probability": 1.0 if outcome == "YES" else 0.0,
                               "confidence": 1.0, "rationale": "perfect", "evidence": []}],
                "decision": {}, "plan": {},
                "receipt": {"settlement": {"outcome": outcome, "realized_pnl_usd": 1.0}},
                "skipped_reason": None, "signature": "stub:sha256:x",
            })
        p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
        scores = AuditLogReader(p).compute_brier_scores()
        assert scores["oracle"] == 0.0

    def test_brier_uninformative_analyst_quarter(self, tmp_path: Path) -> None:
        """An analyst who always predicts p=0.5 gets Brier = 0.25 (uninformative)."""
        p = tmp_path / "uninformative.jsonl"
        entries = []
        for i, outcome in enumerate(["YES", "NO"]):
            entries.append({
                "timestamp": f"2026-01-1{i}T09:00:00Z",
                "market_id": f"mkt_{i}",
                "estimates": [{"analyst_id": "coinflip", "probability": 0.5,
                               "confidence": 0.0, "rationale": "idk", "evidence": []}],
                "decision": {}, "plan": {},
                "receipt": {"settlement": {"outcome": outcome, "realized_pnl_usd": 1.0}},
                "skipped_reason": None, "signature": "stub:sha256:x",
            })
        p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
        scores = AuditLogReader(p).compute_brier_scores()
        assert scores["coinflip"] == 0.25

    def test_brier_no_settled_markets(self, tmp_path: Path) -> None:
        p = tmp_path / "unsettled.jsonl"
        p.write_text(
            json.dumps({"timestamp": "2026-01-10T09:00:00Z", "market_id": "mkt_x",
                        "estimates": [{"analyst_id": "a", "probability": 0.7,
                                       "confidence": 0.7, "rationale": "x", "evidence": []}],
                        "decision": {}, "plan": {}, "receipt": None,
                        "skipped_reason": "agreement_below_threshold",
                        "signature": "stub:sha256:x"}) + "\n",
            encoding="utf-8",
        )
        assert AuditLogReader(p).compute_brier_scores() == {}

# --------------------------------------------------------------------------- #
# compute_stats.
# --------------------------------------------------------------------------- #

class TestComputeStats:
    def test_stats_returns_well_formed_dict(self, sample_reader: AuditLogReader) -> None:
        stats = sample_reader.compute_stats()
        required_keys = {
            "total_trades", "executed_trades", "skipped_trades", "paper_trades",
            "settled_trades", "winning_trades", "losing_trades", "win_rate",
            "avg_stake_usd", "total_realized_pnl_usd", "current_bankroll_usd",
            "peak_bankroll_usd", "current_drawdown_pct", "per_analyst_brier",
            "skipped_reasons", "signature_stub_count",
        }
        assert required_keys.issubset(stats.keys())

    def test_stats_counts(self, sample_reader: AuditLogReader) -> None:
        stats = sample_reader.compute_stats()
        assert stats["total_trades"] == 8
        assert stats["executed_trades"] == 6   # entries 1, 2, 3, 5, 6, 7
        assert stats["skipped_trades"] == 2    # entries 4, 8
        assert stats["paper_trades"] == 1      # entry 5
        assert stats["settled_trades"] == 5    # entries 1, 2, 3, 5, 6
        assert stats["winning_trades"] == 4    # entries 1, 2, 5, 6
        assert stats["losing_trades"] == 1     # entry 3

    def test_stats_win_rate(self, sample_reader: AuditLogReader) -> None:
        stats = sample_reader.compute_stats()
        assert stats["win_rate"] == pytest.approx(4 / 5, abs=0.001)

    def test_stats_total_pnl(self, sample_reader: AuditLogReader) -> None:
        stats = sample_reader.compute_stats()
        # 26.92 + 26.67 - 35.00 + 25.00 + 36.82 = 80.41
        assert stats["total_realized_pnl_usd"] == pytest.approx(80.41, abs=0.01)

    def test_stats_current_bankroll(self, sample_reader: AuditLogReader) -> None:
        stats = sample_reader.compute_stats()
        # 1000 + 80.41 = 1080.41
        assert stats["current_bankroll_usd"] == pytest.approx(1080.41, abs=0.01)

    def test_stats_peak_bankroll(self, sample_reader: AuditLogReader) -> None:
        stats = sample_reader.compute_stats()
        # Peak occurs at the final settled milestone (entry 6):
        # 1000 + 26.92 + 26.67 - 35.00 + 25.00 + 36.82 = 1080.41
        assert stats["peak_bankroll_usd"] == pytest.approx(1080.41, abs=0.01)

    def test_stats_skipped_reasons(self, sample_reader: AuditLogReader) -> None:
        stats = sample_reader.compute_stats()
        assert stats["skipped_reasons"] == {
            "agreement_below_threshold": 1,
            "drawdown_breaker": 1,
        }

    def test_stats_signature_stub_count(self, sample_reader: AuditLogReader) -> None:
        stats = sample_reader.compute_stats()
        # Only entry 8 has a stub signature.
        assert stats["signature_stub_count"] == 1

    def test_stats_avg_stake_usd(self, sample_reader: AuditLogReader) -> None:
        stats = sample_reader.compute_stats()
        # Executed trades: 50 + 40 + 35 + 25 + 45 + 60 = 255, avg = 42.5
        assert stats["avg_stake_usd"] == pytest.approx(42.5, abs=0.01)

    def test_stats_empty_log(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        stats = AuditLogReader(p).compute_stats()
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["total_realized_pnl_usd"] == 0.0
        # Default bankroll when no bankroll_before is found anywhere.
        assert stats["current_bankroll_usd"] == 1000.0
        assert stats["current_drawdown_pct"] == 0.0

    def test_stats_uses_first_bankroll_before(
        self, tmp_audit_log: Path
    ) -> None:
        stats = AuditLogReader(tmp_audit_log).compute_stats()
        # First entry's bankroll_before is 1000.0.
        assert stats["current_bankroll_usd"] == pytest.approx(1000.0 + 33.33, abs=0.01)

# --------------------------------------------------------------------------- #
# AuditEntry convenience properties.
# --------------------------------------------------------------------------- #

class TestAuditEntryProperties:
    def test_is_executed(self, sample_reader: AuditLogReader) -> None:
        entries = sample_reader.read_all()
        # Entry 1 has a receipt, entry 4 doesn't.
        assert entries[0].is_executed is True
        assert entries[3].is_executed is False

    def test_is_skipped(self, sample_reader: AuditLogReader) -> None:
        entries = sample_reader.read_all()
        assert entries[3].is_skipped is True
        assert entries[0].is_skipped is False

    def test_is_paper_via_receipt_mode(self, sample_reader: AuditLogReader) -> None:
        entries = sample_reader.read_all()
        # Entry 5 (index 4) is paper.
        assert entries[4].is_paper is True
        assert entries[0].is_paper is False

    def test_is_paper_via_signature_prefix(self, sample_reader: AuditLogReader) -> None:
        entries = sample_reader.read_all()
        # Entry 5 also has signature starting with "paper:".
        assert entries[4].signature.startswith("paper:")

    def test_realized_pnl_usd(self, sample_reader: AuditLogReader) -> None:
        entries = sample_reader.read_all()
        assert entries[0].realized_pnl_usd == pytest.approx(26.92, abs=0.01)
        assert entries[2].realized_pnl_usd == pytest.approx(-35.00, abs=0.01)
        # Skipped trade has no P&L.
        assert entries[3].realized_pnl_usd is None
        # Open trade has no settlement yet.
        assert entries[6].realized_pnl_usd is None

    def test_won(self, sample_reader: AuditLogReader) -> None:
        entries = sample_reader.read_all()
        assert entries[0].won is True
        assert entries[2].won is False
        assert entries[3].won is None  # skipped
        assert entries[6].won is None  # open

    def test_category(self, sample_reader: AuditLogReader) -> None:
        entries = sample_reader.read_all()
        assert entries[0].category == "politics"
        assert entries[1].category == "crypto"
        assert entries[2].category == "sports"
        assert entries[4].category == "subjective"  # paper best-album

    def test_extra_fields_allowed(self) -> None:
        """AuditEntry has extra='allow' so upstream-added fields don't break."""
        e = AuditEntry.model_validate({
            "timestamp": "2026-01-10T09:00:00Z",
            "market_id": "x",
            "estimates": [],
            "decision": {},
            "plan": {},
            "upstream_extra_field": "anything",
        })
        assert e.market_id == "x"
