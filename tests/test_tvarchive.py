"""The TradingView archive: merging pulls, contract lists and stitching."""
from datetime import date, datetime

from evotrader import tvarchive
from evotrader.tvarchive import NY


def _ts(y, m, d, hh, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=NY).timestamp())


def test_pull_merges_into_existing_file(tmp_path, monkeypatch):
    batches = iter([
        [(100, 1, 2, 0.5, 1.5, 10), (200, 1, 2, 0.5, 1.6, 10)],
        [(200, 1, 2, 0.5, 1.7, 12), (300, 1, 2, 0.5, 1.8, 10)],
    ])
    monkeypatch.setattr(tvarchive.tv, "fetch_bars", lambda *a, **k: next(batches))
    assert tvarchive.pull("CME_MINI:NQ1!", "5m", str(tmp_path)) == (2, 2)
    assert tvarchive.pull("CME_MINI:NQ1!", "5m", str(tmp_path)) == (2, 3)
    rows = tvarchive.read_series(tvarchive.series_path("CME_MINI:NQ1!", "5m", str(tmp_path)))
    assert sorted(rows) == [100, 200, 300]
    assert rows[200][3] == 1.7  # a later pull overwrites a bar captured while forming


def test_contract_list_is_quarterly_and_stops_near_today():
    codes = [c for c, _ in tvarchive.contracts("NQ", 2025, today=date(2026, 9, 19))]
    assert codes[:4] == ["NQH2025", "NQM2025", "NQU2025", "NQZ2025"]
    assert codes[-1] == "NQH2027"
    assert "NQM2027" not in codes


def test_evening_bars_belong_to_the_next_session():
    assert tvarchive.session_date(_ts(2026, 9, 17, 18, 5)) == date(2026, 9, 18)
    assert tvarchive.session_date(_ts(2026, 9, 18, 9, 30)) == date(2026, 9, 18)


def test_stitch_takes_one_contract_per_session_by_volume():
    day1 = [_ts(2026, 3, 10, 10), _ts(2026, 3, 10, 11)]
    day2 = [_ts(2026, 3, 20, 10)]
    front = {t: (100, 101, 99, 100, 1000) for t in day1 + day2[:0]}
    front[day2[0]] = (100, 101, 99, 100, 5)        # expiring: thin on day 2
    back = {t: (200, 201, 199, 200, 10) for t in day1}
    back[day2[0]] = (200, 201, 199, 200, 900)      # rolled: heavy on day 2
    rows, label = tvarchive.stitch({"NQH": front, "NQM": back})
    assert {label[t] for t in day1} == {"NQH"}
    assert label[day2[0]] == "NQM"
    assert len(rows) == 3


def test_status_reports_series(tmp_path):
    path = tvarchive.series_path("CME_MINI:NQZ2024", "5m", str(tmp_path))
    tvarchive.write_series(path, {1_700_000_000: (1, 2, 0.5, 1.5, 3)})
    rows = tvarchive.status(str(tmp_path))
    assert rows[0]["series"] == "CME_MINI_NQZ2024" and rows[0]["bars"] == 1
    assert "CME_MINI_NQZ2024" in tvarchive.format_status(rows, str(tmp_path))
