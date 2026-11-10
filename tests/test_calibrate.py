"""`niche-miner calibrate`: the 90th percentile of cluster frequency sums, written back into
settings.yaml without losing comments, and only once unless forced."""

from datetime import date

import pytest
from helpers import seed_cluster, seed_pain

from niche_miner import db
from niche_miner.calibrate import CalibrationError, calibrate, percentile, write_ceiling
from niche_miner.score import compute_stats
from niche_miner.settings import ScoreWeights, load_settings

W = ScoreWeights()


def test_percentile_interpolates_like_numpy():
    assert percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)
    assert percentile([7], 90) == 7
    with pytest.raises(CalibrationError):
        percentile([], 90)


def test_calibrate_uses_merged_groups_and_skips_noise(conn, tmp_path):
    a = [seed_pain(conn, i, author=f"a{i}") for i in range(1, 4)]
    b = [seed_pain(conn, i, author=f"b{i}") for i in range(4, 6)]
    c = [seed_pain(conn, i, author=f"c{i}") for i in range(6, 12)]
    noise = [seed_pain(conn, 12, author="n")]
    ca, cb, cc = seed_cluster(conn, a), seed_cluster(conn, b), seed_cluster(conn, c)
    seed_cluster(conn, noise, noise=True)
    (tmp_path / "merges.yaml").write_text(f"merges:\n  {ca}: [{cb}]\n")

    cal = calibrate(conn, tmp_path, W)

    merged = compute_stats(db.member_details(conn, [ca, cb]), W).url_spread
    alone = compute_stats(db.member_details(conn, [cc]), W).url_spread
    assert sorted(cal.sums) == sorted([merged, alone]) and cal.clusters == 2
    assert cal.ceiling == round(percentile([merged, alone], 90), 2) and cal.previous == 15
    with pytest.raises(CalibrationError, match="--force"):
        calibrate(conn, tmp_path, W.model_copy(update={"frequency_recalibrated_at":
                                                        date(2026, 11, 10)}))


def test_write_ceiling_keeps_comments_and_needs_both_keys(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("scoring:\n  frequency: 30\n  # the ceiling\n  frequency_ceiling: 15   # hand\n"
                    "  frequency_recalibrated_at: null   # date\n")
    write_ceiling(path, 12.34, date(2026, 12, 1))
    text = path.read_text()
    assert "frequency_ceiling: 12.34   # hand" in text and "# the ceiling" in text
    assert "frequency_recalibrated_at: 2026-12-01   # date" in text
    w = load_settings(tmp_path).scoring
    assert (w.frequency_ceiling, w.frequency_recalibrated_at) == (12.34, date(2026, 12, 1))
    path.write_text("scoring:\n  frequency_ceiling: 15\n")
    with pytest.raises(CalibrationError, match="frequency_recalibrated_at"):
        write_ceiling(path, 10, date(2026, 12, 1))
