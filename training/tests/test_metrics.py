from learned_ce.metrics import q_error, summarize_q_errors


def test_q_error_symmetric():
    assert q_error(100, 200) == 2.0
    assert q_error(200, 100) == 2.0


def test_q_error_perfect():
    assert q_error(42, 42) == 1.0


def test_summarize_q_errors():
    stats = summarize_q_errors([1.0, 2.0, 4.0, 8.0])
    assert stats["median_q_error"] == 3.0
    assert stats["max_q_error"] == 8.0
