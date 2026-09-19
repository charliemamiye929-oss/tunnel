import pytest

from evotrader.dsl import DslError, compile_rule, parse
from evotrader.features import FEATURE_SET


def ev(src, cur, prev=None):
    return compile_rule(src, FEATURE_SET)(cur, prev if prev is not None else cur)


def test_comparisons_and_logic():
    ctx = {"rsi14": 25.0, "close": 110.0, "sma200": 100.0, "vol20": 0.2}
    assert ev("rsi14 < 30 and close > sma200", ctx)
    assert not ev("rsi14 < 30 and close < sma200", ctx)
    assert ev("rsi14 > 90 or close > sma200", ctx)
    assert ev("not (rsi14 > 90)", ctx)


def test_arithmetic_precedence():
    ctx = {"close": 10.0, "sma20": 2.0, "atr14": 1.0}
    assert ev("close > sma20 * 2 + 1", ctx)
    assert ev("(close - sma20) / atr14 > 7", ctx)
    assert not ev("close / 0 > 1", ctx)   # division by zero yields 0, not an error


def test_crossing_functions_use_the_previous_bar():
    cur = {"macd": 1.0, "macd_signal": 0.5}
    prev = {"macd": 0.1, "macd_signal": 0.5}
    assert ev("cross_above(macd, macd_signal)", cur, prev)
    assert not ev("cross_above(macd, macd_signal)", cur, cur)
    assert ev("cross_below(macd_signal, macd)", cur, prev)


def test_prev_and_change():
    cur, prev = {"rsi14": 40.0}, {"rsi14": 30.0}
    assert ev("prev(rsi14) < 35", cur, prev)
    assert ev("change(rsi14) > 5", cur, prev)


def test_unknown_feature_is_rejected():
    with pytest.raises(DslError):
        compile_rule("close > secret_alpha", FEATURE_SET)


@pytest.mark.parametrize("src", [
    "__import__('os').system('ls')",
    "open('/etc/passwd')",
    "close.__class__",
    "rsi14 <",
    "1 +",
    "foo(1, 2)",
    "",
])
def test_malicious_or_malformed_rules_are_rejected(src):
    with pytest.raises(DslError):
        compile_rule(src, FEATURE_SET)


def test_rule_length_is_capped():
    with pytest.raises(DslError):
        parse("rsi14 < 30 and " * 100 + "rsi14 < 30")
