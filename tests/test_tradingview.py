"""TradingView tools, exercised offline against canned scanner/search replies."""
import pytest

from evotrader import tradingview as tv
from evotrader.mcp_server import handle_message


@pytest.fixture(autouse=True)
def fake_tv(monkeypatch):
    calls = []

    def fake_http(url, body=None, **_):
        calls.append((url, body))
        if "symbol_search" in url:
            return {"symbols": [
                {"symbol": "NQ", "type": "futures", "source_id": "CME_MINI",
                 "description": "E-mini Nasdaq-100 Futures",
                 "contracts": [{"symbol": "NQ1!"}, {"symbol": "NQ2!"}]},
                {"symbol": "AAPL", "type": "stock", "source_id": "NASDAQ",
                 "description": "Apple Inc."}]}
        cols = body["columns"]
        if "symbols" in body:
            tickers = body["symbols"]["tickers"]
        else:
            tickers = ["NASDAQ:AAPL"]
        return {"totalCount": len(tickers), "data": [
            {"s": t, "d": [_value(c) for c in cols]} for t in tickers]}

    monkeypatch.setattr(tv, "_http_json", fake_http)
    return calls


def _value(col):
    base = col.split("|")[0]
    if base in ("description", "name"):
        return "Apple Inc."
    if base in ("type", "exchange", "currency"):
        return "stock"
    if base.startswith("Recommend"):
        return 0.6
    return 100.0


def call(name, **args):
    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": name, "arguments": args}}
    result = handle_message(msg)["result"]
    return result["isError"], result["content"][0]["text"]


def test_tradingview_tools_are_listed():
    tools = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {t["name"] for t in tools["result"]["tools"]}
    assert {"quote", "technicals", "bars", "screener", "search_symbol"} <= names


def test_bare_future_resolves_to_its_continuous_contract():
    assert tv.resolve("NQ1!") == "CME_MINI:NQ1!"
    assert tv.resolve("aapl") == "NASDAQ:AAPL"


def test_explicit_exchange_skips_search(fake_tv):
    assert tv.resolve("BINANCE:BTCUSDT") == "BINANCE:BTCUSDT"
    assert fake_tv == []


def test_invalid_ticker_never_reaches_a_url(fake_tv):
    err, text = call("quote", symbols="AAPL;rm")
    assert err and "not a valid TradingView ticker" in text
    assert fake_tv == []


def test_quote_shows_tradingview_rating():
    err, text = call("quote", symbols="NASDAQ:AAPL")
    assert not err
    assert "NASDAQ:AAPL - Apple Inc." in text
    assert "STRONG BUY" in text


def test_technicals_uses_the_timeframe_column_suffix(fake_tv):
    err, text = call("technicals", symbol="NASDAQ:AAPL", timeframe="4h")
    assert not err and "4h timeframe" in text
    assert "Recommend.All|240" in fake_tv[-1][1]["columns"]


def test_bad_timeframe_is_rejected():
    err, text = call("technicals", symbol="NASDAQ:AAPL", timeframe="7m")
    assert err and "timeframe must be one of" in text


def test_screener_applies_preset_and_market():
    err, text = call("screener", preset="oversold", market="america", limit=5)
    assert not err and "oversold in america" in text
    err, text = call("screener", preset="nonsense")
    assert err and "preset must be one of" in text


@pytest.mark.parametrize("value,label", [(0.6, "STRONG BUY"), (0.2, "BUY"), (0.0, "NEUTRAL"),
                                         (-0.2, "SELL"), (-0.7, "STRONG SELL"), (None, "n/a")])
def test_rating_buckets_match_tradingview(value, label):
    assert tv.rating_label(value) == label
