from evotrader.broker import PaperBroker


def test_buy_charges_costs_and_records_position():
    b = PaperBroker(10_000.0, commission_bps=1.0, slippage_bps=5.0)
    assert b.buy("SPY", 5_000.0, 100.0, "2020-01-02", 0, "test rule")
    pos = b.positions["SPY"]
    assert pos.entry_price > 100.0          # slippage crossed the spread
    assert b.cash < 5_000.0                 # commission came out of cash
    assert abs(b.equity({"SPY": 100.0}) - 10_000.0) < 10.0


def test_sell_produces_a_trade_with_pnl():
    b = PaperBroker(10_000.0)
    b.buy("SPY", 5_000.0, 100.0, "2020-01-02", 0, "in")
    trade = b.sell("SPY", 110.0, "2020-02-02", 20, "out")
    assert trade is not None
    assert 0.08 < trade.ret < 0.10          # ~10% less costs
    assert trade.bars_held == 20
    assert "SPY" not in b.positions
    assert b.journal.trades == [trade]


def test_orders_are_bounded_by_cash_and_position_rules():
    b = PaperBroker(1_000.0)
    assert b.buy("SPY", 1e9, 100.0, "d", 0, "huge")     # clipped to cash
    assert b.cash >= 0
    assert not b.buy("SPY", 100.0, 100.0, "d", 0, "dup")  # one lot per symbol
    assert not b.buy("QQQ", 10.0, 100.0, "d", 0, "tiny")  # below min trade value


def test_liquidate_closes_everything():
    b = PaperBroker(10_000.0)
    b.buy("SPY", 3_000.0, 100.0, "d", 0, "in")
    b.buy("QQQ", 3_000.0, 50.0, "d", 0, "in")
    trades = b.liquidate({"SPY": 105.0, "QQQ": 55.0}, "end", 10)
    assert len(trades) == 2 and not b.positions


def test_drawdown_tracks_the_equity_peak():
    b = PaperBroker(10_000.0)
    b.buy("SPY", 10_000.0, 100.0, "d", 0, "in")
    b.mark("d1", {"SPY": 120.0})
    b.mark("d2", {"SPY": 90.0})
    assert b.drawdown({"SPY": 90.0}) < -0.2
