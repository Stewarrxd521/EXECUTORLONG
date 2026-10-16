"""Varias operaciones del bot (trade_id distintos) en el mismo símbolo y dirección.

Binance las junta en una sola posición; un close de un trade_id solo debe cerrar
lo que abrió ese trade_id (su lote), nunca la posición entera mientras haya otros.
Las señales son las mismas que envía botKLINE: el close lleva trade_id pero no quantity.
"""

import asyncio
import json

import pytest

from executor.service import ExecutorService
from executor.trading import Trade
from tests.conftest import make_settings, wait_for

SYM = "ETHUSDT"


def amt(fake, direction="LONG"):
    return fake.positions.get((SYM, direction), {"amt": 0.0})["amt"]


def bot_open(t, trade_id, qty, direction="LONG", price=3000):
    return t.handle_signal({"action": "open", "trade_id": trade_id, "symbol": SYM, "direction": direction,
                            "price": price, "quantity": qty, "notional": round(qty * price, 4), "level": 1.2,
                            "strategy": "close"})


def bot_close(t, trade_id, direction="LONG", reason="TP", price=3010):
    return t.handle_signal({"action": "close", "trade_id": trade_id, "symbol": SYM, "direction": direction,
                            "reason": reason, "close_price": price, "pnl": 0.1, "strategy": "close"})


async def open_lots(t, fake, lots, direction="LONG"):
    total = 0.0
    for tid, qty in lots:
        bot_open(t, tid, qty, direction)
        total += qty
        await wait_for(lambda: abs(amt(fake, direction)) == pytest.approx(total))
        await wait_for(lambda: t.get_trade(SYM, direction) is not None
                       and t.get_trade(SYM, direction).quantity == pytest.approx(total))
    return t.get_trade(SYM, direction)


async def test_two_trade_ids_close_only_their_own_lot(service, fake):
    t = service.trades
    trade = await open_lots(t, fake, [(1, 0.01), (2, 0.02)])
    assert trade.bot_lot_ids() == [1, 2]
    assert trade.lot_qty(1) == pytest.approx(0.01) and trade.lot_qty(2) == pytest.approx(0.02)

    bot_close(t, 1)
    await wait_for(lambda: t.closed and t.closed[-1].paper_trade_id == 1)
    assert amt(fake) == pytest.approx(0.02)                  # el trade_id 2 sigue abierto en Binance
    assert t.get_trade(SYM, "LONG") is trade and trade.quantity == pytest.approx(0.02)
    assert trade.bot_lot_ids() == [2] and t.find_by_paper_id(1, SYM) is None
    rec = t.closed[-1]
    assert rec.status == "TP" and rec.quantity == pytest.approx(0.01) and rec.has_bot_pnl
    assert "cerrado solo el trade_id 1" in t.signals(1)[0]["detail"]

    bot_close(t, 2, reason="SL")
    await wait_for(lambda: t.get_trade(SYM, "LONG") is None)
    assert amt(fake) == 0 and t.closed[-1].paper_trade_id == 2 and t.closed[-1].status == "SL"


async def test_three_trade_ids_closed_in_any_order(service, fake):
    t = service.trades
    await open_lots(t, fake, [(11, 0.01), (12, 0.02), (13, 0.03)])
    bot_close(t, 12)
    await wait_for(lambda: amt(fake) == pytest.approx(0.04))
    bot_close(t, 11)
    await wait_for(lambda: amt(fake) == pytest.approx(0.03))
    trade = t.get_trade(SYM, "LONG")
    assert trade.bot_lot_ids() == [13] and trade.quantity == pytest.approx(0.03)
    bot_close(t, 13)
    await wait_for(lambda: t.get_trade(SYM, "LONG") is None)
    assert amt(fake) == 0
    assert [c.paper_trade_id for c in t.closed[-3:]] == [12, 11, 13]


async def test_simultaneous_closes_of_two_trade_ids(service, fake):
    """botKLINE cierra varios trade_id en el mismo tick (todos tocan su TP a la vez)."""
    t = service.trades
    await open_lots(t, fake, [(21, 0.01), (22, 0.02), (23, 0.03)])
    bot_close(t, 21)
    bot_close(t, 23)
    await wait_for(lambda: amt(fake) == pytest.approx(0.02))
    await asyncio.sleep(0.3)
    trade = t.get_trade(SYM, "LONG")
    assert trade is not None and trade.bot_lot_ids() == [22] and amt(fake) == pytest.approx(0.02)
    assert all(e["ok"] for e in t.signals(5) if e["action"] == "close")


async def test_close_of_unknown_trade_id_does_not_touch_other_lots(service, fake):
    t = service.trades
    await open_lots(t, fake, [(31, 0.01), (32, 0.01)])
    bot_close(t, 99)   # su apertura nunca llegó al executor
    await wait_for(lambda: t.signals(1) and t.signals(1)[0]["action"] == "close")
    assert "no se toca esa posición" in t.signals(1)[0]["detail"]
    await asyncio.sleep(0.3)
    assert amt(fake) == pytest.approx(0.02) and t.get_trade(SYM, "LONG").bot_lot_ids() == [31, 32]


async def test_close_before_open_only_closes_the_late_lot(service, fake):
    t = service.trades
    await open_lots(t, fake, [(41, 0.01)])
    bot_close(t, 42)                       # llega antes que su apertura
    await wait_for(lambda: t._pending_close)
    bot_open(t, 42, 0.02)
    await wait_for(lambda: t.closed and t.closed[-1].paper_trade_id == 42)
    await wait_for(lambda: amt(fake) == pytest.approx(0.01))
    trade = t.get_trade(SYM, "LONG")
    assert trade.bot_lot_ids() == [41] and trade.quantity == pytest.approx(0.01)


async def test_long_and_short_on_same_symbol_are_independent(service, fake):
    t = service.trades
    await open_lots(t, fake, [(51, 0.01), (52, 0.01)], "LONG")
    await open_lots(t, fake, [(53, 0.02)], "SHORT")
    bot_close(t, 53, "SHORT")
    await wait_for(lambda: t.get_trade(SYM, "SHORT") is None)
    assert amt(fake, "LONG") == pytest.approx(0.02) and t.get_trade(SYM, "LONG").bot_lot_ids() == [51, 52]
    bot_close(t, 51, "LONG")
    await wait_for(lambda: amt(fake, "LONG") == pytest.approx(0.01))


async def test_lots_follow_the_multiplier(open_service, fake):
    svc, t = open_service, open_service.trades
    svc.multiplier.configure({"enabled": True, "mode": "manual", "factor": 3})
    await wait_for(lambda: t.ready.is_set())
    bot_open(t, 61, 0.01)
    await wait_for(lambda: amt(fake) == pytest.approx(0.03))
    bot_open(t, 62, 0.01)
    await wait_for(lambda: amt(fake) == pytest.approx(0.06))
    await wait_for(lambda: t.get_trade(SYM, "LONG").bot_lot_ids() == [61, 62])
    bot_close(t, 61)
    await wait_for(lambda: amt(fake) == pytest.approx(0.03))
    assert t.get_trade(SYM, "LONG").bot_lot_ids() == [62]


async def test_external_reduction_scales_lots(service, fake):
    t = service.trades
    trade = await open_lots(t, fake, [(71, 0.01), (72, 0.03)])
    trade.sync_quantity(0.02)              # p. ej. un cierre parcial hecho a mano en Binance
    assert trade.lot_qty(71) == pytest.approx(0.005) and trade.lot_qty(72) == pytest.approx(0.015)


async def test_lots_survive_a_restart(fake, tmp_path, snapshot_file):
    svc = ExecutorService(make_settings(fake, tmp_path))
    await svc.start()
    await wait_for(lambda: svc.core.market.marks and svc.user.conn.connected)
    await open_lots(svc.trades, fake, [(81, 0.01), (82, 0.02)])
    await svc.stop()

    svc2 = ExecutorService(make_settings(fake, tmp_path))
    await svc2.start()
    try:
        await wait_for(lambda: svc2.trades.ready.is_set())
        t = svc2.trades
        assert t.get_trade(SYM, "LONG").lot_qty(82) == pytest.approx(0.02)
        bot_close(t, 82)
        await wait_for(lambda: amt(fake) == pytest.approx(0.01))
        assert t.get_trade(SYM, "LONG").bot_lot_ids() == [81]
    finally:
        await svc2.stop()


def test_trades_saved_before_lots_are_split_evenly():
    raw = {"id": 1, "symbol": SYM, "direction": "LONG", "entry_price": 3000, "quantity": 0.03,
           "open_time": "x", "leverage": 5, "paper_trade_id": 3, "paper_ids": [1, 2, 3]}
    trade = Trade.from_dict(json.loads(json.dumps(raw)))
    assert trade.bot_lot_ids() == [1, 2, 3]
    assert trade.lot_qty(2) == pytest.approx(0.01)
    single = Trade.from_dict({**raw, "paper_ids": [], "paper_trade_id": 0})
    assert single.lot_qty(0) == pytest.approx(0.03) and single.bot_lot_ids() == []


async def test_one_way_account_is_flagged(tmp_path, snapshot_file):
    from tests.fake_binance import FakeBinance

    fake = FakeBinance(hedge=False)
    await fake.start()
    svc = ExecutorService(make_settings(fake, tmp_path))
    await svc.start()
    try:
        await wait_for(lambda: svc.trades.ready.is_set())
        assert any(e["kind"] == "position_mode" for e in svc.core.bus.recent)
    finally:
        await svc.stop()
        await fake.stop()


async def test_hedge_account_is_not_flagged(service):
    await wait_for(lambda: service.trades.ready.is_set())
    assert not any(e["kind"] == "position_mode" for e in service.core.bus.recent)
    assert service.positions_view() == []


async def test_repeated_close_of_the_same_trade_id_is_harmless(service, fake):
    t = service.trades
    await open_lots(t, fake, [(91, 0.01), (92, 0.02)])
    bot_close(t, 91)
    bot_close(t, 91)   # el bot reintenta el envío
    await wait_for(lambda: amt(fake) == pytest.approx(0.02))
    await asyncio.sleep(0.3)
    assert amt(fake) == pytest.approx(0.02) and t.get_trade(SYM, "LONG").bot_lot_ids() == [92]
