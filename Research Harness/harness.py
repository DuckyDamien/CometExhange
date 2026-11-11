"""Instrumented replay engine for research, diagnostics, and stress tests.

This deliberately lives beside, rather than inside, the official run_local.py.
It follows the same matching rules while exposing tick-level records and
deterministic execution stresses. It is not submission code.
"""

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datamodel import Listing, Observation, Order, OrderDepth, Trade, TradingState
from run_local import CURRENCY, LIMITS, MAX_TRADER_DATA, PRODUCTS, SUBMISSION, load_prices, load_trades
from research.metrics import summarize
from research.scenarios import BASE, Scenario, transform_levels


KIT = "SURVIVAL_KIT"
COMPONENTS = ("TELESCOPE_GLASS", "COMET_PILLS", "GAS_MASKS")


def load_trader_factory(path, overrides=None, features=None):
    path = Path(path).resolve()
    overrides = overrides or {}
    features = features or {}
    counter = {"value": 0}

    def factory():
        counter["value"] += 1
        name = f"research_submission_{path.stem}_{counter['value']}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for key, value in overrides.items():
            if hasattr(module, key):
                setattr(module, key, value)

        trader_class = module.Trader
        if features.get("repair") is False and hasattr(trader_class, "_balanced"):
            trader_class._balanced = staticmethod(lambda state: True)
        return trader_class()

    return factory


def _depth(levels):
    bids, asks = levels
    return OrderDepth(
        {int(price): int(volume) for price, volume in bids},
        {int(price): -int(volume) for price, volume in asks},
    )


def _best_mid(bids, asks, fallback):
    if bids and asks:
        return (bids[0][0] + asks[0][0]) / 2.0
    if bids:
        return float(bids[0][0])
    if asks:
        return float(asks[0][0])
    return fallback


def _hedge_imbalance(position):
    kit = position.get(KIT, 0)
    return max(abs(position.get(symbol, 0) + kit) for symbol in COMPONENTS)


def _basket_decomposition(position):
    implied = [position.get(KIT, 0)]
    implied.extend(-position.get(symbol, 0) for symbol in COMPONENTS)
    baskets = int(round(median(implied)))
    residual = dict(position)
    residual[KIT] -= baskets
    for symbol in COMPONENTS:
        residual[symbol] += baskets
    return baskets, residual


def _equities(cash, position, mids, visible):
    mid_equity = sum(cash[p] + position[p] * mids[p] for p in position)
    liquidation = sum(cash.values())
    for symbol, quantity in position.items():
        bids, asks = visible[symbol]
        if quantity > 0:
            price = bids[0][0] if bids else mids[symbol]
        elif quantity < 0:
            price = asks[0][0] if asks else mids[symbol]
        else:
            price = 0.0
        liquidation += quantity * price

    baskets, residual_positions = _basket_decomposition(position)
    structural = sum(cash.values()) + sum(
        residual_positions[symbol] * mids[symbol] for symbol in residual_positions
    )
    return mid_equity, liquidation, structural


def replay(
    trader_path,
    day,
    data_dir=None,
    strict=False,
    scenario=BASE,
    overrides=None,
    features=None,
    start_index=0,
    end_index=None,
    warmup=0,
):
    data_dir = Path(data_dir or ROOT / "data")
    stamps, books = load_prices(data_dir / f"prices_round_1_day_{day}.csv")
    tape = load_trades(data_dir / f"trades_round_1_day_{day}.csv")
    end_index = len(stamps) if end_index is None else min(end_index, len(stamps))
    start_index = max(0, min(start_index, end_index))
    replay_start = max(0, start_index - max(0, warmup))
    selected = stamps[replay_start:end_index]

    products = [p for p in PRODUCTS if p in books[stamps[0]]]
    listings = {p: Listing(p, p, CURRENCY) for p in products}
    position = {p: 0 for p in products}
    cash = {p: 0.0 for p in products}
    last_mid = {p: 0.0 for p in products}
    trader_data, own_previous, market_previous = "", {}, {}
    factory = load_trader_factory(trader_path, overrides, features)
    trader = factory()

    records = []
    violations = errors = timeouts = restarts = 0
    aggressive_fills = resting_fills = 0
    tick_times = []

    for absolute_index, timestamp in enumerate(selected, start=replay_start):
        local_index = absolute_index - replay_start
        if scenario.restart_every and local_index > 0 and local_index % scenario.restart_every == 0:
            trader = factory()
            restarts += 1

        visible = {}
        depths = {}
        for symbol in products:
            raw_bids, raw_asks, raw_mid = books[timestamp].get(symbol, ([], [], 0.0))
            bids, asks = transform_levels(
                raw_bids, raw_asks, symbol, absolute_index, scenario
            )
            visible[symbol] = (bids, asks)
            depths[symbol] = _depth((bids, asks))
            last_mid[symbol] = _best_mid(bids, asks, raw_mid or last_mid[symbol])

        state = TradingState(
            trader_data,
            timestamp,
            listings,
            depths,
            own_previous,
            market_previous,
            dict(position),
            dict(LIMITS),
            Observation(),
        )
        started = time.perf_counter()
        try:
            orders_by_symbol, _conversions, new_data = trader.run(state)
        except Exception:
            orders_by_symbol, new_data = {}, trader_data
            errors += 1
        elapsed = time.perf_counter() - started
        tick_times.append(elapsed)
        limit = 2.0 if local_index == 0 else 0.100
        if elapsed > limit:
            orders_by_symbol = {}
            timeouts += 1
        if isinstance(new_data, str) and len(new_data) <= MAX_TRADER_DATA:
            trader_data = new_data
        if not isinstance(orders_by_symbol, dict) or absolute_index < start_index:
            orders_by_symbol = {}

        pre_imbalance = _hedge_imbalance(position)
        own_now = {}
        tape_now = tape.get(timestamp, {})
        tick_turnover = 0.0
        tick_fills = 0
        repair_cost = 0.0

        for symbol in products:
            bids, asks = visible[symbol]
            orders = orders_by_symbol.get(symbol) or []
            valid = [
                order
                for order in orders
                if isinstance(order, Order)
                and order.symbol == symbol
                and type(order.price) is int
                and type(order.quantity) is int
                and order.quantity != 0
            ]
            buys = sum(order.quantity for order in valid if order.quantity > 0)
            sells = sum(-order.quantity for order in valid if order.quantity < 0)
            if valid and (
                position[symbol] + buys > LIMITS[symbol]
                or position[symbol] - sells < -LIMITS[symbol]
            ):
                violations += 1
                valid = []
            if not valid or scenario.drops_fill(symbol, absolute_index):
                continue

            bid_levels = [[price, volume] for price, volume in bids]
            ask_levels = [[price, volume] for price, volume in asks]
            rest_buys, rest_sells = [], []
            for order in valid:
                remaining = abs(order.quantity)
                buying = order.quantity > 0
                levels = ask_levels if buying else bid_levels
                for level in levels:
                    if remaining == 0 or (
                        level[0] > order.price if buying else level[0] < order.price
                    ):
                        break
                    filled = min(remaining, level[1])
                    if filled <= 0:
                        continue
                    level[1] -= filled
                    remaining -= filled
                    fill_price = level[0] + scenario.slippage if buying else level[0] - scenario.slippage
                    aggressive_fills += 1
                    tick_fills += 1
                    tick_turnover += filled * fill_price
                    if pre_imbalance and bids and asks:
                        repair_cost += filled * (asks[0][0] - bids[0][0]) / 2.0
                    if buying:
                        position[symbol] += filled
                        cash[symbol] -= filled * fill_price
                        own_now.setdefault(symbol, []).append(
                            Trade(symbol, fill_price, filled, SUBMISSION, None, timestamp)
                        )
                    else:
                        position[symbol] -= filled
                        cash[symbol] += filled * fill_price
                        own_now.setdefault(symbol, []).append(
                            Trade(symbol, fill_price, filled, None, SUBMISSION, timestamp)
                        )
                if remaining:
                    (rest_buys if buying else rest_sells).append([order.price, remaining])

            prints = tape_now.get(symbol)
            if not prints or not (rest_buys or rest_sells):
                continue
            best_bid = bids[0][0] if bids else None
            best_ask = asks[0][0] if asks else None
            left = [[price, quantity] for price, quantity in prints]
            sell_prints = [trade for trade in left if best_ask is None or trade[0] < best_ask]
            buy_prints = [trade for trade in left if best_bid is None or trade[0] > best_bid]
            sides = (
                (sorted(rest_buys, key=lambda item: -item[0]), sell_prints, True),
                (sorted(rest_sells, key=lambda item: item[0]), buy_prints, False),
            )
            for mine, side_prints, buying in sides:
                for resting in mine:
                    if not scenario.allows_resting_fill(symbol, absolute_index, buying):
                        continue
                    for trade in side_prints:
                        if resting[1] == 0:
                            break
                        if trade[1] <= 0:
                            continue
                        hit = (
                            trade[0] < resting[0] if strict else trade[0] <= resting[0]
                        ) if buying else (
                            trade[0] > resting[0] if strict else trade[0] >= resting[0]
                        )
                        if not hit:
                            continue
                        filled = min(resting[1], trade[1])
                        resting[1] -= filled
                        trade[1] -= filled
                        fill_price = resting[0] + scenario.slippage if buying else resting[0] - scenario.slippage
                        resting_fills += 1
                        tick_fills += 1
                        tick_turnover += filled * fill_price
                        if buying:
                            position[symbol] += filled
                            cash[symbol] -= filled * fill_price
                            own_now.setdefault(symbol, []).append(
                                Trade(symbol, fill_price, filled, SUBMISSION, None, timestamp)
                            )
                        else:
                            position[symbol] -= filled
                            cash[symbol] += filled * fill_price
                            own_now.setdefault(symbol, []).append(
                                Trade(symbol, fill_price, filled, None, SUBMISSION, timestamp)
                            )

        own_previous = own_now
        market_previous = {
            symbol: [Trade(symbol, price, quantity, None, None, timestamp) for price, quantity in trades]
            for symbol, trades in tape_now.items()
        }
        if absolute_index < start_index:
            continue

        mid_equity, liquidation_equity, structural_equity = _equities(
            cash, position, last_mid, visible
        )
        basket_position, residual_positions = _basket_decomposition(position)
        residual = last_mid[KIT] - sum(last_mid[symbol] for symbol in COMPONENTS)
        bids = {s: visible[s][0][0][0] if visible[s][0] else None for s in (KIT,) + COMPONENTS}
        asks = {s: visible[s][1][0][0] if visible[s][1] else None for s in (KIT,) + COMPONENTS}
        cheap_edge = (
            sum(bids[s] for s in COMPONENTS) - asks[KIT]
            if asks[KIT] is not None and all(bids[s] is not None for s in COMPONENTS)
            else math.nan
        )
        rich_edge = (
            bids[KIT] - sum(asks[s] for s in COMPONENTS)
            if bids[KIT] is not None and all(asks[s] is not None for s in COMPONENTS)
            else math.nan
        )
        executed_edge = None
        if tick_fills and not pre_imbalance:
            available_edges = [edge for edge in (cheap_edge, rich_edge) if not math.isnan(edge)]
            executed_edge = max(available_edges) if available_edges else None
        records.append(
            {
                "day": day,
                "timestamp": timestamp,
                "mid_equity": mid_equity,
                "liquidation_equity": liquidation_equity,
                "structural_equity": structural_equity,
                "basket_residual": residual,
                "cheap_edge": cheap_edge,
                "rich_edge": rich_edge,
                "executed_edge": executed_edge,
                "kit_position": position[KIT],
                "basket_position": basket_position,
                "glass_position": position["TELESCOPE_GLASS"],
                "pills_position": position["COMET_PILLS"],
                "masks_position": position["GAS_MASKS"],
                "residual_glass_position": residual_positions["TELESCOPE_GLASS"],
                "residual_pills_position": residual_positions["COMET_PILLS"],
                "residual_masks_position": residual_positions["GAS_MASKS"],
                "hedge_imbalance": _hedge_imbalance(position),
                "tick_turnover": tick_turnover,
                "tick_fills": tick_fills,
                "repair_cost": repair_cost,
            }
        )

    metrics = summarize(records)
    metrics.update(
        {
            "day": day,
            "scenario": scenario.name,
            "violations": violations,
            "errors": errors,
            "timeouts": timeouts,
            "restarts": restarts,
            "aggressive_fills": aggressive_fills,
            "resting_fills": resting_fills,
            "mean_tick_ms": 1000.0 * sum(tick_times) / len(tick_times) if tick_times else 0.0,
            "max_tick_ms": 1000.0 * max(tick_times) if tick_times else 0.0,
        }
    )
    return {"metrics": metrics, "records": records, "final_positions": dict(position)}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Instrumented competition research replay")
    parser.add_argument("trader")
    parser.add_argument("--day", choices=["-1", "0", "all"], default="all")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    days = [-1, 0] if args.day == "all" else [int(args.day)]
    all_records = []
    all_metrics = []
    for day in days:
        result = replay(args.trader, day, strict=args.strict)
        all_records.extend(result["records"])
        all_metrics.append(result["metrics"])
        print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    if args.output:
        write_csv(args.output / "ticks.csv", all_records)
        write_csv(args.output / "metrics.csv", all_metrics)


if __name__ == "__main__":
    main()
