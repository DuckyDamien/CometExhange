"""Version 4: time-aware basket arbitrage plus conservative warrant trading.

PILL_WARRANTS is a zero-rate European call on COMET_PILLS with strike 1200.
The session date is not exposed, so warrant trades must be attractive across
both possible times to expiry and across a volatility uncertainty band.
"""

import json
import math
from statistics import median

from datamodel import Order, TradingState


KIT = "SURVIVAL_KIT"
PILLS = "COMET_PILLS"
WARRANTS = "PILL_WARRANTS"
COMPONENTS = ("TELESCOPE_GLASS", PILLS, "GAS_MASKS")
REQUIRED = (KIT,) + COMPONENTS + (WARRANTS,)

# Basket controls retained from version 3.
EWMA_ALPHA = 0.02
WARMUP_TICKS = 50
BASE_EDGE = 4
VOL_EDGE_MULTIPLIER = 2.0
EARLY_EDGE_PENALTY = 4.0
STAGE_SIZE = 5
MAX_BASKETS = 25

# Warrant controls. Sampling every 20 ticks sharply reduces bid/ask bounce in
# the realized-volatility estimate compared with tick-by-tick returns.
STRIKE = 1200.0
VOL_SAMPLE_TICKS = 20
VOL_WARMUP_SAMPLES = 25
VOL_LOWER_MULTIPLIER = 0.80
VOL_UPPER_MULTIPLIER = 1.20
MIN_VOLATILITY = 1.0
MAX_VOLATILITY = 6.0
OPTION_EDGE = 8.0
MAX_WARRANT_POSITION = 40
WARRANT_STEP = 10

SESSION_DAYS = 7.0 / 24.0
SESSION_YEARS = SESSION_DAYS / 365.0
SAMPLE_YEARS = SESSION_YEARS * VOL_SAMPLE_TICKS / 10000.0


class Trader:
    @staticmethod
    def _default_state():
        return {
            "count": 0,
            "last_residual": None,
            "delta_var": 16.0,
            "vol_last": None,
            "vol_sum_sq": 0.0,
            "vol_n": 0,
        }

    def _load(self, raw):
        if not raw:
            return self._default_state()
        try:
            saved = json.loads(raw)
            state = self._default_state()
            state.update(saved)
            state["count"] = int(state["count"])
            state["delta_var"] = max(float(state["delta_var"]), 0.01)
            state["vol_sum_sq"] = max(float(state["vol_sum_sq"]), 0.0)
            state["vol_n"] = int(state["vol_n"])
            return state
        except (TypeError, ValueError, json.JSONDecodeError):
            return self._default_state()

    @staticmethod
    def _save(stats):
        return json.dumps(stats, separators=(",", ":"))

    @staticmethod
    def _position(state, symbol):
        return state.position.get(symbol, 0)

    @staticmethod
    def _mid(depth):
        bid, ask = depth.best_bid(), depth.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _update_statistics(self, state, depths, stats):
        mids = {symbol: self._mid(depths[symbol]) for symbol in REQUIRED}
        if any(mids[symbol] is None for symbol in (KIT,) + COMPONENTS):
            return None, None, mids

        residual = mids[KIT] - sum(mids[symbol] for symbol in COMPONENTS)
        previous = stats["last_residual"]
        if previous is not None:
            change = residual - float(previous)
            stats["delta_var"] = (
                (1.0 - EWMA_ALPHA) * stats["delta_var"]
                + EWMA_ALPHA * change * change
            )
        stats["last_residual"] = residual
        stats["count"] += 1

        if state.timestamp % (VOL_SAMPLE_TICKS * 100) == 0:
            pill_mid = mids[PILLS]
            last_sample = stats["vol_last"]
            if last_sample is not None and last_sample > 0 and pill_mid > 0:
                log_return = math.log(pill_mid / float(last_sample))
                stats["vol_sum_sq"] += log_return * log_return
                stats["vol_n"] += 1
            stats["vol_last"] = pill_mid

        residual_vol = math.sqrt(stats["delta_var"])
        annual_vol = None
        if stats["vol_n"] >= VOL_WARMUP_SAMPLES:
            annual_vol = math.sqrt(
                stats["vol_sum_sq"] / (stats["vol_n"] * SAMPLE_YEARS)
            )
            annual_vol = max(MIN_VOLATILITY, min(MAX_VOLATILITY, annual_vol))
        return residual_vol, annual_vol, mids

    @staticmethod
    def _normal_cdf(value):
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def _call_value_delta(self, spot, time_years, volatility):
        if spot <= 0 or time_years <= 0 or volatility <= 0:
            return max(0.0, spot - STRIKE), 1.0 if spot > STRIKE else 0.0
        root_time = math.sqrt(time_years)
        total_vol = volatility * root_time
        d1 = (
            math.log(spot / STRIKE) + 0.5 * volatility * volatility * time_years
        ) / total_vol
        d2 = d1 - total_vol
        delta = self._normal_cdf(d1)
        value = spot * delta - STRIKE * self._normal_cdf(d2)
        return value, delta

    def _warrant_bounds(self, spot, timestamp, volatility):
        elapsed_days = min(SESSION_DAYS, timestamp / 999900.0 * SESSION_DAYS)
        low_time = max(1e-8, (2.0 - elapsed_days) / 365.0)
        high_time = max(low_time, (3.0 - elapsed_days) / 365.0)
        low_vol = max(MIN_VOLATILITY, volatility * VOL_LOWER_MULTIPLIER)
        high_vol = min(MAX_VOLATILITY, volatility * VOL_UPPER_MULTIPLIER)
        low_value, _ = self._call_value_delta(spot, low_time, low_vol)
        high_value, _ = self._call_value_delta(spot, high_time, high_vol)
        middle_time = (low_time + high_time) / 2.0
        middle_value, delta = self._call_value_delta(spot, middle_time, volatility)
        return low_value, high_value, middle_value, delta

    @staticmethod
    def _basket_target(state):
        # Glass and masks are not used for the option hedge, so they identify
        # the actual basket count even when pills contains deliberate delta.
        implied = [
            state.position.get(KIT, 0),
            -state.position.get("TELESCOPE_GLASS", 0),
            -state.position.get("GAS_MASKS", 0),
        ]
        return max(-MAX_BASKETS, min(MAX_BASKETS, int(round(median(implied)))))

    def _desired_positions(self, state, delta):
        basket = self._basket_target(state)
        warrant_position = self._position(state, WARRANTS)
        option_hedge = int(round(-delta * warrant_position)) if delta is not None else 0
        return {
            KIT: basket,
            "TELESCOPE_GLASS": -basket,
            "GAS_MASKS": -basket,
            PILLS: -basket + option_hedge,
        }

    def _repair_orders(self, state, depths, desired):
        orders = {}
        for symbol, target in desired.items():
            difference = target - self._position(state, symbol)
            if difference > 0:
                ask = depths[symbol].best_ask()
                if ask is None:
                    continue
                quantity = min(
                    difference,
                    -depths[symbol].sell_orders[ask],
                    state.position_limits[symbol] - self._position(state, symbol),
                )
                if quantity > 0:
                    orders[symbol] = [Order(symbol, int(ask), int(quantity))]
            elif difference < 0:
                bid = depths[symbol].best_bid()
                if bid is None:
                    continue
                quantity = min(
                    -difference,
                    depths[symbol].buy_orders[bid],
                    state.position_limits[symbol] + self._position(state, symbol),
                )
                if quantity > 0:
                    orders[symbol] = [Order(symbol, int(bid), -int(quantity))]
        return orders

    def _option_orders(self, state, depths, bounds):
        low_value, high_value, _middle_value, delta = bounds
        warrant_depth = depths[WARRANTS]
        bid, ask = warrant_depth.best_bid(), warrant_depth.best_ask()
        if bid is None or ask is None:
            return None

        current = self._position(state, WARRANTS)
        target = current
        if ask + OPTION_EDGE < low_value:
            target = MAX_WARRANT_POSITION
        elif bid - OPTION_EDGE > high_value:
            target = -MAX_WARRANT_POSITION
        if target == current:
            return None

        direction = 1 if target > current else -1
        wanted = min(WARRANT_STEP, abs(target - current))
        available = (
            -warrant_depth.sell_orders[ask]
            if direction > 0
            else warrant_depth.buy_orders[bid]
        )
        capacity = (
            state.position_limits[WARRANTS] - current
            if direction > 0
            else state.position_limits[WARRANTS] + current
        )
        quantity = min(wanted, available, capacity)

        # Reduce the option order until its projected pills hedge fits inside
        # both the position limit and displayed top-level liquidity.
        kit_position = self._position(state, KIT)
        pill_depth = depths[PILLS]
        while quantity > 0:
            projected_warrants = current + direction * quantity
            projected_pills = -kit_position + int(round(-delta * projected_warrants))
            pill_change = projected_pills - self._position(state, PILLS)
            if abs(projected_pills) <= state.position_limits[PILLS]:
                if pill_change > 0:
                    pill_ask = pill_depth.best_ask()
                    if pill_ask is not None and -pill_depth.sell_orders[pill_ask] >= pill_change:
                        break
                elif pill_change < 0:
                    pill_bid = pill_depth.best_bid()
                    if pill_bid is not None and pill_depth.buy_orders[pill_bid] >= -pill_change:
                        break
                else:
                    break
            quantity -= 1
        if quantity <= 0:
            return None

        orders = {
            WARRANTS: [Order(WARRANTS, int(ask if direction > 0 else bid), direction * int(quantity))]
        }
        projected_warrants = current + direction * quantity
        projected_pills = -kit_position + int(round(-delta * projected_warrants))
        pill_change = projected_pills - self._position(state, PILLS)
        if pill_change > 0:
            pill_ask = pill_depth.best_ask()
            orders[PILLS] = [Order(PILLS, int(pill_ask), int(pill_change))]
        elif pill_change < 0:
            pill_bid = pill_depth.best_bid()
            orders[PILLS] = [Order(PILLS, int(pill_bid), int(pill_change))]
        return orders

    @staticmethod
    def _target_basket_size(edge, threshold, residual_volatility):
        excess = max(0.0, edge - threshold)
        stages = 1 + int(excess / max(residual_volatility, 1.0))
        return min(MAX_BASKETS, STAGE_SIZE * stages)

    def _basket_orders(self, state, depths, residual_volatility, hedge_target):
        bids = {symbol: depths[symbol].best_bid() for symbol in (KIT,) + COMPONENTS}
        asks = {symbol: depths[symbol].best_ask() for symbol in (KIT,) + COMPONENTS}
        if any(bids[s] is None or asks[s] is None for s in (KIT,) + COMPONENTS):
            return {}

        progress = min(1.0, max(0.0, state.timestamp / 999900.0))
        threshold = max(
            BASE_EDGE,
            int(
                math.ceil(
                    VOL_EDGE_MULTIPLIER * residual_volatility
                    + EARLY_EDGE_PENALTY * (1.0 - progress)
                )
            ),
        )
        cheap_edge = sum(bids[s] for s in COMPONENTS) - asks[KIT]
        rich_edge = bids[KIT] - sum(asks[s] for s in COMPONENTS)
        if cheap_edge >= threshold:
            direction, edge = 1, cheap_edge
        elif rich_edge >= threshold:
            direction, edge = -1, rich_edge
        else:
            return {}

        current = self._position(state, KIT)
        target = direction * self._target_basket_size(edge, threshold, residual_volatility)
        change = target - current
        if change == 0 or (change > 0) != (direction > 0):
            return {}

        wanted = abs(change)
        if direction > 0:
            capacities = [
                wanted,
                state.position_limits[KIT] - current,
                -depths[KIT].sell_orders[asks[KIT]],
                state.position_limits[PILLS] + self._position(state, PILLS),
            ]
            for symbol in COMPONENTS:
                capacities.append(depths[symbol].buy_orders[bids[symbol]])
            for symbol in ("TELESCOPE_GLASS", "GAS_MASKS"):
                capacities.append(
                    state.position_limits[symbol] + self._position(state, symbol)
                )
            quantity = min(capacities)
            if quantity <= 0:
                return {}
            orders = {KIT: [Order(KIT, int(asks[KIT]), int(quantity))]}
            for symbol in COMPONENTS:
                orders[symbol] = [Order(symbol, int(bids[symbol]), -int(quantity))]
        else:
            capacities = [
                wanted,
                state.position_limits[KIT] + current,
                depths[KIT].buy_orders[bids[KIT]],
                state.position_limits[PILLS] - self._position(state, PILLS),
            ]
            for symbol in COMPONENTS:
                capacities.append(-depths[symbol].sell_orders[asks[symbol]])
            for symbol in ("TELESCOPE_GLASS", "GAS_MASKS"):
                capacities.append(
                    state.position_limits[symbol] - self._position(state, symbol)
                )
            quantity = min(capacities)
            if quantity <= 0:
                return {}
            orders = {KIT: [Order(KIT, int(bids[KIT]), -int(quantity))]}
            for symbol in COMPONENTS:
                orders[symbol] = [Order(symbol, int(asks[symbol]), int(quantity))]
        return orders

    def run(self, state: TradingState):
        stats = self._load(state.trader_data)
        if any(symbol not in state.order_depths for symbol in REQUIRED):
            return {}, 0, self._save(stats)
        depths = {symbol: state.order_depths[symbol] for symbol in REQUIRED}
        residual_volatility, annual_volatility, mids = self._update_statistics(
            state, depths, stats
        )
        saved = self._save(stats)

        bounds = None
        delta = None
        if annual_volatility is not None and mids[PILLS] is not None:
            bounds = self._warrant_bounds(mids[PILLS], state.timestamp, annual_volatility)
            delta = bounds[3]

        desired = self._desired_positions(state, delta)
        repair = self._repair_orders(state, depths, desired)
        if repair:
            return repair, 0, saved

        if bounds is not None:
            option_orders = self._option_orders(state, depths, bounds)
            if option_orders:
                return option_orders, 0, saved

        if residual_volatility is None or stats["count"] < WARMUP_TICKS:
            return {}, 0, saved
        hedge_target = desired[PILLS] + desired[KIT]
        basket_orders = self._basket_orders(
            state, depths, residual_volatility, hedge_target
        )
        return basket_orders, 0, saved
