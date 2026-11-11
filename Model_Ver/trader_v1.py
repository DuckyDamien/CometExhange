"""Version 1: executable SURVIVAL_KIT basket arbitrage.

The kit is worth exactly one unit of each component:

    SURVIVAL_KIT = TELESCOPE_GLASS + COMET_PILLS + GAS_MASKS

This trader only acts when the currently displayed best prices lock in a
positive edge after crossing every required bid/ask spread.  Every leg uses
the same quantity, keeping the portfolio hedged when all displayed liquidity
fills.
"""

from datamodel import Order, TradingState


KIT = "SURVIVAL_KIT"
COMPONENTS = ("TELESCOPE_GLASS", "COMET_PILLS", "GAS_MASKS")

# Require more than a one-cent theoretical profit to avoid trading on tiny
# discrepancies that may disappear when the live bots react to our orders.
MIN_EDGE = 2


class Trader:
    @staticmethod
    def _position(state: TradingState, symbol: str) -> int:
        return state.position.get(symbol, 0)

    def run(self, state: TradingState):
        orders = {}

        required = (KIT,) + COMPONENTS
        if any(symbol not in state.order_depths for symbol in required):
            return orders, 0, state.trader_data

        depths = {symbol: state.order_depths[symbol] for symbol in required}
        best_bids = {symbol: depths[symbol].best_bid() for symbol in required}
        best_asks = {symbol: depths[symbol].best_ask() for symbol in required}

        # We need both sides of every book to evaluate and execute either
        # direction safely.
        if any(best_bids[symbol] is None or best_asks[symbol] is None for symbol in required):
            return orders, 0, state.trader_data

        kit_ask = best_asks[KIT]
        kit_bid = best_bids[KIT]
        component_bid_sum = sum(best_bids[symbol] for symbol in COMPONENTS)
        component_ask_sum = sum(best_asks[symbol] for symbol in COMPONENTS)

        # Direction 1: the kit is cheap. Buy it and sell one of each component.
        cheap_edge = component_bid_sum - kit_ask
        if cheap_edge >= MIN_EDGE:
            capacities = [
                state.position_limits[KIT] - self._position(state, KIT),
                -depths[KIT].sell_orders[kit_ask],
            ]
            for symbol in COMPONENTS:
                capacities.extend(
                    [
                        state.position_limits[symbol] + self._position(state, symbol),
                        depths[symbol].buy_orders[best_bids[symbol]],
                    ]
                )

            quantity = min(capacities)
            if quantity > 0:
                orders[KIT] = [Order(KIT, int(kit_ask), int(quantity))]
                for symbol in COMPONENTS:
                    orders[symbol] = [
                        Order(symbol, int(best_bids[symbol]), -int(quantity))
                    ]

        # Direction 2: the kit is rich. Sell it and buy one of each component.
        rich_edge = kit_bid - component_ask_sum
        if rich_edge >= MIN_EDGE:
            capacities = [
                state.position_limits[KIT] + self._position(state, KIT),
                depths[KIT].buy_orders[kit_bid],
            ]
            for symbol in COMPONENTS:
                capacities.extend(
                    [
                        state.position_limits[symbol] - self._position(state, symbol),
                        -depths[symbol].sell_orders[best_asks[symbol]],
                    ]
                )

            quantity = min(capacities)
            if quantity > 0:
                orders[KIT] = [Order(KIT, int(kit_bid), -int(quantity))]
                for symbol in COMPONENTS:
                    orders[symbol] = [
                        Order(symbol, int(best_asks[symbol]), int(quantity))
                    ]

        return orders, 0, state.trader_data
