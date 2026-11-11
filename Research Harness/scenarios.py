"""Deterministic market and execution stresses for replay tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str = "base"
    liquidity_scale: float = 1.0
    spread_widening: int = 0
    slippage: int = 0
    drop_symbol: str = ""
    drop_every: int = 0
    missing_side_symbol: str = ""
    missing_side: str = ""
    missing_side_every: int = 0
    restart_every: int = 0

    def drops_fill(self, symbol, tick_index):
        return bool(
            self.drop_symbol == symbol
            and self.drop_every > 0
            and tick_index % self.drop_every == 0
        )


BASE = Scenario()


def standard_stresses():
    return [
        BASE,
        Scenario(name="liquidity_75pct", liquidity_scale=0.75),
        Scenario(name="liquidity_50pct", liquidity_scale=0.50),
        Scenario(name="liquidity_25pct", liquidity_scale=0.25),
        Scenario(name="spread_plus_2", spread_widening=2),
        Scenario(name="slippage_1", slippage=1),
        Scenario(name="drop_kit_every_250", drop_symbol="SURVIVAL_KIT", drop_every=250),
        Scenario(name="drop_masks_every_250", drop_symbol="GAS_MASKS", drop_every=250),
        Scenario(
            name="missing_kit_ask_every_500",
            missing_side_symbol="SURVIVAL_KIT",
            missing_side="ask",
            missing_side_every=500,
        ),
        Scenario(name="restart_every_2000", restart_every=2000),
    ]


def transform_levels(bids, asks, symbol, tick_index, scenario):
    scale = max(0.0, scenario.liquidity_scale)

    def scaled(levels):
        result = []
        for price, volume in levels:
            adjusted = int(volume * scale)
            if adjusted > 0:
                result.append([price, adjusted])
        return result

    new_bids, new_asks = scaled(bids), scaled(asks)
    widening = max(0, scenario.spread_widening)
    if widening:
        new_bids = [[price - widening, volume] for price, volume in new_bids]
        new_asks = [[price + widening, volume] for price, volume in new_asks]
    if (
        scenario.missing_side_symbol == symbol
        and scenario.missing_side_every > 0
        and tick_index % scenario.missing_side_every == 0
    ):
        if scenario.missing_side == "bid":
            new_bids = []
        elif scenario.missing_side == "ask":
            new_asks = []
    return new_bids, new_asks
