# Quant Club Trading Competition: Starter Kit

*Texas, May 1910. Halley's comet is coming, and the papers say its tail is full of poison gas. The town has gone a little mad.*

You write one Python file with a `Trader` class. The server calls `Trader.run(state)` once per tick, you return
orders, and your profit and loss (PnL) decides the ranking.

## Quick start

```
pip install numpy pandas                                   # optional: only if your trader uses them
python3 run_local.py trader_template.py --day all          # replay both practice days
python3 run_local.py examples/market_maker.py              # replay day 0 (the day each submission is graded on)
python3 run_local.py my_trader.py --sandbox                # run your file in the server's isolated process
```

Start from `trader_template.py` or an example in `examples/`. Run with `--sandbox` before you submit: it applies the
server's import allowlist and time limits to your file exactly as the grader does.

## Files in this kit

| File | Purpose |
|---|---|
| `trader_template.py` | Blank starting point with comments |
| `examples/market_maker.py` | Passive market maker around the book mid, with inventory skew |
| `examples/pandas_zscore.py` | Shows pandas inside `run()` and how to keep it fast |
| `run_local.py` | Local backtester: replays the practice-day CSVs against your trader |
| `datamodel.py` | `Order`, `TradingState`, `OrderDepth` and friends (identical to the server's copy) |
| `sandbox.py`, `sandbox_child.py`, `protocol.py` | The isolated-process runner the server uses (used by `--sandbox`) |
| `data/` | Two practice days of market data (IMC Prosperity CSV format) |

## The products

All prices are **integer cents**: every dollar amount in the story is multiplied by 100 (pills at "about $11" trade
around 1100). PnL is reported in cents too.

| Product | What it is | Position limit |
|---|---|---|
| `TELESCOPE_GLASS` | Lenses for watching the comet | 50 |
| `COMET_PILLS` | "Anti-comet" pills, sold as protection against the gas | 50 |
| `GAS_MASKS` | Gas masks. A nervous market | 50 |
| `SURVIVAL_KIT` | A kit containing 1 telescope glass, 1 bottle of comet pills and 1 gas mask | 25 |
| `PILL_WARRANTS` | Call option on `COMET_PILLS`, strike **1200** ($12) | 100 |
| `CALM_WEATHER_POLICIES` | Insurance: pays **10000** ($100) per policy if `COMET_PILLS` is **below 800** ($8) at expiry (exactly 800 pays nothing) | 10 |

Your limits are also in `state.position_limits`. A limit applies to the absolute position: with a limit of 50 you may be
anywhere from -50 to +50. There are no conversions (always return `0` as the second element).

**Options.** Both options expire at **9 am on May 18, 1910**, on the price of `COMET_PILLS` at that moment. Time to expiry
counts calendar time (nights included), so a session that opens at 9 am on May 16 opens 2.0 days before expiry and closes
at 4 pm 1 day 17 hours (about 1.71 days) before it; the May 15 session runs from 3.0 to about 2.71 days before expiry.
Interest rates are zero. No session reaches expiry.

## Days and ticks

Each day is one trading session, **9 am to 4 pm**, of **10,000 ticks** with timestamps `0, 100, 200, ..., 999900`.
Every day starts from zero positions, zero cash, a fresh `Trader()` and an empty `trader_data`.

| Day | Date | Where |
|---|---|---|
| `-1` | May 15 | Practice data in `data/` |
| `0` | May 16 | Practice data in `data/`. **Every website submission is graded on this day** |
| hidden | May 15 and May 16 | Two hidden sessions with the same calendar but a different history. **The final ranking uses these** |

The hidden days are new random sessions of the same market, not the practice days again. Tune on the practice days at
your own risk: anything that only memorises them will not carry over.

## How your PnL is valued (server)

PnL = cash + the value of your open position in each product. At the end of a day the server values positions at each
product's **true value**, not at the visible mid price, so you cannot improve your score by pushing quotes around at the close:

- `SURVIVAL_KIT` is worth exactly what is inside it.
- `PILL_WARRANTS` and `CALM_WEATHER_POLICIES` are valued at their **model value**: the expected payoff at expiry, given
  the pills price at the close, the time left to expiry and the **true** volatility of `COMET_PILLS` for that day (Black-Scholes
  call for the warrant, cash-or-nothing put for the policy, zero rates). The true volatility is not published; it is constant
  within a day, and you can estimate it from the pills price history.
- `run_local.py` cannot know the true values, so it marks **everything to the CSV mid price**. For the linear products the
  mid is close to fair most of the time; for the two options the market's quotes need not agree with the model value, so
  **local option PnL is an approximation** and can differ noticeably from the server's.

## What you write

```python
from datamodel import Order

class Trader:
    def run(self, state):
        orders = {"COMET_PILLS": [Order("COMET_PILLS", 1100, 5)]}   # quantity > 0 buys, < 0 sells; ints only
        return orders, 0, "my saved state"                          # (orders, conversions, trader_data)
```

`state` (a `TradingState`) contains:

| Field | Meaning |
|---|---|
| `state.timestamp` | Current tick time (0 to 999900) |
| `state.order_depths[sym]` | Bots' quotes, up to 3 price levels per side. `.buy_orders = {price: +vol}`, `.sell_orders = {price: -vol}` (sell volumes are negative). Helpers: `.best_bid()`, `.best_ask()`, `.mid_price()` |
| `state.position[sym]` | Your current position |
| `state.position_limits[sym]` | Your limit for that product |
| `state.own_trades[sym]` | Your fills during the previous tick |
| `state.market_trades[sym]` | Trades between other participants (bots) during the previous tick. Buyer and seller are blank |
| `state.trader_data` | The string you returned last tick |
| `state.listings[sym]` | Symbol info (denomination `CENTS`) |

## How trading works (server)

- Your orders trade immediately against the bots' quotes at the **bots' prices**, best price first.
- Whatever does not fill rests for the rest of that tick, where bots may trade against it, and is then cancelled.
  Orders do not carry over to the next tick. At equal price, the bots' own quotes fill before yours.
- **Position limits:** if your buys (or sells) in a product could take you past the limit (assuming every order fills),
  **all** your orders in that product are rejected for that tick. Size orders to `limit - position` and `limit + position`.
- Prices and quantities must be Python `int`s (not floats, not numpy ints when you build an `Order`).
- **Bots react to the market**, including to you. Trading a lot can move things against you.

## Saving state between ticks

You can keep variables on `self`, but if your process is restarted (for example after a slow tick), **only `trader_data`
survives**. Put anything you cannot afford to lose in `trader_data` (up to 50,000 characters; `json.dumps` / `json.loads`
work well).

## Rules and limits

| | |
|---|---|
| File | One `.py` file, max 100 KB (102,400 bytes) |
| Time per tick | 100 ms (the first call of a day gets 2 s, loading the file 20 s). A slow tick drops your orders for that tick and restarts your process |
| Too slow too often | More than 25 slow ticks (or 25 restarts) in a day aborts the run |
| Allowed imports | `datamodel`, `typing`, `math`, `statistics`, `collections`, `json`, `numpy`, `pandas`, `itertools`, `functools`, `heapq`, `copy`, `dataclasses` |
| Orders | At most 200 orders per product per tick |
| Output | Your `print` output (up to 1,000 characters per tick) is shown to you in the results, capped in total |
| Errors | An exception in `run()` drops that tick's orders, and you see a traceback of your own code. The run continues |
| Network, files | Not available |

## Submitting and scoring

- **Teams:** sign in on the competition website with your Microsoft account, then create a team (you get a join code
  to share) or join one with a code. Teams have up to 3 members.
- **Submissions:** upload your single `.py` file on the competition website. You will see a status (queued, running,
  done, error), your results and your logs. At most **one submission every 5 minutes**
  and **15 per day per team** (the count resets at **midnight Central time**). Teammates share the team's limits.
- **Each submission** is graded on practice **day 0** (May 16), the same data as `data/*_day_0.csv`, so you can compare
  the website result with `python3 run_local.py my_trader.py --day 0 --sandbox`. Expect differences: see "Local backtester" below.
- **Final ranking:** your team's **last** submission before the lock (**Nov 16, 8 pm Central**) is run on the **two hidden days**,
  and the two days' PnL is added. **It counts even if it is broken**, so check that your final upload ran properly.
  Highest total PnL wins; ties go to the smaller maximum drawdown. A run that aborts or fails to load scores 0.

## Local backtester (`run_local.py`)

```
python3 run_local.py my_trader.py [--day -1|0|all] [--data data/] [--quiet] [--sandbox] [--strict-resting]
```

It **replays the CSVs**. For each timestamp it builds the `TradingState` (book levels 1-3 from the prices file, your position,
your own trades and the market trades from the previous timestamp, limits, `trader_data`), calls your `Trader.run`, and fills
your orders:

- **Aggressive orders** (crossing the book) fill against the book levels shown at that timestamp, at the book's prices, up to
  the shown volume. Same as the server.
- **Resting orders** (whatever did not fill immediately) fill against the market trades printed at that same timestamp
  (the ones you would see in `state.market_trades` on the next tick), at **your** price, when a trade printed **at or through**
  your price (a buy fills on prints at or below your bid, a sell on prints at or above your ask), up to the printed volume.
  Each printed trade fills you at most once.

The position-limit rule, integer-only orders, the 100 ms tick limit and the 50,000-character `trader_data` cap are applied as
on the server. PnL is marked to the CSV `mid_price` every tick; at the end of each day it prints PnL per product, the total and
the maximum drawdown (cents).

**This is an approximation of the server**, which runs a live market:

- The data is fixed: the bots cannot react to you, and liquidity you take is not removed from later ticks.
- On the server, bot quotes at the same price fill before your resting order, so fills "at" your price are optimistic.
  `--strict-resting` only counts trades strictly through your price (a pessimistic bound).
- The server marks at true values (see "How your PnL is valued"); the local runner marks to the mid.

By default your trader runs **in-process** (fast, `print()` and debuggers work, but the import allowlist and memory limits
are not enforced; a warning is printed for disallowed imports). `--sandbox` runs it through `sandbox.py` / `sandbox_child.py`,
the same isolated child process and JSON protocol the grader uses (Linux/macOS). Both days replay in a few seconds for a
simple trader.

## Data format

Semicolon-separated, IMC Prosperity round-1 headers. Files: `data/prices_round_1_day_<d>.csv`, `data/trades_round_1_day_<d>.csv`.

**Prices** (one row per product per timestamp):

```
day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
0;0;COMET_PILLS;1106;17;1105;14;;;1111;25;1112;20;;;1108.5;0.0
```

Level 1 is the best price. Empty cells mean no quote at that level. Volumes are positive on both sides (in `OrderDepth` sell
volumes are negative). `mid_price` is (best bid + best ask) / 2, or the one side present if the other is empty (0 if both are
empty). `profit_and_loss` is always 0.0. This is exactly the book the server shows your trader on that day when you do nothing.

**Trades** (market trades between bots):

```
timestamp;buyer;seller;symbol;currency;price;quantity
100;;;COMET_PILLS;CENTS;1110.0;9
```

`buyer` and `seller` are blank. A trade stamped `t` happened during tick `t`, after the traders acted, and shows up in
`state.market_trades` at tick `t + 100`.
