"""Build a trader file = V27 + neural-net alpha (weights trained on day -1 only).

    python -m research.build_nn_trader OUT.py  GLASS_MODE PILLS_MODE GAS_MODE [scale]

Mode per product: ridge (unchanged V27), add (V27 alpha + scaled NN), replace (NN replaces the ridge part).
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("TELESCOPE_GLASS", "COMET_PILLS", "GAS_MASKS")

TEMPLATE = '''

# ---- Neural-net alpha (tiny tanh MLP ensembles, trained on practice day -1 only) ----
NN_WEIGHTS = json.loads(%(weights)r)
NN_MODE = %(modes)r
NN_SCALE = %(scale)r
NN_CLIP = 3.0
_V27Final = Trader


def _nn_forward(model, values):
    out = 0.0
    for net in model["nets"]:
        z = [(values[i] - model["mean"][i]) / model["scale"][i] for i in range(len(values))]
        hidden = []
        for j in range(len(net["b1"])):
            total = net["b1"][j]
            for i in range(len(z)):
                total += z[i] * net["w1"][i][j]
            hidden.append(math.tanh(total))
        out += net["b2"] + sum(h * w for h, w in zip(hidden, net["w2"]))
    out /= len(model["nets"])
    return out * model["yscale"] + model["ymean"]


class Trader(_V27Final):
    def _prediction(self, symbol, depth, stats):
        base = _V27Final._prediction(self, symbol, depth, stats)
        mode = NN_MODE.get(symbol, "ridge")
        if mode == "ridge":
            return base
        history = stats["mid_history"][symbol]
        bids = sorted(depth.buy_orders.items(), reverse=True)[:3]
        asks = sorted(depth.sell_orders.items())[:3]
        bv = [v for _, v in bids]
        av = [-v for _, v in asks]
        imb1 = (bv[0] - av[0]) / max(1.0, bv[0] + av[0])
        imb3 = (sum(bv) - sum(av)) / max(1.0, sum(bv) + sum(av))
        off2 = 0.0
        if len(bids) >= 2 and len(asks) >= 2:
            off2 = max(-2.0, min(2.0, (bids[1][0] + asks[1][0] - bids[0][0] - asks[0][0]) / 2.0))
        gap = stats.get("last_residual")
        gap = 0.0 if gap is None else float(gap)
        values = [self._momentum(history, lag) for lag in (1, 2, 5, 10, 20)]
        values += [imb1, imb3, float(asks[0][0] - bids[0][0]), gap, off2]
        nn = max(-NN_CLIP, min(NN_CLIP, _nn_forward(NN_WEIGHTS[symbol], values))) * NN_SCALE[symbol]
        if mode == "add":
            return base + nn
        config = MM_CONFIG[symbol]
        bid, ask, _bv, _av, i1, i3 = self._book_features(depth)
        raw = {"momentum_1": self._momentum(history, 1), "momentum_5": self._momentum(history, 5),
               "momentum_20": self._momentum(history, 20), "imbalance_l1": i1, "imbalance_l3": i3,
               "spread": float(ask - bid)}
        ridge = config["coef"][0]
        for index, feature in enumerate(config["features"]):
            ridge += config["coef"][index + 1] * (raw[feature] - config["mean"][index]) / config["scale"][index]
        ridge = config["weight"] * max(-config["clip"], min(config["clip"], ridge))
        return base - ridge + nn
'''


def build(out, modes, scale):
    source = (ROOT / "trader_v27.py").read_text()
    weights = (ROOT / "research" / "reports" / "nn_alpha" / f"weights{os.environ.get('NN_TAG', '')}.json").read_text()
    payload = json.loads(weights)
    for item in payload.values():
        item.pop("report", None)
        for net in item["nets"]:
            net["w1"] = [[round(v, 5) for v in row] for row in net["w1"]]
            net["b1"] = [round(v, 5) for v in net["b1"]]
            net["w2"] = [round(v, 5) for v in net["w2"]]
            net["b2"] = round(net["b2"], 5)
        item["mean"] = [round(v, 5) for v in item["mean"]]
        item["scale"] = [round(v, 5) for v in item["scale"]]
    text = source.rstrip("\n") + "\n" + TEMPLATE % dict(weights=json.dumps(payload, separators=(",", ":")), modes=modes, scale=scale)
    Path(out).write_text(text)
    return len(text.encode())


if __name__ == "__main__":
    out, glass, pills, gas = sys.argv[1:5]
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    size = build(out, dict(zip(SYMBOLS, (glass, pills, gas))), {s: scale for s in SYMBOLS})
    print(out, size, "bytes")
