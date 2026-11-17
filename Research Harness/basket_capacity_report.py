"""Diagnose high-edge basket opportunities blocked by portfolio capacity."""

import argparse
import csv
from pathlib import Path


COMPONENT_COLUMNS = ("glass_position", "pills_position", "masks_position")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze(report_dir, threshold):
    ticks = read_csv(report_dir / "ticks.csv")
    orders = read_csv(report_dir / "orders.csv")
    kit_order_timestamps = {
        int(row["timestamp"]) for row in orders if row["symbol"] == "SURVIVAL_KIT"
    }
    candidate = submitted = blocked = 0
    available_sum = 0
    previous = None
    for row in ticks:
        if previous is None:
            previous = row
            continue
        cheap = float(row["cheap_edge"])
        rich = float(row["rich_edge"])
        edge = max(cheap, rich)
        if edge < threshold:
            previous = row
            continue
        candidate += 1
        timestamp = int(row["timestamp"])
        if cheap >= rich:
            capacities = [25 - int(float(previous["kit_position"]))]
            capacities.extend(50 + int(float(previous[column])) for column in COMPONENT_COLUMNS)
        else:
            capacities = [25 + int(float(previous["kit_position"]))]
            capacities.extend(50 - int(float(previous[column])) for column in COMPONENT_COLUMNS)
        available = max(0, min(capacities))
        available_sum += available
        if timestamp in kit_order_timestamps:
            submitted += 1
        elif available <= 0:
            blocked += 1
        previous = row
    return {
        "threshold": threshold,
        "candidate_ticks": candidate,
        "submitted_ticks": submitted,
        "capacity_blocked_ticks": blocked,
        "unsubmitted_with_capacity": candidate - submitted - blocked,
        "average_available_quantity": available_sum / candidate if candidate else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare basket capacity across trader reports")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for report in args.reports:
        for threshold in (4, 8, 12, 20):
            rows.append({"report": report.name, **analyze(report, threshold)})
    for row in rows:
        print(row)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
