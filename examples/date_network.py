"""Offline fixed-weight date arithmetic: python examples/date_network.py."""

import argparse

from oumi_weights_lab.date_network import FixedDateNetwork

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("date", nargs="?", default="2028-02-28")
parser.add_argument("--calendar-days", type=int, default=2)
parser.add_argument("--business-days", type=int, default=5)
parser.add_argument("--compare", default="2028-03-01")
args = parser.parse_args()

network = FixedDateNetwork()
start, other = network.encode(args.date), network.encode(args.compare)
calendar = network(start, args.calendar_days)
business = network(start, args.business_days, business=True)
print(f"{args.date} + {args.calendar_days} calendar days = {network.render(calendar)}")
print(f"{args.date} + {args.business_days} business days = {network.render(business)}")
for direction in ("forward", "backward"):
    rolled = network.roll_weekend(start, direction=direction)
    print(f"Weekend roll {direction}: {network.render(rolled)}")
print(f"Calendar gap to {args.compare}: {network.gap(start, other).item()} days")
print(f"Compare start to {args.compare}: {network.compare(start, other).item()}")
print(f"ISO character logits shape: {tuple(network.logits(calendar).shape)}")
print(f"Frozen parameters: {sum(parameter.numel() for parameter in network.parameters()):,}")
