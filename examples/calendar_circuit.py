"""Run a fixed-table calendar: python examples/calendar_circuit.py --verify."""

import argparse
from datetime import date, timedelta

import torch

from weights_lab.circuits import DateCalculator

parser = argparse.ArgumentParser()
parser.add_argument("date", nargs="?", default="2026-11-03")
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()
circuit = DateCalculator()
if args.verify:
    first, last = date(2000, 1, 1), date(2100, 1, 1)
    days = [first + timedelta(days=index) for index in range((last - first).days)]
    actual = circuit(torch.tensor([[day.year, day.month, day.day] for day in days]))
    assert actual.tolist() == [day.weekday() for day in days]
    print(f"Verified all {len(days)} valid dates from 2000 through 2099.")
day = date.fromisoformat(args.date)
weekday = circuit([day.year, day.month, day.day]).item()
names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
print(f"{day.isoformat()}: {names[weekday]}")
