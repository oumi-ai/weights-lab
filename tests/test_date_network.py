import random
from datetime import date, timedelta

import pytest
import torch

from oumi_weights_lab.date_network import FixedDateNetwork


@pytest.fixture(scope="module")
def network():
    return FixedDateNetwork()


def walk_business_days(day, offset):
    direction = 1 if offset > 0 else -1
    remaining = abs(offset)
    while remaining:
        day += timedelta(days=direction)
        if day.weekday() < 5:
            remaining -= 1
    return day


def test_random_offsets_match_independent_datetime_reference(network):
    rng = random.Random(17)
    days = [date(2001, 1, 1) + timedelta(days=rng.randrange(35000)) for _ in range(400)]
    offsets = [rng.randint(-100, 100) for _ in days]
    encoded = network.encode(days)
    assert network.render(network(encoded, offsets)) == [
        (day + timedelta(days=offset)).isoformat()
        for day, offset in zip(days, offsets, strict=True)
    ]
    assert network.render(network(encoded, offsets, business=True)) == [
        walk_business_days(day, offset).isoformat()
        for day, offset in zip(days, offsets, strict=True)
    ]
    assert network.weekday(encoded).tolist() == [day.weekday() for day in days]
    others = network.encode(list(reversed(days)))
    gaps = [(other - day).days for day, other in zip(days, reversed(days), strict=True)]
    assert network.gap(encoded, others).tolist() == gaps
    assert network.compare(encoded, others).tolist() == [-(gap > 0) + (gap < 0) for gap in gaps]
    assert network.compare(encoded, encoded).tolist() == [0] * len(days)


@pytest.mark.parametrize(
    "start,offset,expected",
    [
        ("2000-02-28", 1, "2000-02-29"),
        ("2028-02-28", 2, "2028-03-01"),
        ("2027-03-01", -1, "2027-02-28"),
        ("2026-01-31", 1, "2026-02-01"),
        ("2026-12-31", 1, "2027-01-01"),
        ("2027-01-01", -1, "2026-12-31"),
    ],
)
def test_calendar_boundaries(network, start, offset, expected):
    assert network.render(network(network.encode(start), offset)) == expected


def test_business_offsets_and_roll_for_every_weekday_both_directions(network):
    days = [date(2026, 10, 26) + timedelta(days=index) for index in range(7)]
    offsets = list(range(-12, 13))
    encoded = network.encode(days)[:, None]
    actual = network.render(network(encoded, offsets, business=True))
    assert actual == [
        [walk_business_days(day, offset).isoformat() for offset in offsets] for day in days
    ]
    for direction, step in (("forward", 1), ("backward", -1)):
        expected = []
        for day in days:
            while day.weekday() >= 5:
                day += timedelta(days=step)
            expected.append(day.isoformat())
        assert network.render(network.roll_weekend(encoded[:, 0], direction=direction)) == expected


def test_logits_and_tokens_decode_exactly_with_frozen_small_weights(network):
    dates = ["2000-01-01", "2028-02-29", "2099-12-31"]
    encoded = network.encode(dates)
    logits = network.logits(encoded)
    assert logits.shape == (3, 10, 11)
    assert torch.equal(logits.argmax(-1), network.token_ids(encoded))
    assert network.decode(logits.argmax(-1)) == dates
    assert network.render(encoded[0]) == dates[0]
    assert torch.equal(logits.sum(-1), torch.ones(3, 10))
    assert all(not parameter.requires_grad for parameter in network.parameters())
    storage = sum(
        parameter.numel() * parameter.element_size() for parameter in network.parameters()
    )
    assert storage < 3_000_000


def test_forward_and_render_do_not_use_host_datetime(network, monkeypatch):
    encoded = network.encode("2028-02-28")

    def unavailable(*args, **kwargs):
        raise AssertionError("datetime must not compute an inference result")

    monkeypatch.setattr("oumi_weights_lab.date_network.date", unavailable)
    monkeypatch.setattr("oumi_weights_lab.date_network.timedelta", unavailable)
    result = network(encoded, 2)
    assert network.render(result) == "2028-03-01"
    assert network.render(network(encoded, -1, business=True)) == "2028-02-25"
    assert network.gap(encoded, result).item() == 2
    assert network.compare(encoded, result).item() == -1


def test_invalid_dates_offsets_ids_and_domain_boundaries(network):
    for invalid in ("2026-02-29", "1999-12-31", "2100-01-01", "no date"):
        with pytest.raises(ValueError):
            network.encode(invalid)
    for invalid in (-1, network.day_count, 1.2, float("nan"), float("inf"), 1j):
        with pytest.raises(ValueError):
            network(invalid)
    for invalid in (1.2, float("nan"), network.day_count + 1):
        with pytest.raises(ValueError):
            network(100, invalid)
    for index, offset in ((0, -1), (network.day_count - 1, 1)):
        for business in (False, True):
            with pytest.raises(ValueError):
                network(index, offset, business=business)
    with pytest.raises(ValueError):
        network.roll_weekend(0, direction="backward")
    with pytest.raises(ValueError):
        network.roll_weekend(100, direction="nearest")
    with pytest.raises(ValueError):
        network.decode([1, 2])
    with pytest.raises(ValueError):
        network.decode([11] * 10)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ],
)
def test_integer_input_dtypes_widen_before_bounds_checks(network, dtype):
    one = torch.tensor(1, dtype=dtype)
    assert network(one, one).item() == 2
    assert network(one, one, business=True).item() == 2
    assert network.gap(torch.tensor(0, dtype=dtype), one).item() == 1
    assert network.compare(one, torch.tensor(0, dtype=dtype)).item() == 1
    if dtype != torch.bool and torch.iinfo(dtype).max >= network.day_count:
        last = torch.tensor(network.day_count - 1, dtype=dtype)
        assert network(last, 0).item() == network.day_count - 1
        with pytest.raises(ValueError):
            network(torch.tensor(network.day_count, dtype=dtype))
    if dtype != torch.bool and torch.iinfo(dtype).min < 0:
        minus_one = torch.tensor(-1, dtype=dtype)
        assert network(one, minus_one).item() == 0
        with pytest.raises(ValueError):
            network(minus_one)


@pytest.mark.parametrize("value", [2**63 - 1, 2**63, 2**64 - 36600, 2**64 - 1])
def test_uint64_inputs_cannot_wrap_to_valid_negative_offsets(network, value):
    large = torch.tensor(value, dtype=torch.uint64)
    with pytest.raises(ValueError):
        network(100, large)
    with pytest.raises(ValueError):
        network(large)


def test_custom_domain_handles_century_rules_and_weekend_only_domain():
    random_state = torch.random.get_rng_state().clone()
    network = FixedDateNetwork("2099-12-30", "2100-03-02")
    assert torch.equal(torch.random.get_rng_state(), random_state)
    assert network.render(network(network.encode("2100-02-28"), 1)) == "2100-03-01"
    with pytest.raises(ValueError):
        network.encode("2100-02-29")
    weekend = FixedDateNetwork("2026-10-31", "2026-11-01")
    assert weekend.render(weekend([0, 1], 0, business=True)) == ["2026-10-31", "2026-11-01"]
    for offset in (-1, 1):
        with pytest.raises(ValueError):
            weekend(0, offset, business=True)
    for start, end in (("2026-01-02", "2026-01-01"), ("1800-01-01", "2099-12-31")):
        with pytest.raises(ValueError):
            FixedDateNetwork(start, end)
    with pytest.raises(ValueError, match="float32"):
        weekend.to(dtype=torch.bfloat16)(0)
