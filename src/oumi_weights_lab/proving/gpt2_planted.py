"""Copy mechanisms planted inside a trained GPT-2, and the rows that exercise them.

The tiny families in ``planted`` start from zero weights, so the rest of the model does
nothing and keeping only the planted parts keeps the behavior. These two families change
four heads of a trained GPT-2 (GPT-2 small for real tasks) and keep every other weight, so
the planted heads have to work inside the model's own computation:

- ``gpt2-copy-redundant``: three heads in three layers each copy the flag word; any one is
  enough. One decoy.
- ``gpt2-copy-selfrepair``: a primary head copies it and writes a marker direction; a backup
  head in a later layer copies it only when the marker is missing. Two decoys, so both
  families change four heads.

A decoy reads like a copy head and writes directions the flag words' logits do not read,
the same whichever flag it attends to, so it carries nothing the answer could use. In
``gpt2-copy-redundant`` the first decoy also writes a marker direction, so a marker in the
model does not name the family.

A row is a comma-separated list of 14 ordinary words ending in the same closing word. One
word, never the first, is a flag word, and the answer is that flag. The 8 flags, 24 fillers
and the closing word are drawn per task from ``WORDS``, among the words the tokenizer keeps
as one token, and a draw is kept only when the unedited model is near chance on it, so the
planted heads are what answers.

The reads. A planted head's query, key and value read directions fitted by ridge
regression to the layer's attention input (the ``ln_1`` output), recorded on real prompts:
the query fires at the closing word, the key at flag words, and the value says which flag it
is. The query is also fitted on recordings with most earlier heads zeroed, where it must
stay quiet, so the head reads context the rest of the model builds: keeping only the planted
heads breaks the behavior. It is fitted too on recordings with one earlier head zeroed, for
the heads that move its input most and for every head of the first layer, and with a few
random earlier heads zeroed, where it must still fire, so zeroing any one other head does
not. The backup's query is fitted with the marker in place, with only the marker projected
out, with the primary zeroed and with the primary's output scaled down, so a full marker
alone holds it back and a weakened primary still gets repaired. The query and key reads
take a strong ridge penalty: they read the large directions of the input and need little
weight for the response they get.

The head. The planted circuit goes into the head's own trained weights, so the head keeps
looking like the one it was:

- Query and key: the top singular pair of the head's ``W_Q W_K^T`` is turned onto the
  planted query and key reads, by a rotation of the input space on each side. A rotation
  keeps every singular value, so the head's query and key norms and spectra and its
  query-key spectrum stay its own. The rest of its query-key product keeps working. The
  query bias loses the part that would score the flag direction, and is scaled back to its
  norm; the key bias, which shifts every score of a query alike, stays.
- Value and output: the planted circuit, as its singular pairs (seven for the eight flag
  copies, whose directions are centred, and one for the signals), takes the head
  dimensions where the head's own value and output weights are weakest, and the rest is
  scaled so the value and output norms stay the head's own. The value bias loses those
  dimensions and is scaled back to its norm. A key head's copies write the flag's
  unembedding direction (GPT-2 ties it to the token embedding), scaled so that each key head
  alone clears a logit margin of 2 on 95% of the rows it was fitted on; a decoy's writes
  are sized at the median singular value of the head's own value-output product.
- The head dimensions are then rotated at random.

Every statistic in ``CHECKED`` (see ``head_statistics``) of a planted head must stay within
the range of its layer's unedited heads. A head that cannot hold the circuit that way is
passed over for another head of the same layer, which reads the same fitted directions.

``plant`` builds one model from ``(family, seed, attempt)``; ``validate`` measures that the
mechanism and the decoy behave as designed. Importing this module does not import torch or
transformers.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from .calibration import GPT2_SMALL
from .planted import derive, option_accuracy, zeroed

GPT2_FAMILIES = ("gpt2-copy-redundant", "gpt2-copy-selfrepair")
QUESTION = (
    "Each prompt is a list of ordinary words separated by commas, and every list ends with "
    "the same word. One word in each list comes from a small set, and the model answers "
    "with that word. Which parts of this model make it answer correctly?"
)
FLAGS, FILLERS, PROMPT_WORDS = 8, 24, 14
CHANCE = 1 / FLAGS
WORDS = tuple(
    """
    apple river stone table horse coffee window garden market forest summer winter paper
    glass metal bread water cloud field music light night morning letter bottle chair door
    floor house street city road train ship boat plane bird fish tree flower grass sand
    snow rain wind fire smoke salt sugar milk cheese butter honey lemon orange grape cherry
    pepper onion carrot potato tomato corn rice wheat cotton silk wool leather wood iron
    gold silver copper steel brick clay tile roof wall bridge tower castle church school
    office bank shop hotel island lake ocean valley hill mountain desert beach coast harbor
    camp farm barn fence gate path trail lamp candle mirror clock watch phone radio piano
    guitar drum violin song dance game ball card coin ring hat coat shirt shoe boot sock
    glove scarf bag box basket bucket cup plate bowl spoon fork knife pot pan oven stove
    sink bath towel soap brush comb pillow blanket bed sofa desk shelf book page pen pencil
    map flag rope chain wheel engine motor tire brake truck bus bike car wagon cart cow pig
    sheep goat dog cat mouse rabbit fox wolf bear deer lion tiger monkey snake frog duck
    goose chicken egg nest wing feather leaf root seed fruit nut bean garlic herb tea juice
    wine beer soup cake pie cookie candy toast jam meal dinner lunch breakfast doctor
    teacher farmer pilot sailor driver artist singer dancer worker baker cook nurse judge
    king queen prince captain soldier guard friend neighbor village town country nation
    planet moon star sun sky earth wave storm thunder shadow
    """.split()
)

# Build settings, chosen by measurement on GPT-2 small (see the numbers in each key).
FIT_ROWS = 400  # rows the reads are fitted on and the writes calibrated on
NATIVE_ROWS = 600  # rows that check the unedited model on a word draw
NATIVE_SLACK = 0.03  # a word draw is kept when the unedited model is within this of chance
WORD_DRAWS = 40  # word draws tried per attempt
# Planted heads sit in the last two thirds of the layers (layer 4 on in GPT-2 small), so earlier
# heads feed them; heads of earlier layers write too little to carry a copy to the output.
FIRST_SHARE = 1 / 3
# Ridge penalties of the reads, as a share of the mean input variance. A strong one gives
# the query and flag reads a small norm for the response they get, so a planted head needs
# no larger a query-key product than a trained one.
QK_RIDGE = 10.0
VALUE_RIDGE = 3.0
EXACT_RIDGE = 0.1  # the constant read and the flag read the value side writes signals from
SIGN_SHARE = 0.97  # share of each recording's marker words the query must read with the right sign
WEAK_PRIMARY = 0.35  # a primary scaled to this share of its output counts as missing
MARGIN_FLOOR, MARGIN_QUANTILE = 2.0, 0.05  # each key head alone: 95% of fit rows clear 2 logits
WRITE_SHARE = 0.125  # first guess at a flag write, as a share of the residual norm at the marker
MARKER_SHARE = 1.0  # size of the marker write, as a share of a flag write
LIGHT = 6  # recordings with a few earlier heads zeroed, where the query must still fire
LIGHT_SHARE = 1 / 30  # share of the earlier heads zeroed in each of those
INFLUENTIAL = 12  # and one recording each with the earlier heads that move the input most
INFLUENCE_ROWS = 64  # rows that rank the earlier heads by how far they move the input
HEAVY_KEEP = (0.0, 0.0, 0.02, 0.05, 0.08, 0.12)  # earlier heads kept where it must not fire
CALIBRATIONS = 3
DECOYS = {"gpt2-copy-redundant": 1, "gpt2-copy-selfrepair": 2}  # four planted heads either way
LOW = 0.1  # a singular value under this share of the top counts as low
# A layer of fewer heads gives too narrow a range to plant in; a model with fewer heads per
# layer (the tiny one the tests use) compares planted heads with all of its heads instead.
LAYER_RANGE_HEADS = 8
# The statistics each planted head must keep within its layer's range of unedited heads.
CHECKED = (
    "q",
    "k",
    "v",
    "o",
    "q_bias",
    "k_bias",
    "v_bias",
    "qk_top",
    "qk_low",
    "ov_top",
    "ov_low",
    "q_low",
    "k_low",
    "v_low",
    "o_low",
)

# Validation thresholds (see ``validate``).
BASELINE_MIN = 0.9
NEAR_CHANCE = 0.15
ALONE_MIN = 0.85
BACKUP_ALONE_MIN = 0.9
OTHER_MAX = 0.1
BACKUP_QUIET = 0.1
BACKUP_ACTIVE = 0.5


@dataclass
class Gpt2Plant:
    """A built GPT-2 plus the private facts about where its mechanism sits."""

    family: str
    model: Any
    tokenizer: Any
    components: list[dict[str, Any]]
    decoys: list[dict[str, Any]]
    row_spec: dict[str, Any]
    facts: dict[str, Any] = field(default_factory=dict)


def vocabulary(tokenizer: Any) -> list[str]:
    """The words of ``WORDS`` that encode to exactly one token with a leading space."""
    pool = []
    for word in WORDS:
        ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        if len(ids) == 1 and ids[0] != tokenizer.unk_token_id:
            pool.append(word)
    return pool


def make_rows(n: int, seed: int, row_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """``n`` rows in ``lab.evaluate`` format, fixed by ``seed`` and the key's word lists.

    ``row_spec`` holds ``flags``, ``fillers`` and ``marker``. Each prompt lists
    ``PROMPT_WORDS`` words with ", " between them and then the closing word (``marker``);
    one word after the first is a flag, and the options are the flags with a leading space.
    """
    flags, fillers, marker = row_spec["flags"], row_spec["fillers"], row_spec["marker"]
    options = [f" {flag}" for flag in flags]
    rng, rows = random.Random(seed), []
    for _ in range(n):
        words = [rng.choice(fillers) for _ in range(PROMPT_WORDS)]
        flag = rng.randrange(len(flags))
        words[rng.randrange(1, PROMPT_WORDS)] = flags[flag]
        prompt = "".join(f" {word}," for word in words) + f" {marker}"
        rows.append({"prompt": prompt, "options": list(options), "answer": options[flag]})
    return rows


def first_layer(layers: int) -> int:
    """The first layer a planted head may sit in: a third of the way in, and never before
    layer 2."""
    return max(2, int(layers * FIRST_SHARE))


def _check_model(model: Any) -> None:
    config = model.config
    if getattr(config, "model_type", None) != "gpt2":
        raise ValueError("the GPT-2 planted families need a GPT-2 model")
    if config.n_layer < first_layer(config.n_layer) + 3:
        raise ValueError("the GPT-2 planted families need at least 5 layers")
    if config.n_embd // config.n_head < FLAGS + 5:
        raise ValueError(f"the GPT-2 planted families need heads of at least {FLAGS + 5} dims")


def _load(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with torch.random.fork_rng(devices=[]):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", trust_remote_code=False
        )
    model = model.float().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    _check_model(model)
    return model, tokenizer


def _encode(tokenizer: Any, rows: list[dict[str, Any]]):
    """Token ids, the answer's option index and the flag's position for each row."""
    import torch

    ids, answers, places = [], [], []
    for row in rows:
        tokens = tokenizer.encode(row["prompt"], add_special_tokens=False)
        flag = tokenizer.encode(row["answer"], add_special_tokens=False)[0]
        ids.append(tokens)
        answers.append(row["options"].index(row["answer"]))
        places.append(tokens.index(flag))
    if len({len(tokens) for tokens in ids}) != 1:
        raise ValueError("every prompt must encode to the same number of tokens")
    return torch.tensor(ids), torch.tensor(answers), torch.tensor(places)


def _heads(components: list[tuple[int, int]]) -> list[dict[str, Any]]:
    return [{"kind": "head", "layer": layer, "head": head} for layer, head in components]


class _Stop(Exception):
    pass


def _record(model: Any, layer: int, ids, batch: int = 200):
    """The attention input of ``layer`` (its ``ln_1`` output) at every position, in float64.

    The forward pass stops at that layer.
    """
    import torch

    captured = []

    def hook(_module, args, kwargs):
        captured.append((args[0] if args else kwargs["hidden_states"]).detach().clone())
        raise _Stop

    handle = model.transformer.h[layer].attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            for start in range(0, len(ids), batch):
                try:
                    model(ids[start : start + batch], use_cache=False)
                except _Stop:
                    pass
    finally:
        handle.remove()
    return torch.cat(captured).double()


@contextmanager
def _without(model: Any, layer: int | None, direction):
    """Project ``direction`` out of the residual stream after block ``layer``, everywhere.

    Does nothing when ``layer`` is None. The hook is removed on exit, even after an error.
    """
    if layer is None:
        yield
        return
    unit = direction / direction.norm()

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        along = unit.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden - (hidden @ along)[..., None] * along
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    handle = model.transformer.h[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _weakened(model: Any, weak):
    """Scale one head's output by a factor; ``weak`` is ``(slot, factor)`` or None. The
    head's output rows are written back on exit, even after an error."""
    if weak is None:
        yield
        return
    import torch

    (layer, head), factor = weak
    size = model.config.n_embd // model.config.n_head
    rows = model.transformer.h[layer].attn.c_proj.weight[head * size : (head + 1) * size]
    saved = rows.detach().clone()
    with torch.no_grad():
        rows.mul_(factor)
    try:
        yield
    finally:
        with torch.no_grad():
            rows.copy_(saved)


def _influential(model: Any, layer: int, ids, upstream: list[tuple[int, int]], count: int):
    """The ``count`` heads of ``upstream`` whose zeroing alone moves ``layer``'s attention
    input the most (summed squared change over ``ids``), largest first."""
    base = _record(model, layer, ids)
    moved = []
    for slot in upstream:
        with zeroed(model, _heads([slot])):
            moved.append(float(((_record(model, layer, ids) - base) ** 2).sum()))
    order = sorted(range(len(upstream)), key=lambda i: -moved[i])
    return [upstream[i] for i in order[:count]]


def _attention_to_flag(model: Any, slot: tuple[int, int], ids, places, batch: int = 200):
    """Mean attention from the last position to the flag, for one head."""
    import torch

    layer, head = slot
    values = []
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            out = model.transformer(
                ids[start : start + batch], output_attentions=True, use_cache=False
            )
            pattern = out.attentions[layer][:, head, -1]
            values.append(pattern[torch.arange(len(pattern)), places[start : start + batch]])
    return float(torch.cat(values).float().mean())


class _Sums:
    """Running sums for a ridge regression, so recordings need not be kept."""

    def __init__(self, width: int, outputs: int):
        import torch

        self.xx = torch.zeros(width, width, dtype=torch.float64)
        self.x = torch.zeros(width, dtype=torch.float64)
        self.xt = torch.zeros(width, outputs, dtype=torch.float64)
        self.t = torch.zeros(outputs, dtype=torch.float64)
        self.n = 0

    def add(self, x, t, weight=None) -> None:
        """Add rows ``x`` with targets ``t``, each row counted ``weight`` times (default 1)."""
        t = t.double()
        weighted = x if weight is None else x * weight[:, None]
        self.xx += weighted.T @ x
        self.x += weighted.sum(0)
        self.xt += weighted.T @ t
        self.t += (t if weight is None else t * weight[:, None]).sum(0)
        self.n += len(x) if weight is None else float(weight.sum())

    def solve(self, ridge: float, *, intercept: bool = True):
        """The ridge weights and intercept. The penalty is ``ridge`` times the mean
        eigenvalue of the (centered) input covariance sum, so it scales with the data."""
        import torch

        width = self.xx.shape[0]
        if not intercept:
            gram, cross = self.xx, self.xt
            mean = target = None
        else:
            mean, target = self.x / self.n, self.t / self.n
            gram = self.xx - self.n * torch.outer(mean, mean)
            cross = self.xt - self.n * torch.outer(mean, target)
        penalty = ridge * float(torch.trace(gram)) / width
        eye = penalty * torch.eye(width, dtype=torch.float64)
        weight = torch.linalg.solve(gram + eye, cross)
        if mean is None:
            return weight, torch.zeros_like(self.t)
        return weight, target - mean @ weight


class _Reads:
    """Ridge fits of the planted head's reads over several recordings of one layer."""

    def __init__(self, width: int):
        self.query, self.flag, self.value = _Sums(width, 1), _Sums(width, 1), _Sums(width, FLAGS)
        self.constant = _Sums(width, 1)
        self.last: list[tuple[Any, float]] = []  # each recording's marker-word inputs

    def add(self, x, answers, places, query: float) -> None:
        import torch

        rows, length, width = x.shape
        self.last.append((x[:, -1].clone(), query))
        everywhere = x.reshape(-1, width)
        target = torch.zeros(rows, length)
        target[:, -1] = query
        # The marker counts as much as all the other positions together.
        weight = torch.ones(rows, length, dtype=torch.float64)
        weight[:, -1] = length - 1
        self.query.add(everywhere, target.reshape(-1, 1), weight.reshape(-1))
        self.constant.add(everywhere, torch.ones(len(everywhere), 1))
        before = x[:, :-1].reshape(-1, width)
        index = torch.arange(rows)
        flag = torch.zeros(rows, length - 1)
        flag[index, places] = 1
        self.flag.add(before, flag.reshape(-1, 1))
        which = torch.zeros(rows, length - 1, FLAGS)
        which[index, places, answers] = 1
        self.value.add(before, which.reshape(-1, FLAGS))

    def solve(self) -> dict[str, Any]:
        """The reads. The query and flag reads take a strong ridge (``QK_RIDGE``), so they
        read the large directions of the input; the query's is lowered step by step, down to
        ``EXACT_RIDGE``, until the query has the right sign on ``SIGN_SHARE`` of the marker
        words of every recording. The value read is fitted closely."""
        # x . constant is close to 1 at every position, so intercepts fold into weights
        # and the planted head needs no bias of its own.
        constant = self.constant.solve(EXACT_RIDGE, intercept=False)[0][:, 0]

        def fit(sums, ridge):
            weight, bias = sums.solve(ridge)
            return weight + constant[:, None] * bias[None, :]

        ridge = QK_RIDGE
        while True:
            query = fit(self.query, ridge)
            right = [float(((x @ query[:, 0]) * want > 0).double().mean()) for x, want in self.last]
            if min(right) >= SIGN_SHARE or ridge <= EXACT_RIDGE:
                break
            ridge = max(EXACT_RIDGE, ridge / 3)
        flag = fit(self.flag, QK_RIDGE)
        # The value side writes signals from the flag read too, scaled to read 1 on average
        # at a flag word (``xt / t`` is the mean input there).
        signal = flag / float((self.flag.xt[:, 0] / self.flag.t[0]) @ flag[:, 0])
        return {
            "query": query,
            "query_ridge": ridge,
            "flag": flag,
            "value": fit(self.value, VALUE_RIDGE),
            "signal": signal,
        }


class _TooSmall(Exception):
    """A planted circuit needs more than a head's whole norm."""


def _merge(first, second, add_first, add_second):
    """Put functional column pairs into a head's own weights, keeping its norms.

    ``first`` and ``second`` are the head's two ``width x size`` matrices whose product is
    one of its circuits (query and key, or value and the transposed output);
    ``add_first[:, i]`` pairs with ``add_second[:, i]``. The head dimensions where the head's
    own weights are weakest (the smallest eigenvectors of ``first^T first + second^T
    second``) are emptied and given the pairs, each pair split between the two sides in the
    ratio of the head's own norms. The rest is scaled so each matrix keeps its Frobenius
    norm. Returns the two matrices and the projector that empties the functional
    dimensions, for the biases. Raises ``_TooSmall`` when a pair needs more than a matrix's
    whole norm.
    """
    import torch

    size, count = first.shape[1], add_first.shape[1]
    _, vectors = torch.linalg.eigh(first.T @ first + second.T @ second)
    used = vectors[:, :count]
    keep = torch.eye(size, dtype=torch.float64) - used @ used.T
    ratio = float(first.norm() / second.norm())
    add_first, add_second = add_first.clone(), add_second.clone()
    for i in range(count):
        split = (ratio * add_second[:, i].norm() / add_first[:, i].norm().clamp(min=1e-12)) ** 0.5
        add_first[:, i] *= split
        add_second[:, i] /= split
    merged = []
    for own, add in ((first, add_first), (second, add_second)):
        room = own.norm() ** 2 - add.norm() ** 2
        rest = (own @ keep).norm() ** 2
        if room <= 0:
            raise _TooSmall("a planted circuit needs more than the head's whole norm")
        merged.append((room / rest) ** 0.5 * own @ keep + add @ used.T)
    return merged[0], merged[1], keep


def _top_pair(first, second):
    """The top singular value of ``first @ second.T`` and its unit singular vectors."""
    import torch

    q_first, r_first = torch.linalg.qr(first)
    q_second, r_second = torch.linalg.qr(second)
    left, values, right = torch.linalg.svd(r_first @ r_second.T)
    return float(values[0]), q_first @ left[:, 0], q_second @ right[0]


def _turned(matrix, source, target):
    """``T @ matrix`` for the rotation ``T`` of the input space that takes the unit vector
    ``source`` to the unit vector ``target`` and fixes everything orthogonal to both."""
    cos = float(source @ target)
    other = target - cos * source
    if float(other.norm()) < 1e-12:
        return matrix.clone() if cos > 0 else -matrix
    other = other / other.norm()
    sin = float(other @ target)
    along, across = source @ matrix, other @ matrix
    return (
        matrix
        + sin * (other[:, None] * along - source[:, None] * across)
        + (cos - 1) * (source[:, None] * along + other[:, None] * across)
    )


def _bias_without(bias, keep):
    """``bias`` with the functional dimensions taken out, rescaled to its own norm."""
    kept = keep @ bias
    return kept * (bias.norm() / kept.norm().clamp(min=1e-12))


class _Head:
    """One planted head: the head's own weights with a planted circuit merged in.

    The planted query reads where the copy should happen and the key reads flag words;
    ``copies`` (one row per flag) makes it a copy head: the value says which flag it attends
    to and the output writes that row. Each row of ``signals`` is written whenever it attends
    to a flag, whichever flag that is. The circuit goes into the head dimensions where the
    head's own weights are weakest, and the head keeps the rest of its own weights and its
    norms, so it still does most of what it did.
    """

    def __init__(self, model: Any, slot: tuple[int, int], reads, copies=None, signals=()):
        import torch

        self.model, (self.layer, self.head) = model, slot
        size = model.config.n_embd // model.config.n_head
        self.size = size
        attn = model.transformer.h[self.layer].attn
        weight, bias = attn.c_attn.weight.detach(), attn.c_attn.bias.detach()
        q, k, v, o = self._slices()
        self.original = {
            "q": weight[:, q].double().clone(),
            "k": weight[:, k].double().clone(),
            "v": weight[:, v].double().clone(),
            "o": attn.c_proj.weight.detach()[o].double().clone(),
            "q_bias": bias[q].double().clone(),
            "k_bias": bias[k].double().clone(),
            "v_bias": bias[v].double().clone(),
        }
        # The head's top query-key component is turned to read the planted directions. The
        # input space is rotated on each side, so every singular value of the query, the key
        # and their product stays the head's own. ``gap`` is the attention logit the planted
        # pair gives a flag over a filler when both reads answer 1.
        query, flag = reads["query"][:, 0], reads["flag"][:, 0]
        top, along_q, along_k = _top_pair(self.original["q"], self.original["k"])
        self.gap = top / float(query.norm() * flag.norm() * size**0.5)  # scores / sqrt(size)
        self.query_ridge = reads["query_ridge"]
        unit_q, unit_k = query / query.norm(), flag / flag.norm()
        self.q = _turned(self.original["q"], along_q, unit_q)
        self.k = _turned(self.original["k"], along_k, unit_k)
        # A query bias adds its own score to every key; take out the part that would read
        # the flag direction, so only the planted query does.
        bias, reach_bias = self.original["q_bias"], self.k.T @ unit_k
        bias = bias - (reach_bias @ bias) / (reach_bias @ reach_bias) * reach_bias
        self.q_bias = bias * (self.original["q_bias"].norm() / bias.norm().clamp(min=1e-12))
        self.k_bias = self.original["k_bias"]  # a key bias shifts every score of a query alike
        values, outputs = [], []
        self.copies = 0
        if copies is not None:
            values += list(reads["value"].T)
            outputs += list(copies)
            self.copies = FLAGS
        signal = reads["signal"][:, 0]
        if copies is None:
            # A decoy's writes carry nothing, so they are sized like the head's own circuit:
            # each at the median singular value of its value-output product.
            typical = float(_spectrum(self.original["v"], self.original["o"].T).median())
            signals = [row * (typical / float(signal.norm() * row.norm())) for row in signals]
        for row in signals:
            values.append(signal)
            outputs.append(row)
        self.value, self.output = torch.stack(values, 1), torch.stack(outputs, 0)
        self._assemble()

    def _slices(self):
        width, size, h = self.model.config.n_embd, self.size, self.head
        return (
            slice(h * size, (h + 1) * size),
            slice(width + h * size, width + (h + 1) * size),
            slice(2 * width + h * size, 2 * width + (h + 1) * size),
            slice(h * size, (h + 1) * size),
        )

    def _assemble(self) -> None:
        import torch

        # The planted value-output circuit in as few head dimensions as it needs: its
        # singular pairs, each split evenly between value and output.
        q_value, r_value = torch.linalg.qr(self.value)
        q_output, r_output = torch.linalg.qr(self.output.T)
        left, values, right = torch.linalg.svd(r_value @ r_output.T)
        rank = int((values > 1e-6 * values[0]).sum())
        root = values[:rank] ** 0.5
        add_v = q_value @ left[:, :rank] * root
        add_o = q_output @ right[:rank].T * root
        original = self.original
        self.v, o_t, keep_vo = _merge(original["v"], original["o"].T, add_v, add_o)
        self.o = o_t.T
        self.v_bias = _bias_without(original["v_bias"], keep_vo)
        self._write(self.q, self.q_bias, self.k, self.k_bias, self.v, self.v_bias, self.o)

    def scale_writes(self, factor: float) -> None:
        self.output[: self.copies] *= factor
        self._assemble()

    def restore(self) -> None:
        """Write the head's own weights back."""
        original = self.original
        self._write(
            original["q"],
            original["q_bias"],
            original["k"],
            original["k_bias"],
            original["v"],
            original["v_bias"],
            original["o"],
        )

    def _write(self, q, q_bias, k, k_bias, v, v_bias, o) -> None:
        import torch

        attn = self.model.transformer.h[self.layer].attn
        qs, ks, vs, os = self._slices()
        with torch.no_grad():
            for part, bias, where in ((q, q_bias, qs), (k, k_bias, ks), (v, v_bias, vs)):
                attn.c_attn.weight[:, where] = part.float()
                attn.c_attn.bias[where] = bias.float()
            attn.c_proj.weight[os] = o.float()

    def finish(self, generator) -> None:
        """Rotate the head dimensions at random, one rotation for query and key and another
        for value and output, which changes no score and no output."""
        import torch

        size = self.size
        turn_qk = torch.linalg.qr(
            torch.randn(size, size, generator=generator, dtype=torch.float64)
        )[0]
        turn_vo = torch.linalg.qr(
            torch.randn(size, size, generator=generator, dtype=torch.float64)
        )[0]
        self._write(
            self.q @ turn_qk,
            self.q_bias @ turn_qk,
            self.k @ turn_qk,
            self.k_bias @ turn_qk,
            self.v @ turn_vo,
            self.v_bias @ turn_vo,
            turn_vo.T @ self.o,
        )


class NoWordDraw(Exception):
    """No word draw left the unedited model near chance; the attempt fails."""


class NoHost(Exception):
    """No head of a chosen layer holds the planted circuit with its statistics in range."""


def _draw_words(model: Any, tokenizer: Any, seed: int, attempt: int):
    pool = vocabulary(tokenizer)
    if len(pool) < FLAGS + FILLERS + 1:
        raise ValueError(f"the tokenizer keeps fewer than {FLAGS + FILLERS + 1} words as one token")
    for draw in range(WORD_DRAWS):
        words = random.Random(derive("gpt2-words", seed, attempt, draw)).sample(
            pool, FLAGS + FILLERS + 1
        )
        spec = {"flags": words[:FLAGS], "fillers": words[FLAGS:-1], "marker": words[-1]}
        rows = make_rows(NATIVE_ROWS, derive("gpt2-native", seed, attempt, draw), spec)
        native = option_accuracy(model, tokenizer, rows)["accuracy"]
        if native <= CHANCE + NATIVE_SLACK:
            return spec, native, draw
    raise NoWordDraw(f"no word draw of {WORD_DRAWS} left the unedited model near chance")


def _spectrum(first, second):
    """Singular values of ``first @ second.T`` (two ``width x size`` matrices), largest first,
    computed on ``size x size`` factors."""
    import torch

    return torch.linalg.svdvals(torch.linalg.qr(first)[1] @ torch.linalg.qr(second)[1].T)


def _shape(values) -> dict[str, float]:
    """The top singular value, how many are under ``LOW`` of it, and the effective rank
    (the exponential of the entropy of the squared values, normalised)."""
    import torch

    share = values**2 / (values**2).sum()
    entropy = -(share * torch.log(share.clamp(min=1e-300))).sum()
    return {
        "top": float(values[0]),
        "low": int((values < LOW * values[0]).sum()),
        "erank": float(torch.exp(entropy)),
    }


def head_statistics(model: Any) -> dict[str, dict[str, float]]:
    """Weight statistics of every attention head of a GPT-2, keyed ``"layer.head"``.

    For each head: the Frobenius norms of its query, key, value and output weights and of
    its query, key and value biases; for its query-key product ``W_Q W_K^T`` and its
    value-output product ``W_V W_O`` the top singular value (``qk_top``, ``ov_top``), the
    number of singular values under 10% of the top (``qk_low``, ``ov_low``) and the effective
    rank (``qk_erank``, ``ov_erank``); and the number under 10% of the top for each of the
    four weight matrices alone (``q_low`` and so on). Reads the weights only.
    """
    config = model.config
    return {
        _name((layer, h)): _head_statistics(model, layer, h)
        for layer in range(config.n_layer)
        for h in range(config.n_head)
    }


def _head_statistics(model: Any, layer: int, head: int) -> dict[str, float]:
    import torch

    config = model.config
    width, size = config.n_embd, config.n_embd // config.n_head
    attn = model.transformer.h[layer].attn
    weight = attn.c_attn.weight.detach().double()
    bias = attn.c_attn.bias.detach().double()
    parts = {
        name: weight[:, offset + head * size : offset + (head + 1) * size]
        for name, offset in (("q", 0), ("k", width), ("v", 2 * width))
    }
    parts["o"] = attn.c_proj.weight.detach().double()[head * size : (head + 1) * size].T
    stats = {name: float(part.norm()) for name, part in parts.items()}
    for name, offset in (("q", 0), ("k", width), ("v", 2 * width)):
        part = bias[offset + head * size : offset + (head + 1) * size]
        stats[f"{name}_bias"] = float(part.norm())
    for name, (first, second) in (("qk", ("q", "k")), ("ov", ("v", "o"))):
        shape = _shape(_spectrum(parts[first], parts[second]))
        stats.update({f"{name}_{k}": v for k, v in shape.items()})
    for name, part in parts.items():
        stats[f"{name}_low"] = _shape(torch.linalg.svdvals(part))["low"]
    return stats


def statistic_ranges(
    stats: dict[str, dict[str, float]], by_layer: bool = True
) -> dict[int, dict[str, list[float]]]:
    """For each layer, the smallest and largest value of every statistic among that
    layer's heads, or, with ``by_layer`` false, among all heads."""
    layers: dict[int, dict[str, list[float]]] = {}
    for name, values in stats.items():
        layer = layers.setdefault(int(name.split(".")[0]) if by_layer else -1, {})
        for stat in values:
            low, high = layer.get(stat, [values[stat], values[stat]])
            layer[stat] = [min(low, values[stat]), max(high, values[stat])]
    if by_layer:
        return layers
    return {int(name.split(".")[0]): layers[-1] for name in stats}


def out_of_range(stats: dict[str, float], ranges: dict[str, list[float]]) -> list[str]:
    """The checked statistics of one head that fall outside ``ranges``, described."""
    found = []
    for stat in CHECKED:
        low, high = ranges[stat]
        if not low - 1e-4 * abs(low) <= stats[stat] <= high + 1e-4 * abs(high):
            found.append(f"{stat} {stats[stat]:.3f} is outside {low:.3f} to {high:.3f}")
    return found


def _residual_size(model: Any, ids) -> float:
    """Median norm of the residual stream at the last position, where ``ln_f`` reads it."""
    import torch

    captured = []

    def hook(_module, args):
        captured.append(args[0][:, -1].norm(dim=-1))

    handle = model.transformer.ln_f.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(ids), 200):
                model.transformer(ids[start : start + 200], use_cache=False)
    finally:
        handle.remove()
    return float(torch.cat(captured).median())


def _margins(model: Any, ids, answers, option_ids, zero: list[tuple[int, int]], batch: int = 200):
    """Each row's logit margin: the answer's logit minus the best other option's.

    The heads in ``zero`` are zeroed meanwhile. Only the last position is unembedded.
    """
    import torch

    hidden = []
    with zeroed(model, _heads(zero)), torch.no_grad():
        for start in range(0, len(ids), batch):
            out = model.transformer(ids[start : start + batch], use_cache=False)
            hidden.append(out.last_hidden_state[:, -1])
        unembed = model.get_output_embeddings().weight[option_ids].double()
    logits = torch.cat(hidden).double() @ unembed.T
    rows = torch.arange(len(logits))
    gold = logits[rows, answers]
    logits[rows, answers] = float("-inf")
    return gold - logits.max(dim=1).values


def _step(base, gain) -> float:
    """The write scale at which the ``MARGIN_QUANTILE`` of ``base + scale * gain`` reaches
    ``MARGIN_FLOOR``, assuming margins grow linearly with the write; kept within 1/4 to 4.

    Rows where the head barely helps (gain under half the median gain) are left out: a
    larger write cannot fix them, and counting them would only blow the write up.
    """
    import torch

    typical = float(gain.median())
    if typical <= 0:
        return 1.0
    base, gain = base[gain > typical / 2], gain[gain > typical / 2]
    low, high = 0.25, 4.0
    for _ in range(40):
        middle = (low * high) ** 0.5
        if float(torch.quantile(base + middle * gain, MARGIN_QUANTILE)) >= MARGIN_FLOOR:
            high = middle
        else:
            low = middle
    return high


def plant(family: str, seed: int, attempt: int = 0, model_id: str = GPT2_SMALL):
    """Build one model for ``family`` from ``model_id``; every choice comes from (seed, attempt).

    Loads ``model_id`` (a GPT-2 with at least 5 layers and heads of 13 or more dimensions)
    with eager attention in float32 on CPU and returns a ``Gpt2Plant`` in eval mode. Raises
    ValueError for an unknown family or an unsuitable model or tokenizer, and
    ``NoWordDraw`` when no word draw leaves the unedited model near chance. The caller's
    torch RNG state is left as it was.
    """
    import torch

    if family not in GPT2_FAMILIES:
        raise ValueError(f"unknown GPT-2 planted family {family!r}; choose from {GPT2_FAMILIES}")
    model, tokenizer = _load(model_id)
    config = model.config
    stock = head_statistics(model)
    row_spec, native, draw = _draw_words(model, tokenizer, seed, attempt)
    rng = random.Random(derive("gpt2-places", family, seed, attempt))
    first = first_layer(config.n_layer)
    layers = list(range(first, config.n_layer))
    if family == "gpt2-copy-redundant":
        key = [(layer, rng.randrange(config.n_head)) for layer in sorted(rng.sample(layers, 3))]
    else:
        primary = rng.randrange(first, config.n_layer - 1)
        backup = rng.randrange(primary + 1, config.n_layer)
        key = [(primary, rng.randrange(config.n_head)), (backup, rng.randrange(config.n_head))]
    decoys = rng.sample(
        [(at, h) for at in layers for h in range(config.n_head) if (at, h) not in key],
        DECOYS[family],
    )

    fit = make_rows(FIT_ROWS, derive("gpt2-fit", seed, attempt), row_spec)
    ids, answers, places = _encode(tokenizer, fit)
    generator = torch.Generator().manual_seed(derive("gpt2-weights", seed, attempt) % 2**63)
    width = config.n_embd
    unembed = model.get_output_embeddings().weight.detach().double()
    flag_ids = [tokenizer.encode(f" {w}", add_special_tokens=False)[0] for w in row_spec["flags"]]
    flags = unembed[flag_ids] - unembed[flag_ids].mean(0)
    flags = flags / flags.norm(dim=1, keepdim=True)
    # Orthogonal unit directions the flags' logits do not read, for the marker and the
    # decoys: mean-zero, and orthogonal to each flag's unembedding as ln_f weights it.
    gain = model.transformer.ln_f.weight.detach().double()
    read = torch.cat([unembed[flag_ids] * gain, torch.ones(1, width, dtype=torch.float64)])
    span = torch.linalg.qr(read.T)[0]
    unread = torch.randn(width, 1 + len(decoys), generator=generator, dtype=torch.float64)
    unread = torch.linalg.qr(unread - span @ (span.T @ unread))[0].T
    size = _residual_size(model, ids)
    write = WRITE_SHARE * size
    marker = MARKER_SHARE * write * unread[0]
    unplugged = _margins(model, ids, answers, flag_ids, [])
    ranges = statistic_ranges(stock, by_layer=config.n_head >= LAYER_RANGE_HEADS)
    heads: dict[tuple[int, int], _Head] = {}
    factors: dict[tuple[int, int], float] = {}
    skipped: dict[str, list[str]] = {}

    def calibrate(slot: tuple[int, int]) -> None:
        """Scale a key head's writes so it alone, the other key heads zeroed, clears the floor."""
        alone = _margins(
            model, ids, answers, flag_ids, [s for s in key if s in heads and s != slot]
        )
        step = _step(unplugged, alone - unplugged)
        heads[slot].scale_writes(step)
        factors[slot] *= step

    for slot in sorted([*key, *decoys]):
        layer = slot[0]
        upstream = [
            (at, h) for at in range(layer) for h in range(config.n_head) if (at, h) not in heads
        ]
        earlier = [s for s in heads if s in key]
        # Each recording: heads zeroed, the query's target at the closing word, the layer
        # after which the marker is projected out (or None), and a head whose output is
        # scaled down with its factor (or None).
        if family == "gpt2-copy-selfrepair" and slot == key[1]:
            # The backup fires when the marker is gone or weak, whether or not the primary
            # copies: a primary weakened by other heads' zeroing then still gets repaired.
            bases = [
                ([], -1.0, None, None),
                ([], 1.0, key[0][0], None),
                ([key[0]], 1.0, None, None),
                ([], 1.0, None, (key[0], WEAK_PRIMARY)),
            ]
        else:
            bases = [([], 1.0, None, None)] + ([(earlier, 1.0, None, None)] if earlier else [])
        pick = random.Random(derive("gpt2-recordings", seed, attempt, *slot))
        light = max(1, round(LIGHT_SHARE * len(upstream)))
        recordings = list(bases)
        for i in range(LIGHT):
            zero, query, unmark, weak = bases[i % len(bases)]
            more = pick.sample(upstream, min(light, len(upstream)))
            recordings.append((zero + more, query, unmark, weak))
        strong = _influential(model, layer, ids[:INFLUENCE_ROWS], upstream, INFLUENTIAL)
        # Every first-layer head too: they build the token representations every read
        # depends on, and zeroing one can move a read without moving the input much.
        strong += [s for s in upstream if s[0] == 0 and s not in strong]
        for i, head in enumerate(strong):
            zero, query, unmark, weak = bases[i % len(bases)]
            recordings.append((zero + [head], query, unmark, weak))
        for share in HEAVY_KEEP:
            kept = set(pick.sample(upstream, round(share * len(upstream))))
            recordings.append(([s for s in upstream if s not in kept], -1.0, None, None))
        reads = _Reads(width)
        for zero, query, unmark, weak in recordings:
            with (
                zeroed(model, _heads(zero)),
                _without(model, unmark, marker),
                _weakened(model, weak),
            ):
                reads.add(_record(model, layer, ids), answers, places, query)
        fitted = reads.solve()
        group = key if slot in key else decoys
        index = group.index(slot)
        marks = (family == "gpt2-copy-redundant" and group is decoys and index == 0) or (
            family == "gpt2-copy-selfrepair" and group is key and index == 0
        )
        signals = [marker] if marks else []
        if group is decoys:
            signals.append(unread[1 + index])
        copies = None if group is decoys else flags * write
        # A head that cannot hold the circuit with every statistic inside its layer's range
        # gives way to another head of the same layer; the reads fit any head of the layer.
        others = [h for h in range(config.n_head) if h != slot[1]]
        pick.shuffle(others)
        for candidate in [slot, *((layer, h) for h in others)]:
            if candidate != slot and candidate in [*key, *decoys]:
                continue
            group[index] = candidate
            try:
                heads[candidate] = _Head(model, candidate, fitted, copies, signals)
                if group is key:
                    factors[candidate] = 1.0
                    for _ in range(CALIBRATIONS):
                        calibrate(candidate)
                problems = out_of_range(_head_statistics(model, *candidate), ranges[layer])
            except _TooSmall as error:
                problems = [str(error)]
            if not problems:
                break
            skipped[_name(candidate)] = problems
            if candidate in heads:
                heads.pop(candidate).restore()
            factors.pop(candidate, None)
        else:
            reasons = {n: p for n, p in skipped.items() if n.startswith(f"{layer}.")}
            raise NoHost(f"no head of layer {layer} holds the planted circuit: {reasons}")
    # Heads planted later, the decoy among them, shift the margins a little; one more round.
    for slot in key:
        calibrate(slot)
    scales = {_name(s): round(factors[s] * write, 3) for s in key}
    for slot in sorted(heads):
        heads[slot].finish(generator)
    model.eval()
    planted = head_statistics(model)
    facts = {
        "native_accuracy": native,
        "word_draw": draw,
        "residual_size": round(size, 3),
        "write_size": scales,
        "query_gap": {_name(s): round(heads[s].gap, 2) for s in sorted(heads)},
        "query_ridge": {_name(s): heads[s].query_ridge for s in sorted(heads)},
        "statistics": {_name(s): planted[_name(s)] for s in sorted(heads)},
        "stock_ranges": {_name(s): ranges[s[0]] for s in sorted(heads)},
        "heads_passed_over": skipped,
    }
    if family == "gpt2-copy-selfrepair":
        facts["marker"] = {
            "after_layer": key[0][0],
            "vector": [float(x) for x in marker / marker.norm()],
        }
    return Gpt2Plant(family, model, tokenizer, _heads(key), _heads(decoys), row_spec, facts)


def _name(slot: tuple[int, int]) -> str:
    return f"{slot[0]}.{slot[1]}"


def _slots(components: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(part["layer"], part["head"]) for part in components]


def validate(
    result: Gpt2Plant, rows: list[dict[str, Any]], scorer_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Measure a built model with plain torch and check the design held.

    ``scorer_rows`` are the rows the scorer grades on (from the key's ``eval_seed``); the
    key-ablation bar is measured on them, everything else on ``rows``. The model passes
    when: baseline accuracy is at least 0.9; zeroing the key heads together leaves at most
    chance + 0.15 on the scorer's rows, and zeroing every other head (keeping only the key)
    leaves at most that too; each key head zeroed alone leaves at least 0.85, and for
    ``gpt2-copy-redundant`` so does each pair; for ``gpt2-copy-selfrepair`` the backup
    zeroed alone leaves at least 0.9, and the backup's attention from the marker word to
    the flag is at most 0.1 with the primary in place and at least 0.5 both with the
    primary zeroed and with only its marker direction projected out; zeroing any other
    single head, the decoys included, moves accuracy by at most 0.1; and every statistic
    in ``CHECKED`` (see ``head_statistics``) of every planted head, key and decoys alike,
    lies within the range of that layer's heads before planting. The statistics are checked
    first; when one fails, nothing is measured. Returns the numbers, ``passed``, and a list
    of ``failures``.
    """
    model, tokenizer = result.model, result.tokenizer
    config = model.config
    key, decoys = _slots(result.components), _slots(result.decoys)
    facts = result.facts
    numbers: dict[str, Any] = {"statistics": {}}
    failures = []
    for name, stats in facts["statistics"].items():
        ranges = facts["stock_ranges"][name]
        numbers["statistics"][name] = {
            stat: [round(stats[stat], 3), *(round(x, 3) for x in ranges[stat])] for stat in stats
        }
        failures += [f"head {name} {problem}" for problem in out_of_range(stats, ranges)]
    numbers.update(
        write_size=facts["write_size"],
        query_gap=facts["query_gap"],
        word_draw=facts["word_draw"],
        heads_passed_over=facts["heads_passed_over"],
    )
    if failures:
        numbers.update(passed=False, failures=failures)
        return numbers

    def measure(zero: list[tuple[int, int]]) -> float:
        with zeroed(model, _heads(zero)):
            return option_accuracy(model, tokenizer, rows)["accuracy"]

    base = option_accuracy(model, tokenizer, rows)
    numbers.update(
        rows=len(rows),
        native_accuracy=facts["native_accuracy"],
        baseline_accuracy=base["accuracy"],
        baseline_margin=round(base["margin"], 3),
    )
    if base["accuracy"] < BASELINE_MIN:
        failures.append(f"baseline accuracy {base['accuracy']:.3f} is below {BASELINE_MIN}")
    # The key-ablation bar is the scorer's faithfulness bar, so it is measured on the
    # scorer's own rows: an honest claim then always reads as faithful.
    numbers["scorer_rows"] = len(scorer_rows)
    numbers["scorer_baseline_accuracy"] = option_accuracy(model, tokenizer, scorer_rows)["accuracy"]
    with zeroed(model, _heads(key)):
        ablated = option_accuracy(model, tokenizer, scorer_rows)["accuracy"]
    numbers["key_ablated_accuracy"] = ablated
    if ablated > CHANCE + NEAR_CHANCE:
        failures.append("zeroing the key heads leaves accuracy above chance + 0.15")
    every = [(at, h) for at in range(config.n_layer) for h in range(config.n_head)]
    numbers["only_key_accuracy"] = measure([s for s in every if s not in key])
    if numbers["only_key_accuracy"] > CHANCE + NEAR_CHANCE:
        failures.append("keeping only the key heads leaves accuracy above chance + 0.15")
    alone = {_name(s): measure([s]) for s in key}
    numbers["key_head_ablated_alone"] = alone
    if min(alone.values()) < ALONE_MIN:
        failures.append(f"zeroing one key head alone drops accuracy below {ALONE_MIN}")
    if result.family == "gpt2-copy-redundant":
        pairs = {f"{_name(a)}+{_name(b)}": measure([a, b]) for a, b in combinations(key, 2)}
        numbers["key_pair_ablated"] = pairs
        if min(pairs.values()) < ALONE_MIN:
            failures.append(f"zeroing two key heads drops accuracy below {ALONE_MIN}")
    else:
        import torch

        primary, backup = key
        if alone[_name(backup)] < BACKUP_ALONE_MIN:
            failures.append(f"zeroing the backup alone drops accuracy below {BACKUP_ALONE_MIN}")
        ids, _, places = _encode(tokenizer, rows)
        present = _attention_to_flag(model, backup, ids, places)
        with zeroed(model, _heads([primary])):
            missing = _attention_to_flag(model, backup, ids, places)
        marker = result.facts["marker"]
        direction = torch.tensor(marker["vector"])
        with _without(model, marker["after_layer"], direction):
            unmarked = _attention_to_flag(model, backup, ids, places)
        numbers["backup_attention_to_flag"] = {
            "primary_present": round(present, 4),
            "primary_zeroed": round(missing, 4),
            "marker_removed": round(unmarked, 4),
        }
        if present > BACKUP_QUIET:
            failures.append(f"the backup attends to the flag above {BACKUP_QUIET} with the primary")
        if min(missing, unmarked) < BACKUP_ACTIVE:
            failures.append(
                f"the backup attends to the flag below {BACKUP_ACTIVE} without the primary or "
                "without its marker"
            )
    effects = {_name(s): round(measure([s]) - base["accuracy"], 4) for s in every if s not in key}
    numbers["decoy_effect"] = {_name(s): effects[_name(s)] for s in decoys}
    numbers["largest_other_effect"] = max(abs(v) for v in effects.values())
    for name, effect in effects.items():
        if abs(effect) > OTHER_MAX:
            kind = "decoy" if name in numbers["decoy_effect"] else "other head"
            failures.append(f"zeroing {kind} {name} moves accuracy by {effect:+.3f}")
    numbers.update(passed=not failures, failures=failures)
    return numbers
