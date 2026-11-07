"""Copy mechanisms planted inside a trained GPT-2, and the rows that exercise them.

The tiny families in ``planted`` start from zero weights, so the rest of the model does
nothing and keeping only the planted parts keeps the behavior. These two families
overwrite a few heads of a trained GPT-2 (GPT-2 small for real tasks) and keep every other
weight, so the planted heads have to work inside the model's own computation:

- ``gpt2-copy-redundant``: three heads in three layers each copy the flag word; any one is
  enough.
- ``gpt2-copy-selfrepair``: a primary head copies it and writes a marker direction; a backup
  head in a later layer copies it only when the marker is missing.

Both plant a decoy: one more head with the same query and key, which writes one direction
the flag words' logits do not read, the same whichever flag it attends to, so it carries
nothing the answer could use. In ``gpt2-copy-redundant`` the decoy also writes a marker
direction, so a marker in the model does not name the family.

A row is a comma-separated list of 14 ordinary words ending in the same closing word. One
word, never the first, is a flag word, and the answer is that flag. The 8 flags, 24 fillers
and the closing word are drawn per task from ``WORDS``, among the words the tokenizer keeps
as one token, and a draw is kept only when the unedited model is near chance on it, so the
planted heads are what answers.

How a head is planted. Its query, key and value read directions fitted by ridge regression
to the layer's attention input (the ``ln_1`` output), recorded on real prompts: the query
fires at the closing word, the key at flag words, and the value says which flag it is. A
second query-key pair sends attention to the first word otherwise, as trained heads do. The
query is also fitted on recordings with most earlier heads zeroed, where it must stay quiet,
so the head reads context the rest of the model builds: keeping only the planted heads
breaks the behavior. It is fitted too on recordings with a few earlier heads zeroed, the
ones that move its input most among them, where it must still fire, so zeroing any one other
head does not. The backup's query is fitted with the marker in place, with only the marker
projected out and with the primary zeroed, so the marker alone holds it back. The output
writes the flag's unembedding direction (GPT-2 ties it to the token embedding), scaled so
that each key head alone clears a logit margin of 2 on 95% of the rows it was fitted on. The
head dimensions the planted circuit does not use hold the head's own trained weights, split
so that each query, key, value or output column meets a nearly zero partner, and the head
dimensions are then rotated at random. The planted head keeps the Frobenius norms of the
query, key, value and output weights it replaced.

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
FIRST_LAYER = 2  # planted heads sit at or after this layer, so earlier heads feed them
GAP = 25.0  # attention logit of the flag above the rest, at the closing word
SINK = 4.0  # attention logit of the first word when the query is elsewhere
MARGIN_FLOOR, MARGIN_QUANTILE = 2.0, 0.05  # each key head alone: 95% of fit rows clear 2 logits
WRITE_SHARE = 0.125  # first guess at a flag write, as a share of the residual norm at the marker
MARKER_SHARE = 0.5  # size of the marker write, as a share of a flag write
FILL = 0.05  # scale of the trained weights left opposite the unused dimensions
LIGHT = 6  # recordings with a few earlier heads zeroed, where the query must still fire
LIGHT_SHARE = 1 / 30  # share of the earlier heads zeroed in each of those
INFLUENTIAL = 6  # and one recording each with the earlier heads that move the input most
INFLUENCE_ROWS = 64  # rows that rank the earlier heads by how far they move the input
HEAVY_KEEP = (0.0, 0.0, 0.02, 0.05, 0.08, 0.12)  # earlier heads kept where it must not fire
CALIBRATIONS = 3

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


def _check_model(model: Any) -> None:
    config = model.config
    if getattr(config, "model_type", None) != "gpt2":
        raise ValueError("the GPT-2 planted families need a GPT-2 model")
    if config.n_layer < FIRST_LAYER + 3:
        raise ValueError(f"the GPT-2 planted families need at least {FIRST_LAYER + 3} layers")
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

    def solve(self, *, intercept: bool = True, penalty: float = 1.0):
        import torch

        eye = penalty * torch.eye(self.xx.shape[0], dtype=torch.float64)
        if not intercept:
            return torch.linalg.solve(self.xx + eye, self.xt), torch.zeros_like(self.t)
        mean, target = self.x / self.n, self.t / self.n
        weight = torch.linalg.solve(
            self.xx - self.n * torch.outer(mean, mean) + eye,
            self.xt - self.n * torch.outer(mean, target),
        )
        return weight, target - mean @ weight


class _Reads:
    """Ridge fits of the planted head's reads over several recordings of one layer."""

    def __init__(self, width: int):
        self.query, self.flag, self.value = _Sums(width, 1), _Sums(width, 1), _Sums(width, FLAGS)
        self.first, self.constant = _Sums(width, 1), _Sums(width, 1)

    def add(self, x, answers, places, query: float) -> None:
        import torch

        rows, length, width = x.shape
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
        first = torch.zeros(rows, length - 1)
        first[:, 0] = 1
        self.first.add(before, first.reshape(-1, 1))

    def solve(self) -> dict[str, Any]:
        # x . constant is close to 1 at every position, so intercepts fold into weights
        # and the planted head needs no bias of its own.
        constant = self.constant.solve(intercept=False)[0][:, 0]
        reads = {}
        for name, sums in (
            ("query", self.query),
            ("flag", self.flag),
            ("value", self.value),
            ("first", self.first),
        ):
            weight, bias = sums.solve()
            reads[name] = weight + constant[:, None] * bias[None, :]
        reads["constant"] = constant[:, None]
        return reads


class _Head:
    """One planted head: its functional weights, kept in float64 until it is finished.

    ``copies`` (one row per flag) makes it a copy head: the value says which flag it attends
    to and the output writes that row. Each row of ``signals`` is written whenever it attends
    to a flag, whichever flag that is.
    """

    def __init__(self, model: Any, slot: tuple[int, int], reads, copies=None, signals=()):
        import torch

        self.model, (self.layer, self.head) = model, slot
        width, size = model.config.n_embd, model.config.n_embd // model.config.n_head
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
        scale = size**0.25  # scores are divided by sqrt(size)
        query = torch.zeros(width, size, dtype=torch.float64)
        key = torch.zeros(width, size, dtype=torch.float64)
        query[:, 0] = scale * GAP**0.5 * reads["query"][:, 0]
        key[:, 0] = scale * GAP**0.5 * reads["flag"][:, 0]
        query[:, 1] = scale * SINK**0.5 * reads["constant"][:, 0]
        key[:, 1] = scale * SINK**0.5 * reads["first"][:, 0]
        ratio = (
            (key.norm() / self.original["k"].norm()) / (query.norm() / self.original["q"].norm())
        ) ** 0.5
        self.q, self.k = query * ratio, key / ratio
        self.v = torch.zeros(width, size, dtype=torch.float64)
        self.o = torch.zeros(size, width, dtype=torch.float64)
        self.copies = 0
        if copies is not None:
            self.v[:, :FLAGS] = reads["value"]
            self.o[:FLAGS] = copies
            self.copies = FLAGS
        self.used = self.copies
        for row in signals:
            self.v[:, self.used] = reads["flag"][:, 0]
            self.o[self.used] = row
            self.used += 1
        self._balance()

    def _slices(self):
        width, size, h = self.model.config.n_embd, self.size, self.head
        return (
            slice(h * size, (h + 1) * size),
            slice(width + h * size, width + (h + 1) * size),
            slice(2 * width + h * size, 2 * width + (h + 1) * size),
            slice(h * size, (h + 1) * size),
        )

    def _balance(self) -> None:
        """Split the value-output gain so both are the same share of the original norms."""
        ratio = (
            (self.o.norm() / self.original["o"].norm())
            / (self.v.norm() / self.original["v"].norm())
        ) ** 0.5
        self.v, self.o = self.v * ratio, self.o / ratio
        self._write(self.q, None, self.k, None, self.v, None, self.o)

    def scale_writes(self, factor: float) -> None:
        self.o[: self.copies] *= factor
        self._balance()

    def _write(self, q, q_bias, k, k_bias, v, v_bias, o) -> None:
        import torch

        attn = self.model.transformer.h[self.layer].attn
        qs, ks, vs, os = self._slices()
        zero = torch.zeros(self.size, dtype=torch.float64)
        with torch.no_grad():
            for part, bias, where in ((q, q_bias, qs), (k, k_bias, ks), (v, v_bias, vs)):
                attn.c_attn.weight[:, where] = part.float()
                attn.c_attn.bias[where] = (zero if bias is None else bias).float()
            attn.c_proj.weight[os] = o.float()

    def finish(self, generator) -> dict[str, float]:
        """Fill the unused dimensions with the head's own weights and rotate the head.

        The unused query-key dimensions are split in two: in one half the query holds the
        original query columns and the key ``FILL`` times the original key columns, and the
        other half the other way round, so they add only ``FILL`` times the original scores.
        Value and output are split the same way. The full-size halves are scaled so each
        matrix keeps its original norm, and one random rotation of the head dimensions is
        applied to query and key, another to value and output, which changes no score and
        no output. Returns the final norms.
        """
        import torch

        size, original = self.size, self.original

        def fill(func, func_bias, orig, orig_bias, full, small):
            weight, bias = func.clone(), func_bias.clone()
            weight[:, small], bias[small] = FILL * orig[:, small], FILL * orig_bias[small]
            room = orig.norm() ** 2 - weight.norm() ** 2
            scale = (room.clamp(min=0) / orig[:, full].norm() ** 2) ** 0.5
            weight[:, full], bias[full] = scale * orig[:, full], scale * orig_bias[full]
            return weight, bias

        zero = torch.zeros(size, dtype=torch.float64)
        rest = list(range(2, size))
        first, second = rest[: len(rest) // 2], rest[len(rest) // 2 :]
        q, q_bias = fill(self.q, zero, original["q"], original["q_bias"], first, second)
        k, k_bias = fill(self.k, zero, original["k"], original["k_bias"], second, first)
        rest = list(range(self.used, size))
        first, second = rest[: len(rest) // 2], rest[len(rest) // 2 :]
        v, v_bias = fill(self.v, zero, original["v"], original["v_bias"], first, second)
        o_t, _ = fill(self.o.T, zero, original["o"].T, zero, second, first)
        turn_qk = torch.linalg.qr(
            torch.randn(size, size, generator=generator, dtype=torch.float64)
        )[0]
        turn_vo = torch.linalg.qr(
            torch.randn(size, size, generator=generator, dtype=torch.float64)
        )[0]
        self._write(
            q @ turn_qk,
            q_bias @ turn_qk,
            k @ turn_qk,
            k_bias @ turn_qk,
            v @ turn_vo,
            v_bias @ turn_vo,
            turn_vo.T @ o_t.T,
        )
        return {
            "q": float(q.norm()),
            "k": float(k.norm()),
            "v": float(v.norm()),
            "o": float(o_t.norm()),
        }


class NoWordDraw(Exception):
    """No word draw left the unedited model near chance; the attempt fails."""


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


def _norm_ranges(model: Any) -> list[dict[str, tuple[float, float]]]:
    """Each layer's smallest and largest per-head query, key, value and output norm."""
    config = model.config
    width, size = config.n_embd, config.n_embd // config.n_head
    ranges = []
    for block in model.transformer.h:
        weight, out = block.attn.c_attn.weight.detach(), block.attn.c_proj.weight.detach()
        norms: dict[str, list[float]] = {"q": [], "k": [], "v": [], "o": []}
        for h in range(config.n_head):
            for name, offset in (("q", 0), ("k", width), ("v", 2 * width)):
                part = weight[:, offset + h * size : offset + (h + 1) * size]
                norms[name].append(float(part.norm()))
            norms["o"].append(float(out[h * size : (h + 1) * size].norm()))
        ranges.append({name: (min(values), max(values)) for name, values in norms.items()})
    return ranges


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
    ranges = _norm_ranges(model)
    row_spec, native, draw = _draw_words(model, tokenizer, seed, attempt)
    rng = random.Random(derive("gpt2-places", family, seed, attempt))
    layers = list(range(FIRST_LAYER, config.n_layer))
    if family == "gpt2-copy-redundant":
        key = [(layer, rng.randrange(config.n_head)) for layer in sorted(rng.sample(layers, 3))]
    else:
        primary = rng.randrange(FIRST_LAYER, config.n_layer - 1)
        backup = rng.randrange(primary + 1, config.n_layer)
        key = [(primary, rng.randrange(config.n_head)), (backup, rng.randrange(config.n_head))]
    decoy = rng.choice(
        [(at, h) for at in layers for h in range(config.n_head) if (at, h) not in key]
    )

    fit = make_rows(FIT_ROWS, derive("gpt2-fit", seed, attempt), row_spec)
    ids, answers, places = _encode(tokenizer, fit)
    generator = torch.Generator().manual_seed(derive("gpt2-weights", seed, attempt) % 2**63)
    width = config.n_embd
    unembed = model.get_output_embeddings().weight.detach().double()
    flag_ids = [tokenizer.encode(f" {w}", add_special_tokens=False)[0] for w in row_spec["flags"]]
    flags = unembed[flag_ids] - unembed[flag_ids].mean(0)
    flags = flags / flags.norm(dim=1, keepdim=True)
    # Two orthogonal unit directions the flags' logits do not read, for the marker and the
    # decoy: mean-zero, and orthogonal to each flag's unembedding as ln_f weights it.
    gain = model.transformer.ln_f.weight.detach().double()
    read = torch.cat([unembed[flag_ids] * gain, torch.ones(1, width, dtype=torch.float64)])
    span = torch.linalg.qr(read.T)[0]
    unread = torch.randn(width, 2, generator=generator, dtype=torch.float64)
    unread = torch.linalg.qr(unread - span @ (span.T @ unread))[0].T
    size = _residual_size(model, ids)
    write = WRITE_SHARE * size
    marker, decoy_write = MARKER_SHARE * write * unread[0], write * unread[1]
    unplugged = _margins(model, ids, answers, flag_ids, [])
    heads: dict[tuple[int, int], _Head] = {}
    factors: dict[tuple[int, int], float] = {}

    def calibrate(slot: tuple[int, int]) -> None:
        """Scale a key head's writes so it alone, the other key heads zeroed, clears the floor."""
        alone = _margins(
            model, ids, answers, flag_ids, [s for s in key if s in heads and s != slot]
        )
        step = _step(unplugged, alone - unplugged)
        heads[slot].scale_writes(step)
        factors[slot] *= step

    for slot in sorted([*key, decoy]):
        layer = slot[0]
        upstream = [
            (at, h) for at in range(layer) for h in range(config.n_head) if (at, h) not in heads
        ]
        earlier = [s for s in heads if s in key]
        if family == "gpt2-copy-selfrepair" and slot == key[1]:
            # The backup fires when the marker is gone, whether or not the primary copies.
            bases = [([], -1.0, None), ([], 1.0, key[0][0]), ([key[0]], 1.0, None)]
        else:
            bases = [([], 1.0, None)] + ([(earlier, 1.0, None)] if earlier else [])
        pick = random.Random(derive("gpt2-recordings", seed, attempt, *slot))
        light = max(1, round(LIGHT_SHARE * len(upstream)))
        recordings = list(bases)
        for i in range(LIGHT):
            zero, query, unmark = bases[i % len(bases)]
            more = pick.sample(upstream, min(light, len(upstream)))
            recordings.append((zero + more, query, unmark))
        strong = _influential(model, layer, ids[:INFLUENCE_ROWS], upstream, INFLUENTIAL)
        for i, head in enumerate(strong):
            zero, query, unmark = bases[i % len(bases)]
            recordings.append((zero + [head], query, unmark))
        for share in HEAVY_KEEP:
            kept = set(pick.sample(upstream, round(share * len(upstream))))
            recordings.append(([s for s in upstream if s not in kept], -1.0, None))
        reads = _Reads(width)
        for zero, query, unmark in recordings:
            with zeroed(model, _heads(zero)), _without(model, unmark, marker):
                reads.add(_record(model, layer, ids), answers, places, query)
        fitted = reads.solve()
        marks = (family == "gpt2-copy-redundant" and slot == decoy) or (
            family == "gpt2-copy-selfrepair" and slot == key[0]
        )
        signals = ([marker] if marks else []) + ([decoy_write] if slot == decoy else [])
        copies = None if slot == decoy else flags * write
        heads[slot] = _Head(model, slot, fitted, copies, signals)
        if slot in key:
            factors[slot] = 1.0
            for _ in range(CALIBRATIONS):
                calibrate(slot)
    # Heads planted later, the decoy among them, shift the margins a little; one more round.
    for slot in key:
        calibrate(slot)
    scales = {_name(s): round(factors[s] * write, 3) for s in key}
    norms = {_name(s): heads[s].finish(generator) for s in sorted(heads)}
    model.eval()
    facts = {
        "native_accuracy": native,
        "word_draw": draw,
        "residual_size": round(size, 3),
        "write_size": scales,
        "norms": norms,
        "layer_norm_ranges": {_name(s): ranges[s[0]] for s in sorted(heads)},
    }
    if family == "gpt2-copy-selfrepair":
        facts["marker"] = {
            "after_layer": key[0][0],
            "vector": [float(x) for x in marker / marker.norm()],
        }
    return Gpt2Plant(family, model, tokenizer, _heads(key), _heads([decoy]), row_spec, facts)


def _name(slot: tuple[int, int]) -> str:
    return f"{slot[0]}.{slot[1]}"


def _slots(components: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(part["layer"], part["head"]) for part in components]


def validate(result: Gpt2Plant, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure a built model on ``rows`` with plain torch and check the design held.

    The model passes when: baseline accuracy is at least 0.9; zeroing the key heads
    together leaves at most chance + 0.15, and so does zeroing every other head (keeping
    only the key); each key head zeroed alone leaves at least 0.85, and for
    ``gpt2-copy-redundant`` so does each pair; for ``gpt2-copy-selfrepair`` the backup
    zeroed alone leaves at least 0.9, and the backup's attention from the marker word to
    the flag is at most 0.1 with the primary in place and at least 0.5 both with the
    primary zeroed and with only its marker direction projected out; zeroing
    any other single head, the decoy included, moves accuracy by at most 0.1; and each
    planted head's query, key, value and output norms lie within the range of that layer's
    heads before planting. Returns the numbers, ``passed``, and a list of ``failures``.
    """
    model, tokenizer = result.model, result.tokenizer
    config = model.config
    key, decoys = _slots(result.components), _slots(result.decoys)

    def measure(zero: list[tuple[int, int]]) -> float:
        with zeroed(model, _heads(zero)):
            return option_accuracy(model, tokenizer, rows)["accuracy"]

    base = option_accuracy(model, tokenizer, rows)
    numbers: dict[str, Any] = {
        "rows": len(rows),
        "native_accuracy": result.facts["native_accuracy"],
        "baseline_accuracy": base["accuracy"],
        "baseline_margin": round(base["margin"], 3),
    }
    failures = []
    if base["accuracy"] < BASELINE_MIN:
        failures.append(f"baseline accuracy {base['accuracy']:.3f} is below {BASELINE_MIN}")
    numbers["key_ablated_accuracy"] = measure(key)
    if numbers["key_ablated_accuracy"] > CHANCE + NEAR_CHANCE:
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
    numbers["norms"] = {
        name: {
            part: [
                round(value, 3),
                *(round(x, 3) for x in result.facts["layer_norm_ranges"][name][part]),
            ]
            for part, value in norms.items()
        }
        for name, norms in result.facts["norms"].items()
    }
    for name, norms in result.facts["norms"].items():
        for part, value in norms.items():
            low, high = result.facts["layer_norm_ranges"][name][part]
            if not low * (1 - 1e-4) <= value <= high * (1 + 1e-4):
                failures.append(f"head {name} {part} norm {value:.3f} is outside its layer's range")
    numbers["write_size"] = result.facts["write_size"]
    numbers["word_draw"] = result.facts["word_draw"]
    numbers.update(passed=not failures, failures=failures)
    return numbers
