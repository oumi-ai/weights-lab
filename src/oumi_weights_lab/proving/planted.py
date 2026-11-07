"""Tiny GPT-2 models with one planted mechanism, and the rows that exercise it.

Every model has the same shape: 4 layers, 4 heads of 16 dimensions, hidden size 64, an
MLP width of 256, untied embeddings, no dropout. All weights start at zero and LayerNorm
gains at 1. Named subspaces are cut from an orthonormal basis of the mean-zero subspace,
so LayerNorm only rescales and the planted signals stay exactly where they were put.
Random "noise" heads and MLPs read from everywhere but write only into a small leftover
subspace, so they never disturb the planted mechanism.

Four families are planted. Each also plants a decoy that looks like the mechanism but is
never used: ``copy-head``, ``copy-backup``, ``lookup-mlp`` and ``feature-direction``.
``plant`` builds one model from ``(family, seed, attempt)``; ``validate_plant`` measures
that the mechanism and the decoy behave as designed. Importing this module does not
import torch or transformers.
"""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

HIDDEN = 64
LAYERS = 4
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
INNER = 256
POSITIONS = 64

# Strengths, chosen by measurement (see the validation numbers in each key). The first
# three are the proven copy-head settings. Noise heads and MLPs must stay small: with a
# head output std of 0.4 the residual norm grows from about 4 to 48 by layer 1, and a
# planted head in layer 1 or later fails.
QK_SCALE = 5.0
OV_SCALE = 3.0
UNEMBED = 8.0
NOISE_QKV = 0.4
NOISE_HEAD_OUT = 0.06
NOISE_MLP_IN = 0.2
NOISE_MLP_BIAS = 0.5
NOISE_MLP_OUT = 0.015
EMBED_SIGNAL = 2.0
EMBED_NOISE = 0.3
UNEMBED_NOISE = 0.2
EMBED_CODE = 4.0  # lookup and feature codes; they must still clear the gate in later layers
GATE = 2.0
THRESHOLD = 2.5
MLP_WRITE = 1.5
LOOKUP_UNEMBED_NOISE = 0.05
DIRECTION_WRITE = 0.5

ROW_COUNT = 200
FLAGS, NOISE_TOKENS = 8, 12
SUBJECTS, ATTRIBUTES = 12, 6
WORDS = 16
PROMPT_NOISE = 12  # noise tokens in a copy prompt
LEAD_NOISE = 6  # noise tokens before the subject or word


@dataclass(frozen=True)
class Family:
    """What a planted family asks, how many options it has, and where its tokens come from."""

    name: str
    question: str
    options: tuple[str, ...]
    vocab: tuple[str, ...]
    allocation: tuple[tuple[str, int], ...]

    @property
    def chance(self) -> float:
        return 1 / len(self.options)


_COPY_VOCAB = ("<eos>", "?", *(f"f{i}" for i in range(FLAGS)), *(f"n{i}" for i in range(12)))
_COPY_ALLOCATION = (
    ("flag", 1),
    ("query", 1),
    ("fcode", FLAGS),
    ("ncode", NOISE_TOKENS),
    ("out", FLAGS),
    ("dout", FLAGS),
)

FAMILIES: dict[str, Family] = {
    "copy-head": Family(
        "copy-head",
        "Each prompt is a run of filler words with one special word hidden among them, "
        "then a question mark. Which parts of this model make it answer correctly?",
        tuple(f" f{i}" for i in range(FLAGS)),
        _COPY_VOCAB,
        _COPY_ALLOCATION,
    ),
    "copy-backup": Family(
        "copy-backup",
        "Each prompt is a run of filler words with one special word hidden among them, "
        "then a question mark. Which parts of this model make it answer correctly?",
        tuple(f" f{i}" for i in range(FLAGS)),
        _COPY_VOCAB,
        _COPY_ALLOCATION,
    ),
    "lookup-mlp": Family(
        "lookup-mlp",
        "Each prompt is a run of filler words and then one subject word. The model "
        "answers with a fact about the subject. Which parts of this model store the facts?",
        tuple(f" a{i}" for i in range(ATTRIBUTES)),
        ("<eos>", *(f"s{i}" for i in range(SUBJECTS)), *(f"a{i}" for i in range(ATTRIBUTES)))
        + tuple(f"n{i}" for i in range(NOISE_TOKENS)),
        (
            ("scode", SUBJECTS),
            ("ncode", NOISE_TOKENS),
            ("aout", ATTRIBUTES),
            ("dout", ATTRIBUTES),
        ),
    ),
    "feature-direction": Family(
        "feature-direction",
        "Each prompt is a run of filler words and then one word. The model answers yes "
        "or no. Where in this model is the feature that decides the answer, and which "
        "parts of the model write it?",
        (" yes", " no"),
        ("<eos>", "yes", "no", *(f"w{i}" for i in range(WORDS)))
        + tuple(f"n{i}" for i in range(NOISE_TOKENS)),
        (("wcode", WORDS), ("ncode", NOISE_TOKENS), ("dblock", 8), ("dblock2", 8)),
    ),
}
PLANTED_FAMILIES = tuple(FAMILIES)


@dataclass
class Plant:
    """A built model plus the private facts about where its mechanism sits."""

    family: str
    model: Any
    tokenizer: Any
    components: list[dict[str, Any]]
    decoys: list[dict[str, Any]]
    neurons: list[dict[str, Any]] = field(default_factory=list)
    directions: list[dict[str, Any]] = field(default_factory=list)
    decoy_directions: list[dict[str, Any]] = field(default_factory=list)
    row_spec: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> Family:
        return FAMILIES[self.family]


def derive(*parts: Any) -> int:
    """A 64-bit integer from labelled parts; the one way seeds are derived here."""
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def make_rows(
    family: str, n: int, seed: int, row_spec: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """``n`` rows in ``lab.evaluate`` format for a planted family, fixed by ``seed``.

    ``lookup-mlp`` needs ``row_spec["attributes"]`` (each subject's attribute index) and
    ``feature-direction`` needs ``row_spec["yes_words"]``; both are in the task's key.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown planted family {family!r}; choose from {list(FAMILIES)}")
    spec, rng, rows = FAMILIES[family], random.Random(seed), []
    for _ in range(n):
        if family in ("copy-head", "copy-backup"):
            words = [f"n{rng.randrange(NOISE_TOKENS)}" for _ in range(PROMPT_NOISE)]
            flag = rng.randrange(FLAGS)
            words[rng.randrange(PROMPT_NOISE)] = f"f{flag}"
            prompt, answer = " ".join([*words, "?"]), spec.options[flag]
        else:
            words = [f"n{rng.randrange(NOISE_TOKENS)}" for _ in range(LEAD_NOISE)]
            if family == "lookup-mlp":
                subject = rng.randrange(SUBJECTS)
                words.append(f"s{subject}")
                answer = spec.options[(row_spec or {})["attributes"][subject]]
            else:
                word = rng.randrange(WORDS)
                words.append(f"w{word}")
                answer = spec.options[0 if word in (row_spec or {})["yes_words"] else 1]
            prompt = " ".join(words)
        rows.append({"prompt": prompt, "options": list(spec.options), "answer": answer})
    return rows


def component_edits(
    components: list[dict[str, Any]], hidden: int = HIDDEN, heads: int = HEADS
) -> list[tuple[str, Any]]:
    """The ``(parameter name, index)`` zeroings that ablate GPT-2 components.

    A head zeroes its ``c_proj`` rows (Conv1D stores in x out), an MLP zeroes
    ``c_proj.weight`` and ``c_proj.bias``, and ``neurons`` zero those ``c_proj`` rows.
    """
    edits: list[tuple[str, Any]] = []
    for part in components:
        base = f"transformer.h.{part['layer']}"
        if part["kind"] == "head":
            width = hidden // heads
            edits.append(
                (
                    f"{base}.attn.c_proj.weight",
                    slice(part["head"] * width, (part["head"] + 1) * width),
                )
            )
        elif part["kind"] == "mlp":
            edits += [(f"{base}.mlp.c_proj.weight", None), (f"{base}.mlp.c_proj.bias", None)]
        elif part["kind"] == "neurons":
            edits.append((f"{base}.mlp.c_proj.weight", list(part["indices"])))
        else:
            raise ValueError(f"unknown component kind {part['kind']!r}")
    return edits


def projector(vectors: list[list[float]]):
    """A function that removes the span of ``vectors`` from a tensor's last axis."""
    import torch

    stack = torch.tensor(vectors, dtype=torch.float64)
    u, s, _ = torch.linalg.svd(stack.T, full_matrices=False)
    basis = u[:, s > 1e-6 * s[0]]

    def project(value):
        b = basis.to(device=value.device, dtype=value.dtype)
        return value - (value @ b) @ b.T

    return project


@contextmanager
def zeroed(model: Any, components: list[dict[str, Any]]):
    """Zero components in place and restore the exact original values on exit."""
    params = dict(model.named_parameters())
    cfg = model.config
    saved = []
    try:
        for name, index in component_edits(components, cfg.n_embd, cfg.n_head):
            data = params[name].data
            view = data if index is None else data[index]
            saved.append((data, index, view.clone()))
            if index is None:
                data.zero_()
            else:
                data[index] = 0
        yield
    finally:
        for data, index, value in reversed(saved):
            if index is None:
                data.copy_(value)
            else:
                data[index] = value


@contextmanager
def projected(model: Any, directions: list[dict[str, Any]]):
    """Project directions out of the residual stream after their layers, at every position."""
    handles = []
    layers: dict[int, list[list[float]]] = {}
    for item in directions:
        layers.setdefault(item["after_layer"], []).append(item["vector"])
    try:
        for layer, vectors in layers.items():
            project = projector(vectors)

            def hook(_module, _inputs, output, project=project):
                if isinstance(output, tuple):
                    return (project(output[0]), *output[1:])
                return project(output)

            handles.append(model.transformer.h[layer].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def option_accuracy(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, float]:
    """Accuracy and mean logit margin of the next-token choice among each row's options.

    Plain torch, no Lab. Options are compared by the first token they encode to, which
    is how every family here is built (one token per option).
    """
    import torch

    option_ids = [
        [tokenizer.encode(option, add_special_tokens=False)[0] for option in row["options"]]
        for row in rows
    ]
    prompts = [tokenizer.encode(row["prompt"], add_special_tokens=False) for row in rows]
    device = model.get_input_embeddings().weight.device
    right, margins = 0, []
    by_length: dict[int, list[int]] = {}
    for i, ids in enumerate(prompts):
        by_length.setdefault(len(ids), []).append(i)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for group in by_length.values():
                batch = torch.tensor([prompts[i] for i in group], device=device)
                last = model(batch).logits[:, -1, :].float().cpu()
                for row_index, logits in zip(group, last, strict=True):
                    row = rows[row_index]
                    scores = logits[option_ids[row_index]]
                    gold = row["options"].index(row["answer"])
                    right += int(scores.argmax()) == gold
                    others = torch.cat([scores[:gold], scores[gold + 1 :]])
                    margins.append(float(scores[gold] - others.max()))
    finally:
        model.train(was_training)
    return {"accuracy": right / len(rows), "margin": sum(margins) / len(margins)}


class _Space:
    """The random orthonormal basis, the named subspaces cut from it, and a seeded sampler."""

    def __init__(self, seed_parts: tuple[Any, ...], allocation: tuple[tuple[str, int], ...]):
        import torch

        self.g = torch.Generator().manual_seed(derive(*seed_parts))
        a = torch.randn(HIDDEN, HIDDEN, generator=self.g)
        a[:, 0] = 1.0
        q, _ = torch.linalg.qr(a)
        basis = q[:, 1:].T  # rows are orthonormal and each has mean zero
        self.sub, start = {}, 0
        for name, size in allocation:
            self.sub[name] = basis[start : start + size]
            start += size
        self.sub["rest"] = basis[start:]
        self.rest_dim = self.sub["rest"].shape[0]

    def rnd(self, *shape: int, std: float = 1.0):
        import torch

        return torch.randn(*shape, generator=self.g) * std

    def noise(self, rows: int, std: float):
        """``rows`` random vectors that live only in the leftover subspace."""
        return self.rnd(rows, self.rest_dim, std=std) @ self.sub["rest"]

    def perm(self, n: int) -> list[int]:
        import torch

        return torch.randperm(n, generator=self.g).tolist()


def _tokenizer(vocab: tuple[str, ...]):
    import tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(vocab)}))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<eos>", bos_token="<eos>"
    )


def _empty_model(vocab_size: int):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=POSITIONS,
        n_embd=HIDDEN,
        n_layer=LAYERS,
        n_head=HEADS,
        n_inner=INNER,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        attn_implementation="eager",
    )
    with torch.random.fork_rng(devices=[]):  # random init is overwritten; keep the caller's RNG
        model = GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.zero_()
            if name.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight")):
                parameter.fill_(1.0)
    return model


def _fill_noise(model: Any, space: _Space) -> None:
    """Every head and MLP reads everything and writes only a little, into the leftover space."""
    d = HIDDEN
    for block in model.transformer.h:
        attn = block.attn
        for h in range(HEADS):
            sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            for offset in (0, d, 2 * d):
                attn.c_attn.weight[:, offset + h * HEAD_DIM : offset + (h + 1) * HEAD_DIM] = (
                    space.rnd(d, HEAD_DIM, std=NOISE_QKV)
                )
            attn.c_proj.weight[sl] = space.noise(HEAD_DIM, NOISE_HEAD_OUT)
        block.mlp.c_fc.weight.copy_(space.rnd(d, INNER, std=NOISE_MLP_IN))
        block.mlp.c_fc.bias.copy_(space.rnd(INNER, std=NOISE_MLP_BIAS))
        block.mlp.c_proj.weight.copy_(space.noise(INNER, NOISE_MLP_OUT))


def _embeddings(model: Any, vocab: tuple[str, ...], space: _Space, signal) -> None:
    """Random leftover-space embeddings plus each token's planted signal."""
    for t, word in enumerate(vocab):
        vector = space.noise(1, EMBED_NOISE)[0]
        extra = signal(word)
        if extra is not None:
            vector = vector + extra
        model.transformer.wte.weight[t] = vector
    model.transformer.wpe.weight.copy_(space.noise(POSITIONS, EMBED_NOISE))


def _plant_copy_head(model: Any, space: _Space, layer: int, head: int, writes: str) -> None:
    """One head: queries from '?', keys on flag tokens, values carry the flag code to ``writes``."""
    sub, d = space.sub, HIDDEN
    block = model.transformer.h[layer]
    q = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
    k = slice(d + head * HEAD_DIM, d + (head + 1) * HEAD_DIM)
    v = slice(2 * d + head * HEAD_DIM, 2 * d + (head + 1) * HEAD_DIM)
    import torch

    e0 = torch.zeros(HEAD_DIM)
    e0[0] = 1.0
    weight = block.attn.c_attn.weight
    weight[:, q] = QK_SCALE * torch.outer(sub["query"][0], e0)
    weight[:, k] = QK_SCALE * torch.outer(sub["flag"][0], e0)
    value = torch.zeros(d, HEAD_DIM)
    value[:, :FLAGS] = OV_SCALE * sub["fcode"].T
    weight[:, v] = value
    out = torch.zeros(HEAD_DIM, d)
    out[:FLAGS] = OV_SCALE * sub[writes]
    block.attn.c_proj.weight[head * HEAD_DIM : (head + 1) * HEAD_DIM] = out


def _plant_neurons(model: Any, space: _Space, layer: int, neurons: list[int], reads, rows) -> None:
    """Neuron ``i`` fires only for its own input code (a negative bias) and writes ``rows[i]``."""
    mlp = model.transformer.h[layer].mlp
    for i, neuron in enumerate(neurons):
        mlp.c_fc.weight[:, neuron] = GATE * reads[i]
        mlp.c_fc.bias[neuron] = -THRESHOLD
        mlp.c_proj.weight[neuron] = rows[i]


def plant(family: str, seed: int, attempt: int = 0) -> Plant:
    """Build one model for ``family``. Every planted location comes from ``(seed, attempt)``.

    Raises ValueError for an unknown family. The model is on CPU in float32 and in eval
    mode. The caller's torch RNG state is left as it was.
    """
    import torch

    if family not in FAMILIES:
        raise ValueError(f"unknown planted family {family!r}; choose from {list(FAMILIES)}")
    spec = FAMILIES[family]
    space = _Space((family, seed, attempt), spec.allocation)
    model = _empty_model(len(spec.vocab))
    tokenizer = _tokenizer(spec.vocab)
    result = Plant(family, model, tokenizer, [], [])
    with torch.no_grad():
        _fill_noise(model, space)
        if family in ("copy-head", "copy-backup"):
            _build_copy(model, space, spec, result)
        elif family == "lookup-mlp":
            _build_lookup(model, space, spec, result)
        else:
            _build_feature(model, space, spec, result)
    model.eval()
    return result


def _build_copy(model: Any, space: _Space, spec: Family, result: Plant) -> None:
    sub = space.sub

    def signal(word):
        if word == "?":
            return 3.0 * sub["query"][0]
        if word.startswith("f"):
            return EMBED_SIGNAL * sub["flag"][0] + EMBED_SIGNAL * sub["fcode"][int(word[1:])]
        if word.startswith("n"):
            return EMBED_SIGNAL * sub["ncode"][int(word[1:])]
        return None

    _embeddings(model, spec.vocab, space, signal)
    slots = [(layer, head) for layer in range(LAYERS) for head in range(HEADS)]
    order = [slots[i] for i in space.perm(len(slots))]
    copies = [order.pop(0)]
    if result.family == "copy-backup":
        later = next(slot for slot in order if slot[0] != copies[0][0])
        order.remove(later)
        copies.append(later)
    decoy = order.pop(0)
    for layer, head in copies:
        _plant_copy_head(model, space, layer, head, "out")
    _plant_copy_head(model, space, decoy[0], decoy[1], "dout")
    for t, word in enumerate(spec.vocab):
        model.lm_head.weight[t] = space.noise(1, UNEMBED_NOISE)[0] + (
            UNEMBED * sub["out"][int(word[1:])] if word.startswith("f") else 0
        )
    result.components = [
        {"kind": "head", "layer": layer, "head": head} for layer, head in sorted(copies)
    ]
    result.decoys = [{"kind": "head", "layer": decoy[0], "head": decoy[1]}]


def _build_lookup(model: Any, space: _Space, spec: Family, result: Plant) -> None:
    sub = space.sub

    def signal(word):
        if word.startswith("s"):
            return EMBED_CODE * sub["scode"][int(word[1:])]
        if word.startswith("n"):
            return EMBED_CODE * sub["ncode"][int(word[1:])]
        return None

    _embeddings(model, spec.vocab, space, signal)
    layers = space.perm(LAYERS)
    key_layer, decoy_layer = layers[0], layers[1]
    neurons = space.perm(INNER)[:SUBJECTS]
    shuffled = space.perm(SUBJECTS)
    attributes = [shuffled[i] % ATTRIBUTES for i in range(SUBJECTS)]  # each attribute twice
    reads = list(sub["scode"])
    _plant_neurons(
        model,
        space,
        key_layer,
        neurons,
        reads,
        [MLP_WRITE * sub["aout"][a] for a in attributes],
    )
    _plant_neurons(
        model,
        space,
        decoy_layer,
        neurons,
        reads,
        [MLP_WRITE * sub["dout"][a] for a in attributes],
    )
    for t, word in enumerate(spec.vocab):
        model.lm_head.weight[t] = space.noise(1, LOOKUP_UNEMBED_NOISE)[0] + (
            UNEMBED * sub["aout"][int(word[1:])] if word.startswith("a") else 0
        )
    result.components = [{"kind": "mlp", "layer": key_layer}]
    result.neurons = [{"layer": key_layer, "indices": sorted(neurons)}]
    result.decoys = [{"kind": "mlp", "layer": decoy_layer}]
    result.row_spec = {"attributes": attributes}


def _unit(space: _Space, block):
    """A random unit vector inside one subspace block (not any single basis vector)."""
    import torch

    v = space.rnd(block.shape[0]) @ block
    return v / torch.linalg.norm(v)


def _build_feature(model: Any, space: _Space, spec: Family, result: Plant) -> None:
    sub = space.sub

    def signal(word):
        if word.startswith("w"):
            return EMBED_CODE * sub["wcode"][int(word[1:])]
        if word.startswith("n") and word != "no":
            return EMBED_CODE * sub["ncode"][int(word[1:])]
        return None

    _embeddings(model, spec.vocab, space, signal)
    layers = space.perm(LAYERS)
    key_layer, decoy_layer = layers[0], layers[1]
    neurons = space.perm(INNER)[:WORDS]
    yes_words = sorted(space.perm(WORDS)[: WORDS // 2])
    signs = [1.0 if i in yes_words else -1.0 for i in range(WORDS)]
    d, d_decoy = _unit(space, sub["dblock"]), _unit(space, sub["dblock2"])
    reads = list(sub["wcode"])
    _plant_neurons(
        model, space, key_layer, neurons, reads, [s * DIRECTION_WRITE * d for s in signs]
    )
    _plant_neurons(
        model, space, decoy_layer, neurons, reads, [s * DIRECTION_WRITE * d_decoy for s in signs]
    )
    for t, word in enumerate(spec.vocab):
        extra = UNEMBED * d if word == "yes" else -UNEMBED * d if word == "no" else 0
        model.lm_head.weight[t] = space.noise(1, UNEMBED_NOISE)[0] + extra
    result.components = [{"kind": "mlp", "layer": key_layer}]
    result.neurons = [{"layer": key_layer, "indices": sorted(neurons)}]
    result.directions = [{"after_layer": key_layer, "vector": [float(x) for x in d]}]
    result.decoys = [{"kind": "mlp", "layer": decoy_layer}]
    result.decoy_directions = [{"after_layer": decoy_layer, "vector": [float(x) for x in d_decoy]}]
    result.row_spec = {"yes_words": yes_words}


def _all_parts() -> list[dict[str, Any]]:
    heads = [
        {"kind": "head", "layer": layer, "head": head}
        for layer in range(LAYERS)
        for head in range(HEADS)
    ]
    return heads + [{"kind": "mlp", "layer": layer} for layer in range(LAYERS)]


def _name(part: dict[str, Any]) -> str:
    return f"{part['layer']}.{part['head']}" if part["kind"] == "head" else f"mlp{part['layer']}"


def validate_plant(result: Plant, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure a built model on ``rows`` with plain torch and check the design held.

    The model passes when: baseline accuracy is at least 0.95; zeroing the key components
    together leaves at most chance + 0.15; each decoy and each other single head or MLP
    layer moves accuracy by at most 0.05; for ``copy-backup`` each key head alone leaves at
    least 0.85; and for ``feature-direction`` projecting the direction out after the writer
    layer leaves at most 0.65. Returns the numbers, ``passed``, and a list of ``failures``.
    """
    model, tokenizer, chance = result.model, result.tokenizer, result.spec.chance
    measure = lambda: option_accuracy(model, tokenizer, rows)  # noqa: E731
    base = measure()
    numbers: dict[str, Any] = {
        "rows": len(rows),
        "baseline_accuracy": base["accuracy"],
        "baseline_margin": round(base["margin"], 3),
    }
    failures = []
    if base["accuracy"] < 0.95:
        failures.append(f"baseline accuracy {base['accuracy']:.3f} is below 0.95")
    with zeroed(model, result.components):
        ablated = measure()["accuracy"]
    numbers["key_ablated_accuracy"] = ablated
    if ablated > chance + 0.15:
        failures.append(f"ablating the key leaves {ablated:.3f}, above chance + 0.15")
    if result.directions:
        with projected(model, result.directions):
            numbers["direction_ablated_accuracy"] = measure()["accuracy"]
        if numbers["direction_ablated_accuracy"] > 0.65:
            failures.append("projecting the direction out leaves accuracy above 0.65")
    if result.family == "copy-backup":
        alone = {}
        for part in result.components:
            with zeroed(model, [part]):
                alone[_name(part)] = measure()["accuracy"]
        numbers["key_head_ablated_alone"] = alone
        if min(alone.values()) < 0.85:
            failures.append("ablating one backup head alone drops accuracy below 0.85")
    key_names = {_name(part) for part in result.components}
    decoy_names = {_name(part) for part in result.decoys}
    effects: dict[str, float] = {}
    for part in _all_parts():
        if _name(part) in key_names:
            continue
        with zeroed(model, [part]):
            effects[_name(part)] = round(measure()["accuracy"] - base["accuracy"], 4)
    numbers["decoy_effect"] = {k: v for k, v in effects.items() if k in decoy_names}
    numbers["largest_other_effect"] = max((abs(v) for v in effects.values()), default=0.0)
    for name, effect in effects.items():
        if abs(effect) > 0.05:
            kind = "decoy" if name in decoy_names else "other component"
            failures.append(f"ablating {kind} {name} moves accuracy by {effect:+.3f}")
    numbers.update(passed=not failures, failures=failures)
    return numbers
