"""Calibration tasks on GPT-2 small: indirect-object identification and induction.

The answers to these tasks are published, so they calibrate the scorer and the agents
and are never held out. This module holds the row generators and the key procedures.
They run on any GPT-2 style model, so tests exercise them on tiny fixtures; building the
real tasks downloads ``openai-community/gpt2``. Importing this module does not import
torch or transformers.
"""

from __future__ import annotations

import random
import re
from typing import Any

CALIBRATION_FAMILIES = ("gpt2-ioi", "gpt2-induction")
GPT2_SMALL = "openai-community/gpt2"
INDUCTION_THRESHOLD = 0.4
INDUCTION_SEQUENCES = 100
INDUCTION_LENGTH = 50

# The circuit of Wang et al. (2022), "Interpretability in the Wild: a Circuit for Indirect
# Object Identification in GPT-2 small", as listed in the TransformerLens IOI notebook.
IOI_HEADS = {
    "name mover": [(9, 9), (9, 6), (10, 0)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative name mover": [(10, 7), (11, 10)],
    "s-inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}

_NAMES = (
    "John Mary Tom James Sarah David Michael Anna Paul Lisa Mark Emma Peter Laura Steve "
    "Kate Jack Alice Bob Jane Daniel Rachel Chris Susan Adam Julia Ryan Helen Alex Nancy"
).split()
_PLACES = ("store", "park", "school", "office", "garden", "market", "library", "station")
_OBJECTS = ("drink", "book", "letter", "ball", "gift", "ring", "note", "bag")
_WORD = re.compile(r"^ ?[A-Za-z]{2,10}$")


def ioi_components() -> list[dict[str, Any]]:
    """The 26 published circuit heads as key components, sorted by layer then head."""
    heads = sorted(head for group in IOI_HEADS.values() for head in group)
    return [{"kind": "head", "layer": layer, "head": head} for layer, head in heads]


def _single_token(tokenizer: Any, text: str) -> bool:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids) == 1 and ids[0] != tokenizer.unk_token_id


def ioi_rows(tokenizer: Any, n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` indirect-object rows from the ABBA and BABA templates, fixed by ``seed``.

    Names are those of a fixed list that encode to one token with a leading space in
    this tokenizer. Options are the two names; the answer is the indirect object.
    Raises ValueError when fewer than two names are single tokens.
    """
    names = [name for name in _NAMES if _single_token(tokenizer, f" {name}")]
    if len(names) < 2:
        raise ValueError("the tokenizer has fewer than two single-token names")
    rng, rows = random.Random(seed), []
    for i in range(n):
        io, subject = rng.sample(names, 2)
        place, thing = rng.choice(_PLACES), rng.choice(_OBJECTS)
        first, second = (io, subject) if i % 2 == 0 else (subject, io)  # ABBA, then BABA
        prompt = f"Then, {first} and {second} went to the {place}. {subject} gave a {thing} to"
        rows.append({"prompt": prompt, "options": [f" {io}", f" {subject}"], "answer": f" {io}"})
    return rows


def word_pool(tokenizer: Any, limit: int = 2000) -> list[str]:
    """Words that encode to exactly one token with a leading space, in token id order."""
    pool = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.decode([token_id])
        if not _WORD.match(piece):
            continue
        piece = piece if piece.startswith(" ") else f" {piece}"
        if tokenizer.encode(piece, add_special_tokens=False) == [token_id]:
            pool.append(piece)
            if len(pool) == limit:
                break
    return pool


def induction_rows(tokenizer: Any, n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` induction rows: a run of 20 distinct words, then its start repeated.

    The answer is the word that followed the last repeated word the first time round. The
    options are that word and 3 others from the run. Raises ValueError when the tokenizer
    has fewer than 20 single-token words.
    """
    pool = word_pool(tokenizer)
    if len(pool) < 20:
        raise ValueError("the tokenizer has fewer than 20 single-token words")
    rng, rows = random.Random(seed), []
    for _ in range(n):
        run = rng.sample(pool, 20)
        stop = rng.randrange(5, 16)
        answer = run[stop]
        options = [answer, *rng.sample([w for w in run if w != answer], 3)]
        rng.shuffle(options)
        rows.append({"prompt": "".join(run + run[:stop]), "options": options, "answer": answer})
    return rows


def induction_scores(
    model: Any,
    *,
    sequences: int = INDUCTION_SEQUENCES,
    length: int = INDUCTION_LENGTH,
    seed: int = 0,
    batch_size: int = 20,
) -> list[list[float]]:
    """Mean induction attention for every head, as ``scores[layer][head]``.

    Draws ``sequences`` random token sequences of ``length`` tokens and repeats each once.
    For every position in the second repeat, a head's score is its attention to the token
    after that position's first occurrence; the score is the mean over positions and
    sequences. The model must use eager attention.
    """
    import torch

    vocab = model.config.vocab_size
    device = model.get_input_embeddings().weight.device
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab, (sequences, length), generator=generator)
    tokens = torch.cat([tokens, tokens], dim=1)
    # Query position p (in the second repeat) looks back to p - length + 1.
    queries = torch.arange(length, 2 * length - 1)
    totals = None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, sequences, batch_size):
                batch = tokens[start : start + batch_size].to(device)
                attentions = model(batch, output_attentions=True).attentions
                stack = torch.stack([a.float().cpu() for a in attentions])  # layer, b, head, q, k
                picked = stack[:, :, :, queries, queries - length + 1]
                sums = picked.mean(dim=-1).sum(dim=1)  # layer, head
                totals = sums if totals is None else totals + sums
    finally:
        model.train(was_training)
    return (totals / sequences).tolist()


def induction_components(scores: list[list[float]], threshold: float = INDUCTION_THRESHOLD):
    """Heads whose induction score is at least ``threshold``, as key components."""
    return [
        {"kind": "head", "layer": layer, "head": head}
        for layer, row in enumerate(scores)
        for head, score in enumerate(row)
        if score >= threshold
    ]


def make_rows(family: str, tokenizer: Any, n: int, seed: int) -> list[dict[str, Any]]:
    """Rows for a calibration family."""
    if family == "gpt2-ioi":
        return ioi_rows(tokenizer, n, seed)
    if family == "gpt2-induction":
        return induction_rows(tokenizer, n, seed)
    raise ValueError(f"unknown calibration family {family!r}; choose from {CALIBRATION_FAMILIES}")
