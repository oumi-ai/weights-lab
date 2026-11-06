"""Building blocks for new methods: run_with, jacobian, jvp, vjp and sweep."""

import pytest
import torch

from oumi_weights_lab.budget import BudgetExceeded

PROMPT = "hello world yes no hello"
KINDS = ["gpt2", "llama_gqa"]
BLOCK = {"gpt2": "transformer.h.0", "llama_gqa": "model.layers.0"}
LATE = {"gpt2": "transformer.h.1", "llama_gqa": "model.layers.1"}
UP = {"gpt2": "transformer.h.0.mlp.c_fc", "llama_gqa": "model.layers.0.mlp.up_proj"}
ATTN_IN = {
    "gpt2": "transformer.h.0.attn.c_proj",
    "llama_gqa": "model.layers.0.self_attn.o_proj",
}
HIDDEN = 16


@pytest.fixture(params=KINDS)
def setup(request, tiny_model, make_lab):
    model, tokenizer = tiny_model(request.param)
    return make_lab(model, tokenizer), request.param


def hook_counts(model):
    return [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()
    ]


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def same_state(model, before):
    after = model.state_dict()
    return all(torch.equal(after[name], value) for name, value in before.items())


def encoded(lab, prompt):
    """Model inputs without token_type_ids, which the test tokenizer adds and GPT-2 would embed."""
    ids = lab.tokenizer(prompt, return_tensors="pt")
    return {"input_ids": ids["input_ids"], "attention_mask": ids["attention_mask"]}


def logits_of(lab, prompt=PROMPT):
    ids = encoded(lab, prompt)
    with torch.no_grad():
        return lab.model(**ids).logits[:, -1].float()


class Boom(Exception):
    pass


# run_with -------------------------------------------------------------------------------


def test_run_with_matches_patch_and_record(setup):
    lab, kind = setup
    value = torch.randn(1, 1, HIDDEN)
    with lab.patch(BLOCK[kind], value), lab.record([UP[kind]]) as captured:
        expected = logits_of(lab)
    result = lab.run_with(
        PROMPT, activations=[{"module": BLOCK[kind], "value": value}], record=[UP[kind]]
    )
    assert torch.allclose(result["logits"], expected, atol=1e-6)
    assert torch.equal(result["records"][UP[kind]][0], captured[UP[kind]][0])
    assert result["forward_passes"] == 1
    assert result["logits"].dtype == torch.float32 and result["logits"].device.type == "cpu"


def test_run_with_input_intervention_and_record_dicts(setup):
    lab, kind = setup
    with (
        lab.patch(UP[kind], lambda x: x * 0, at="input"),
        lab.record(ATTN_IN[kind], positions="all", at="input") as captured,
    ):
        expected = logits_of(lab)
    result = lab.run_with(
        PROMPT,
        activations=[{"module": UP[kind], "at": "input", "value": lambda x: x * 0}],
        record=[{"module": ATTN_IN[kind], "at": "input", "positions": "all"}],
    )
    assert torch.allclose(result["logits"], expected, atol=1e-6)
    assert result["records"][ATTN_IN[kind]][0].shape[1] == 5
    assert torch.equal(result["records"][ATTN_IN[kind]][0], captured[ATTN_IN[kind]][0])


def test_run_with_logit_modes(setup):
    lab, _ = setup
    assert lab.run_with(PROMPT)["logits"].shape == (1, 6)
    assert lab.run_with(PROMPT, logits="all")["logits"].shape == (1, 5, 6)
    assert lab.run_with(PROMPT, logits=None)["logits"] is None
    with pytest.raises(ValueError, match="logits"):
        lab.run_with(PROMPT, logits="some")


def test_run_with_weights_equals_edit_then_undo(setup):
    lab, kind = setup
    name = UP[kind] + ".weight"
    replacement = torch.randn_like(lab.model.get_parameter(name))
    before = snapshot(lab.model)
    result = lab.run_with(PROMPT, weights={name: replacement})
    assert same_state(lab.model, before)
    lab.edit(name, replacement)
    expected = logits_of(lab)
    lab.undo()
    assert torch.allclose(result["logits"], expected, atol=1e-6)
    assert not torch.allclose(result["logits"], logits_of(lab))
    assert same_state(lab.model, before)


def test_run_with_weights_validation(setup):
    lab, kind = setup
    name = UP[kind] + ".weight"
    with pytest.raises(ValueError, match="shape"):
        lab.run_with(PROMPT, weights={name: torch.zeros(2, 2)})
    with pytest.raises(ValueError, match="not a parameter"):
        lab.run_with(PROMPT, weights={"nope.weight": torch.zeros(1)})
    assert lab.forward_passes == 0


def test_run_with_tied_weights_replace_both_uses(tiny_model, make_lab):
    model, tokenizer = tiny_model("gpt2")
    lab = make_lab(model, tokenizer)
    assert model.lm_head.weight is model.transformer.wte.weight
    replacement = torch.randn_like(model.lm_head.weight)
    before = snapshot(model)
    via_head = lab.run_with(PROMPT, weights={"lm_head.weight": replacement})
    via_embed = lab.run_with(PROMPT, weights={"transformer.wte.weight": replacement})
    assert torch.allclose(via_head["logits"], via_embed["logits"])
    lab.edit("transformer.wte.weight", replacement)
    expected = logits_of(lab)
    lab.undo()
    assert torch.allclose(via_head["logits"], expected, atol=1e-6)
    assert same_state(model, before)
    with pytest.raises(ValueError, match="tied"):
        lab.run_with(
            PROMPT,
            weights={"lm_head.weight": replacement, "transformer.wte.weight": replacement + 1},
        )


def test_run_with_refuses_quantized_parameter(setup, monkeypatch):
    lab, kind = setup
    from oumi_weights_lab import methods

    packed = type("Packed", (), {"method": "nf4"})()
    monkeypatch.setattr(methods, "lookup", lambda model, name, dtype=None: packed)
    with pytest.raises(ValueError, match="lab.edit"):
        lab.run_with(PROMPT, weights={UP[kind] + ".weight": torch.zeros(1)})
    assert lab.forward_passes == 0


def test_run_with_cleans_up_after_error_in_callable(setup):
    lab, kind = setup
    before, counts = snapshot(lab.model), hook_counts(lab.model)
    name = UP[kind] + ".weight"

    def explode(selected):
        raise Boom

    with pytest.raises(Boom):
        lab.run_with(
            PROMPT,
            activations=[{"module": BLOCK[kind], "value": explode}],
            record=[UP[kind]],
            weights={name: torch.zeros_like(lab.model.get_parameter(name))},
        )
    assert hook_counts(lab.model) == counts
    assert same_state(lab.model, before)
    assert lab.patches() == []
    assert lab.forward_passes == 1  # a failed call still counts


def test_run_with_cleans_up_after_keyboard_interrupt(setup):
    lab, kind = setup
    counts = hook_counts(lab.model)

    def interrupt(selected):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        lab.run_with(PROMPT, activations=[{"module": BLOCK[kind], "value": interrupt}])
    assert hook_counts(lab.model) == counts
    assert lab.patches() == []


def test_run_with_grad_reaches_parameters_and_no_grad_detaches(setup):
    lab, kind = setup
    parameter = lab.model.get_parameter(UP[kind] + ".weight")
    flags = [p.requires_grad for p in lab.model.parameters()]
    detached = lab.run_with(PROMPT, record=[UP[kind]])
    assert not detached["logits"].requires_grad
    assert not detached["records"][UP[kind]][0].requires_grad
    kept = lab.run_with(PROMPT, record=[UP[kind]], grad=True)
    assert kept["logits"].requires_grad
    (gradient,) = torch.autograd.grad(kept["logits"].sum(), parameter, allow_unused=True)
    assert gradient is not None and gradient.abs().sum() > 0
    assert kept["records"][UP[kind]][0].grad_fn is not None
    assert [p.requires_grad for p in lab.model.parameters()] == flags
    assert all(p.grad is None for p in lab.model.parameters())


def test_run_with_batch_matches_single_prompts(setup):
    lab, kind = setup
    prompts = ["hello world yes", "hello world yes no hello"]
    batch = lab.run_with(prompts, record=[UP[kind]])
    assert batch["forward_passes"] == 2
    for row, prompt in enumerate(prompts):
        single = lab.run_with(prompt, record=[UP[kind]])
        assert torch.allclose(batch["logits"][row], single["logits"][0], atol=1e-5)
        assert torch.allclose(
            batch["records"][UP[kind]][0][row], single["records"][UP[kind]][0][0], atol=1e-5
        )


def test_run_with_dict_inputs_pass_through(setup):
    lab, _ = setup
    result = lab.run_with(encoded(lab, PROMPT))
    assert torch.allclose(result["logits"], logits_of(lab), atol=1e-6)


def test_absolute_positions_with_padded_batch_raise(setup):
    lab, kind = setup
    prompts = ["hello world yes", "hello world yes no hello"]
    with pytest.raises(ValueError, match="left padding shifts"):
        lab.run_with(prompts, record=[{"module": UP[kind], "positions": 0}])
    with pytest.raises(ValueError, match="left padding shifts"):
        lab.run_with(
            prompts, activations=[{"module": UP[kind], "positions": [1], "value": lambda x: x}]
        )
    assert lab.forward_passes == 0
    lab.run_with(prompts, record=[{"module": UP[kind], "positions": -1}])
    lab.run_with(prompts, record=[{"module": UP[kind], "positions": "all"}])
    # Equal-length prompts have no padding, so absolute positions are fine.
    lab.run_with(["hello world", "yes no"], record=[{"module": UP[kind], "positions": 0}])


def test_run_with_rejects_bad_specs(setup):
    lab, kind = setup
    with pytest.raises(ValueError, match="module"):
        lab.run_with(PROMPT, activations=[{"value": torch.zeros(1)}])
    with pytest.raises(ValueError, match="value"):
        lab.run_with(PROMPT, activations=[{"module": BLOCK[kind]}])
    with pytest.raises(ValueError, match="at must be"):
        lab.run_with(PROMPT, activations=[{"module": BLOCK[kind], "at": "attention", "value": 0}])
    with pytest.raises(ValueError, match="unknown"):
        lab.run_with(PROMPT, activations=[{"module": BLOCK[kind], "value": 0, "extra": 1}])
    assert lab.forward_passes == 0


def test_run_with_counts_against_budget(setup):
    lab, _ = setup
    lab.budget.limit = 1
    lab.run_with(PROMPT)
    with pytest.raises(BudgetExceeded):
        lab.run_with(PROMPT)


# derivatives ----------------------------------------------------------------------------


def test_jacobian_of_linear_layer_matches_weights(setup):
    lab, kind = setup
    weight = lab.model.get_parameter(UP[kind] + ".weight").detach()
    matrix = weight.T if kind == "gpt2" else weight  # Conv1D stores [in, out]
    jac = lab.jacobian(PROMPT, UP[kind], UP[kind], source_at="input", target_at="output")
    assert jac.shape == (1, matrix.shape[0], 1, matrix.shape[1])
    assert torch.allclose(jac[0, :, 0, :], matrix, atol=1e-6)
    assert jac.dtype == torch.float32 and jac.device.type == "cpu"
    assert lab.forward_passes == 1


def test_jacobian_matches_finite_differences_for_a_block(tiny_model, make_lab):
    model, tokenizer = tiny_model("gpt2")
    model.double()
    lab = make_lab(model, tokenizer)
    block = BLOCK["gpt2"]
    jac = lab.jacobian(PROMPT, block, block, source_at="input", target_at="output")
    base = lab.run_with(PROMPT, record=[{"module": block, "at": "input"}], logits=None)["records"][
        block
    ][0]
    eps = 1e-6
    for index in range(HIDDEN):
        shift = torch.zeros_like(base)
        shift[..., index] = eps
        outs = []
        for sign in (1, -1):
            run = lab.run_with(
                PROMPT,
                activations=[{"module": block, "at": "input", "value": base + sign * shift}],
                record=[block],
                logits=None,
            )
            outs.append(run["records"][block][0][0, 0])
        numeric = (outs[0] - outs[1]) / (2 * eps)
        assert torch.allclose(jac[0, :, 0, index].double(), numeric, atol=1e-4)


def test_jvp_and_vjp_match_the_jacobian(setup):
    lab, kind = setup
    source, target = BLOCK[kind], LATE[kind]
    kwargs = {"source_at": "output", "target_at": "output"}
    jac = lab.jacobian(PROMPT, source, target, **kwargs)
    t_shape, s_shape = jac.shape[: jac.ndim // 2], jac.shape[jac.ndim // 2 :]
    assert jac.abs().sum() > 0
    tangent, cotangent = torch.randn(s_shape), torch.randn(t_shape)
    flat = jac.reshape(t_shape.numel(), s_shape.numel())
    value, derivative = lab.jvp(PROMPT, source, target, tangent, **kwargs)
    assert torch.allclose(derivative, (flat @ tangent.reshape(-1)).reshape(t_shape), atol=1e-5)
    value2, gradient = lab.vjp(PROMPT, source, target, cotangent, **kwargs)
    assert torch.allclose(gradient, (cotangent.reshape(-1) @ flat).reshape(s_shape), atol=1e-5)
    recorded = lab.run_with(PROMPT, record=[target])["records"][target][0]
    assert torch.allclose(value, recorded[0], atol=1e-6)
    assert torch.equal(value, value2)
    assert lab.forward_passes == 4  # jacobian, jvp, vjp, run_with


def test_derivatives_leave_no_gradients_flags_hooks_or_modes(setup):
    lab, kind = setup
    flags = [p.requires_grad for p in lab.model.parameters()]
    counts = hook_counts(lab.model)
    lab.model.train()
    modes = [m.training for m in lab.model.modules()]
    lab.jacobian(PROMPT, BLOCK[kind], LATE[kind])
    assert [p.requires_grad for p in lab.model.parameters()] == flags
    assert all(p.grad is None for p in lab.model.parameters())
    assert hook_counts(lab.model) == counts
    assert [m.training for m in lab.model.modules()] == modes


def test_target_before_source_gives_zero(setup):
    lab, kind = setup
    assert lab.jacobian(PROMPT, BLOCK[kind], LATE[kind]).abs().sum() > 0
    assert lab.jacobian(PROMPT, LATE[kind], BLOCK[kind]).abs().sum() == 0


def test_source_or_target_that_runs_twice_or_never_raises(setup):
    lab, kind = setup
    module = lab.model.get_submodule(UP[kind])
    root = lab.model.get_submodule(BLOCK[kind])

    def again(hooked, args, kwargs):
        module(torch.zeros(1, 1, HIDDEN))

    handle = root.register_forward_pre_hook(again, with_kwargs=True)
    try:
        with pytest.raises(ValueError, match=UP[kind].replace(".", r"\.")):
            lab.jacobian(PROMPT, UP[kind], LATE[kind], source_at="input")
        with pytest.raises(ValueError, match=UP[kind].replace(".", r"\.")):
            lab.jacobian(PROMPT, BLOCK[kind], UP[kind])
    finally:
        handle.remove()
    assert lab.jacobian(PROMPT, UP[kind], LATE[kind], source_at="input").shape[-2:] == (1, HIDDEN)
    # lm_head runs once; a module the forward never reaches must be named.
    spare = torch.nn.Linear(HIDDEN, HIDDEN)
    lab.model.add_module("spare", spare)
    try:
        with pytest.raises(ValueError, match="spare.*did not run"):
            lab.jacobian(PROMPT, BLOCK[kind], "spare")
    finally:
        del lab.model.spare


def test_derivative_input_validation(setup):
    lab, kind = setup
    with pytest.raises(ValueError, match="at must be"):
        lab.jacobian(PROMPT, BLOCK[kind], UP[kind], source_at="attention")
    with pytest.raises(ValueError, match="one sequence"):
        lab.jacobian(["hello world", "yes no"], BLOCK[kind], UP[kind])
    with pytest.raises(ValueError, match="tangent"):
        lab.jvp(PROMPT, BLOCK[kind], UP[kind], torch.zeros(3))
    with pytest.raises(ValueError, match="cotangent"):
        lab.vjp(PROMPT, BLOCK[kind], UP[kind], torch.zeros(3))


# sweep ----------------------------------------------------------------------------------


def heads(kind, head):
    """An intervention zeroing one head's slice of the o_proj input at the last token."""
    mask = torch.ones(HIDDEN)
    mask[head * 4 : (head + 1) * 4] = 0
    return {"module": ATTN_IN[kind], "at": "input", "value": lambda x, m=mask: x * m}


def test_sweep_rows_equal_run_with(setup):
    lab, kind = setup
    entries = [None] + [[heads(kind, h)] for h in range(2)]
    entries.append([heads(kind, 0), {"module": BLOCK[kind], "value": torch.ones(HIDDEN)}])
    results = lab.sweep(PROMPT, entries, record=[UP[kind]], batch_size=3)
    assert len(results) == len(entries)
    for entry, result in zip(entries, results, strict=True):
        alone = lab.run_with(PROMPT, activations=entry or (), record=[UP[kind]])
        assert result["logits"].shape == alone["logits"].shape
        assert torch.allclose(result["logits"], alone["logits"], atol=1e-5)
        assert torch.allclose(
            result["records"][UP[kind]][0], alone["records"][UP[kind]][0], atol=1e-5
        )
    assert torch.allclose(results[0]["logits"], logits_of(lab), atol=1e-5)
    assert not torch.allclose(results[1]["logits"], results[0]["logits"])


def test_sweep_rows_on_one_module_stay_separate(setup):
    lab, kind = setup
    values = [torch.full((HIDDEN,), float(i)) for i in range(4)]
    entries = [[{"module": BLOCK[kind], "value": value}] for value in values]
    results = lab.sweep(PROMPT, entries, record=[BLOCK[kind]], batch_size=4)
    for value, result in zip(values, results, strict=True):
        recorded = result["records"][BLOCK[kind]][0]
        assert recorded.shape[0] == 1
        assert torch.allclose(recorded[0, 0], value)
    assert len({tuple(r["logits"].flatten().tolist()) for r in results}) == 4


def test_sweep_counts_one_pass_per_row_and_leaves_no_hooks(setup):
    lab, kind = setup
    counts = hook_counts(lab.model)
    lab.sweep(PROMPT, [None, [heads(kind, 0)]] * 3, batch_size=4)
    assert lab.forward_passes == 6
    assert hook_counts(lab.model) == counts
    assert lab.patches() == []


def test_sweep_fails_before_the_chunk_that_crosses_the_budget(setup):
    lab, _ = setup
    lab.budget.limit = 5
    counts = hook_counts(lab.model)
    with pytest.raises(BudgetExceeded):
        lab.sweep(PROMPT, [None] * 8, batch_size=3)
    # Chunks of 3, 3, 2: the first runs, the second would reach 6 and never starts.
    assert lab.forward_passes == 3
    assert hook_counts(lab.model) == counts


def test_sweep_cleans_up_after_error(setup):
    lab, kind = setup
    counts, before = hook_counts(lab.model), snapshot(lab.model)

    def explode(selected):
        raise Boom

    with pytest.raises(Boom):
        lab.sweep(PROMPT, [None, [{"module": BLOCK[kind], "value": explode}]])
    assert hook_counts(lab.model) == counts
    assert same_state(lab.model, before)


def test_sweep_input_forms_and_validation(setup):
    lab, kind = setup
    result = lab.sweep(encoded(lab, PROMPT), [None, None])
    assert torch.allclose(result[1]["logits"], logits_of(lab), atol=1e-5)
    with pytest.raises(ValueError, match="one prompt"):
        lab.sweep(["hello world", "yes no"], [None])
    with pytest.raises(ValueError, match="interventions"):
        lab.sweep(PROMPT, [])
    with pytest.raises(ValueError, match="batch_size"):
        lab.sweep(PROMPT, [None], batch_size=0)
    with pytest.raises(ValueError, match="unknown"):
        lab.sweep(PROMPT, [[{"module": BLOCK[kind], "value": 0, "weights": 1}]])
    assert lab.forward_passes == 1 + 2


def test_sweep_positions_negative_and_absolute(setup):
    lab, kind = setup
    first = [{"module": BLOCK[kind], "positions": 0, "value": torch.zeros(HIDDEN)}]
    last = [{"module": BLOCK[kind], "positions": -1, "value": torch.zeros(HIDDEN)}]
    results = lab.sweep(PROMPT, [first, last])
    for entry, result in zip([first, last], results, strict=True):
        alone = lab.run_with(PROMPT, activations=entry)
        assert torch.allclose(result["logits"], alone["logits"], atol=1e-5)


# text preparation -----------------------------------------------------------------------


def test_text_is_prepared_the_way_scoring_prepares_it(setup):
    lab, _ = setup
    lab.tokenizer.bos_token = None
    calls = []
    lab.tokenizer.encode = lambda text, add_special_tokens=True, original=lab.tokenizer.encode: (
        calls.append(add_special_tokens),
        original(text, add_special_tokens=add_special_tokens),
    )[1]
    with lab.attach(lambda prompt: "world " + prompt, mode="prefix"):
        result = lab.run_with("hello")
        sweep = lab.sweep("hello", [None])
        jac = lab.jacobian("hello", "lm_head", "lm_head", source_at="input")
    assert calls and not any(calls)  # special tokens follow scoring's rule: none added
    expected = lab.run_with(encoded(lab, "world hello"))
    assert torch.allclose(result["logits"], expected["logits"], atol=1e-6)
    assert torch.allclose(sweep[0]["logits"], expected["logits"], atol=1e-5)
    assert jac.shape == (1, 6, 1, HIDDEN)
    # Without the attachment the prompt is just "hello".
    assert not torch.allclose(lab.run_with("hello")["logits"], expected["logits"])


def test_logit_attachments_apply_to_returned_logits(setup):
    lab, _ = setup
    plain = lab.run_with(PROMPT)["logits"]
    with lab.attach(lambda *_: torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])):
        shifted = lab.run_with(PROMPT)["logits"]
        swept = lab.sweep(PROMPT, [None])[0]["logits"]
        with pytest.raises(ValueError, match="logit attachment"):
            lab.run_with(["hello world", "yes no hello"])
    assert torch.allclose(shifted - plain, torch.tensor([[0.0, 0, 0, 0, 10, 0]]), atol=1e-5)
    assert torch.allclose(swept, shifted, atol=1e-5)


def test_one_tensor_under_both_tied_names_is_accepted_when_it_needs_a_cast(tiny_model, make_lab):
    model, tokenizer = tiny_model("gpt2")
    lab = make_lab(model, tokenizer)
    replacement = torch.randn(model.lm_head.weight.shape, dtype=torch.float64)
    assert replacement.dtype != model.lm_head.weight.dtype
    both = lab.run_with(
        PROMPT, weights={"lm_head.weight": replacement, "transformer.wte.weight": replacement}
    )
    one = lab.run_with(PROMPT, weights={"lm_head.weight": replacement})
    assert torch.allclose(both["logits"], one["logits"], atol=1e-6)
    with pytest.raises(ValueError, match="tied"):
        lab.run_with(
            PROMPT,
            weights={"lm_head.weight": replacement, "transformer.wte.weight": replacement.clone()},
        )
