import pytest
import torch
import torch.nn.functional as functional

PROMPT, TARGET = "hello world yes", " no hello"
MLP = "transformer.h.0.mlp"
PROJ = "transformer.h.0.mlp.c_proj"


class Interrupted(BaseException):
    pass


@pytest.fixture
def lab(tiny_model, make_lab):
    return make_lab(*tiny_model("gpt2"))


def inputs(lab):
    prefix = lab.tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    answer = lab.tokenizer(TARGET, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prefix + answer])
    labels = torch.tensor([[-100] * len(prefix) + answer])
    return ids, labels


def reference(lab, name, at):
    """The retain_grad reference: one hook, the same masked loss, then backward."""
    ids, labels = inputs(lab)
    kept = []
    module = lab.model.get_submodule(name)
    if at == "output":

        def hook(_, args, output):
            output.retain_grad()
            kept.append(output)

        handle = module.register_forward_hook(hook)
    else:

        def hook(_, args):
            args[0].retain_grad()
            kept.append(args[0])

        handle = module.register_forward_pre_hook(hook)
    try:
        with torch.enable_grad():
            logits = lab.model(input_ids=ids).logits
            loss = functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
            )
            loss.backward()
        return kept[0].grad.clone()
    finally:
        handle.remove()
        lab.model.zero_grad()


def hook_counts(lab, names):
    total = 0
    for name in names:
        module = lab.model.get_submodule(name)
        total += len(module._forward_hooks) + len(module._forward_pre_hooks)
    return total


@pytest.mark.parametrize(("name", "at"), [(MLP, "output"), (PROJ, "input")])
def test_activation_gradient_matches_retain_grad(lab, name, at):
    # The reference needs gradient to flow through the model, so it runs with parameters trainable.
    for parameter in lab.model.parameters():
        parameter.requires_grad_(True)
    want = reference(lab, name, at)
    got = lab.gradient(PROMPT, target=TARGET, modules=[name], at=at)
    assert list(got) == [name]
    assert got[name].shape == (1, *want.shape[1:])
    assert got[name].device.type == "cpu"
    assert torch.allclose(got[name][0], want[0], atol=1e-6)
    assert got[name].abs().sum() > 0


def test_activation_positions(lab):
    full = lab.gradient(PROMPT, target=TARGET, modules=MLP)[MLP]
    one = lab.gradient(PROMPT, target=TARGET, modules=MLP, positions=[-2])[MLP]
    assert one.shape == (1, 1, full.shape[2])
    assert torch.equal(one, full[:, -2:-1])
    sliced = lab.gradient(PROMPT, target=TARGET, modules=MLP, positions=slice(1, 4))[MLP]
    assert torch.equal(sliced, full[:, 1:4])
    scalar = lab.gradient(PROMPT, target=TARGET, modules=MLP, positions=0)[MLP]
    assert torch.equal(scalar, full[:, :1])


def test_last_position_output_gradient_is_zero(lab):
    # The last position predicts nothing scored, so "all" is the useful default.
    full = lab.gradient(PROMPT, target=TARGET, modules="lm_head")["lm_head"]
    assert full.shape[1] == 5
    assert full[:, -1].abs().sum() == 0
    assert full[:, :-1].abs().sum() > 0


def test_frozen_model_activation_gradient(lab):
    want = lab.gradient(PROMPT, target=TARGET, modules=[MLP])[MLP]
    for parameter in lab.model.parameters():
        parameter.requires_grad_(False)
    got = lab.gradient(PROMPT, target=TARGET, modules=[MLP])[MLP]
    assert got.abs().sum() > 0
    assert torch.allclose(got, want, atol=1e-6)
    frozen_input = lab.gradient(PROMPT, target=TARGET, modules=[PROJ], at="input")[PROJ]
    assert frozen_input.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in lab.model.parameters())


def test_modules_only_returns_no_parameters(lab):
    assert list(lab.gradient(PROMPT, target=TARGET, modules=MLP)) == [MLP]
    assert list(lab.gradient(PROMPT, target=TARGET, modules=[MLP, PROJ], at="input")) == [MLP, PROJ]


def test_modules_with_parameters(lab):
    name = "transformer.h.0.mlp.c_fc.weight"
    got = lab.gradient(PROMPT, target=TARGET, modules=[MLP], parameters=[name])
    assert list(got) == [MLP, name]
    plain = lab.gradient(PROMPT, target=TARGET, parameters=[name])[name]
    assert torch.allclose(got[name], plain, atol=1e-6)
    assert got[name].shape[0] == got[MLP].shape[0] == 1


def test_parameter_gradient_unchanged_without_modules(lab):
    names = list(lab.gradient(PROMPT, target=TARGET))
    assert MLP not in names
    assert "transformer.h.0.mlp.c_fc.weight" in names


def test_activation_gradient_pass_count(lab):
    ids = torch.tensor([[2, 3, 4, 5], [3, 2, 5, 4], [4, 5, 2, 3]])
    before = lab.forward_passes
    got = lab.gradient({"input_ids": ids}, modules=[MLP])[MLP]
    assert got.shape[:2] == (3, 4)
    assert lab.forward_passes - before == 3


def test_activation_rows_match_single_rows(lab):
    ids = torch.tensor([[2, 3, 4, 5], [3, 2, 5, 4]])
    batch = lab.gradient({"input_ids": ids}, modules=[MLP])[MLP]
    for row in range(2):
        alone = lab.gradient({"input_ids": ids[row : row + 1]}, modules=[MLP])[MLP]
        assert torch.allclose(batch[row : row + 1], alone, atol=1e-6)


def test_module_called_twice_raises(lab):
    block = lab.model.transformer.h[0]
    original = block.forward

    def twice(*args, **kwargs):
        block.mlp(torch.zeros(1, 2, 16))
        return original(*args, **kwargs)

    block.forward = twice
    with pytest.raises(ValueError, match=r"transformer\.h\.0\.mlp"):
        lab.gradient(PROMPT, target=TARGET, modules=[MLP])
    assert hook_counts(lab, [MLP]) == 0


def test_activation_gradient_hooks_removed(lab):
    names = [MLP, PROJ]
    lab.gradient(PROMPT, target=TARGET, modules=names)
    lab.gradient(PROMPT, target=TARGET, modules=[PROJ], at="input")
    assert hook_counts(lab, names) == 0

    def explode(_):
        raise RuntimeError("boom")

    with lab.patch(MLP, explode), pytest.raises(RuntimeError, match="boom"):
        lab.gradient(PROMPT, target=TARGET, modules=names)
    assert hook_counts(lab, names) == 0
    with pytest.raises(ValueError):
        lab.gradient(PROMPT, target=TARGET, modules=names, at="attention")
    assert hook_counts(lab, names) == 0


def test_activation_gradient_interrupt_during_registration(lab, monkeypatch):
    from oumi_weights_lab import inspection

    original = inspection._TemporaryHook.register
    target = lab.model.get_submodule(PROJ)

    def register(self, **kwargs):
        original(self, **kwargs)
        if self.module is target:
            raise Interrupted

    monkeypatch.setattr(inspection._TemporaryHook, "register", register)
    with pytest.raises(Interrupted):
        lab.gradient(PROMPT, target=TARGET, modules=[MLP, PROJ])
    assert hook_counts(lab, [MLP, PROJ]) == 0


def test_activation_gradient_restores_state(lab):
    lab.model.train()
    lab.model.transformer.h[0].mlp.eval()
    modes = [m.training for m in lab.model.modules()]
    torch.manual_seed(5)
    state = torch.get_rng_state()
    lab.gradient(PROMPT, target=TARGET, modules=[MLP], parameters=["lm_head.weight"])
    assert [m.training for m in lab.model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), state)


def test_activation_gradient_preserves_forward_values(lab):
    ids, _ = inputs(lab)
    for parameter in lab.model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        want = lab.model(input_ids=ids).logits
    with lab.record("lm_head", positions="all") as seen:
        lab.gradient(PROMPT, target=TARGET, modules=[MLP])
    assert torch.allclose(seen["lm_head"][0], want, atol=1e-6)


def test_activation_gradient_refusals(lab):
    with pytest.raises(ValueError, match="at must be"):
        lab.gradient(PROMPT, target=TARGET, modules=MLP, at="attention")
    with pytest.raises(ValueError, match="modules"):
        lab.gradient(PROMPT, target=TARGET, positions=[-1])
    with pytest.raises(ValueError, match="modules"):
        lab.gradient(PROMPT, target=TARGET, at="input")
    with pytest.raises(AttributeError, match="nonexistent"):
        lab.gradient(PROMPT, target=TARGET, modules=["transformer.h.0.nonexistent"])
    with pytest.raises(ValueError, match="distinct"):
        lab.gradient(PROMPT, target=TARGET, modules=[MLP, MLP])
    with pytest.raises(ValueError, match="nonempty"):
        lab.gradient(PROMPT, target=TARGET, modules=[])
    with pytest.raises(ValueError, match="positions"):
        lab.gradient(PROMPT, target=TARGET, modules=MLP, positions=[])
    with pytest.raises(ValueError, match="outside"):
        lab.gradient(PROMPT, target=TARGET, modules=MLP, positions=[50])


def test_module_and_parameter_names_must_not_collide(lab, monkeypatch):
    # Real names never collide; force it to check the guard.
    monkeypatch.setattr(lab.model, "named_parameters", lambda **_: iter([("lm_head", object())]))
    with pytest.raises(ValueError, match="both"):
        lab.gradient(PROMPT, target=TARGET, modules=["lm_head"], parameters=["lm_head"])


def test_module_that_never_runs_raises(lab):
    lab.model.transformer.h[0].extra = torch.nn.Linear(16, 16)
    with pytest.raises(ValueError, match="did not run"):
        lab.gradient(PROMPT, target=TARGET, modules=["transformer.h.0.extra"])


LN = "transformer.h.0.ln_1"


def standin_reference(lab, name):
    """Gradient at a view of the module's input that only the module sees."""
    ids, labels = inputs(lab)
    kept = []

    def hook(_, args):
        view = args[0].view_as(args[0])
        view.retain_grad()
        kept.append(view)
        return (view, *args[1:])

    handle = lab.model.get_submodule(name).register_forward_pre_hook(hook)
    try:
        with torch.enable_grad():
            logits = lab.model(input_ids=ids).logits
            loss = functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
            )
            loss.backward()
        return kept[0].grad.clone()
    finally:
        handle.remove()
        lab.model.zero_grad()


def test_input_gradient_counts_only_the_modules_use(lab):
    for parameter in lab.model.parameters():
        parameter.requires_grad_(True)
    want = standin_reference(lab, LN)
    raw = reference(lab, LN, "input")
    got = lab.gradient(PROMPT, target=TARGET, modules=LN, at="input")[LN]
    assert torch.allclose(got[0], want[0], atol=1e-6)
    assert not torch.allclose(got[0], raw[0], atol=1e-6)


def test_input_gradient_same_frozen_or_trainable(lab):
    trainable = lab.gradient(PROMPT, target=TARGET, modules=LN, at="input")[LN]
    for parameter in lab.model.parameters():
        parameter.requires_grad_(False)
    frozen = lab.gradient(PROMPT, target=TARGET, modules=LN, at="input")[LN]
    assert frozen.abs().sum() > 0
    assert torch.allclose(frozen, trainable, atol=1e-6)
