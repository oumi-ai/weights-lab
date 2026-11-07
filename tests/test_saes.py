import json
import sys

import pytest

pytest.importorskip("sae_lens")

import torch  # noqa: E402

from oumi_weights_lab import saes  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:(?s).*model_from_pretrained_kwargs")

D = 16


def build_gemma2(tokenizer):
    from transformers import AutoModelForCausalLM, Gemma2Config

    config = Gemma2Config(
        vocab_size=6,
        hidden_size=D,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        sliding_window=8,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=1,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()


@pytest.fixture
def lab_for(tiny_model, make_lab, tiny_tokenizer):
    def build(kind):
        if kind == "gemma2":
            return make_lab(build_gemma2(tiny_tokenizer), tiny_tokenizer)
        return make_lab(*tiny_model(kind))

    return build


def make_sae(
    path,
    hook_name,
    *,
    architecture="standard",
    d_in=D,
    d_sae=24,
    d_out=None,
    model_name="test-model",
    seed=0,
    kwargs=None,
    hook_name_out=None,
    **config,
):
    """Save a small SAELens SAE with random weights and return it."""
    import sae_lens
    from sae_lens.saes.sae import SAEMetadata

    metadata = {"model_name": model_name, "hook_name": hook_name, "prepend_bos": False}
    if kwargs is not None:
        metadata["model_from_pretrained_kwargs"] = kwargs
    if hook_name_out is not None:
        metadata["hook_name_out"] = hook_name_out
    classes = {
        "standard": (sae_lens.StandardSAE, sae_lens.StandardSAEConfig),
        "jumprelu": (sae_lens.JumpReLUSAE, sae_lens.JumpReLUSAEConfig),
        "topk": (sae_lens.TopKSAE, sae_lens.TopKSAEConfig),
        "gated": (sae_lens.GatedSAE, sae_lens.GatedSAEConfig),
        "transcoder": (sae_lens.Transcoder, sae_lens.TranscoderConfig),
        "skip_transcoder": (sae_lens.SkipTranscoder, sae_lens.SkipTranscoderConfig),
    }
    model_class, config_class = classes[architecture]
    if "transcoder" in architecture:
        config = {"d_out": d_out or d_in, "apply_b_dec_to_input": False, **config}
    cfg = config_class(d_in=d_in, d_sae=d_sae, metadata=SAEMetadata(**metadata), **config)
    sae = model_class(cfg)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in sae.named_parameters():
            values = torch.randn(parameter.shape, generator=generator) * 0.3
            if name == "threshold":
                values = values.abs() * 0.1
            parameter.copy_(values)
    sae.save_model(path)
    return sae


def run(lab, text="hello world yes no hello"):
    inputs = lab.tokenizer(text, return_tensors="pt", return_token_type_ids=False)
    with torch.no_grad():
        return lab.model(**inputs).logits


def record(lab, module, at, positions="all", text="hello world yes no hello"):
    with lab.record(module, positions=positions, at=at) as captured:
        run(lab, text)
    return captured[module][0]


# Hook-name mapping


def test_gpt2_sites(lab_for):
    lab = lab_for("gpt2")
    expected = {
        "blocks.1.hook_resid_pre": ("transformer.h.1", "input"),
        "blocks.1.hook_resid_post": ("transformer.h.1", "output"),
        "blocks.1.hook_resid_mid": ("transformer.h.1.ln_2", "input"),
        "blocks.1.hook_mlp_in": ("transformer.h.1.ln_2", "input"),
        "blocks.1.hook_attn_out": ("transformer.h.1.attn", "output"),
        "blocks.1.hook_mlp_out": ("transformer.h.1.mlp", "output"),
        "blocks.1.attn.hook_z": ("transformer.h.1.attn.c_proj", "input"),
        "blocks.1.ln2.hook_normalized": ("transformer.h.1.ln_2", "input"),
        "blocks.1.mlp.hook_in": ("transformer.h.1.mlp", "input"),
    }
    for hook, (module, at) in expected.items():
        site = saes.hook_site(lab, hook)
        assert (site.module, site.at) == (module, at), hook
        assert site.width == D
    assert saes.hook_site(lab, "blocks.1.attn.hook_z").width == D


def test_llama_sites(lab_for):
    lab = lab_for("llama_gqa")
    expected = {
        "blocks.0.hook_resid_pre": ("model.layers.0", "input"),
        "blocks.0.hook_resid_post": ("model.layers.0", "output"),
        "blocks.0.hook_resid_mid": ("model.layers.0.post_attention_layernorm", "input"),
        "blocks.0.hook_attn_out": ("model.layers.0.self_attn", "output"),
        "blocks.0.hook_mlp_out": ("model.layers.0.mlp", "output"),
        "blocks.0.attn.hook_z": ("model.layers.0.self_attn.o_proj", "input"),
        "blocks.0.ln2.hook_normalized": ("model.layers.0.post_attention_layernorm", "input"),
    }
    for hook, (module, at) in expected.items():
        site = saes.hook_site(lab, hook)
        assert (site.module, site.at) == (module, at), hook
    assert saes.hook_site(lab, "blocks.0.attn.hook_z").width == D


def test_gemma2_sites_read_the_extra_norms(lab_for):
    lab = lab_for("gemma2")
    expected = {
        "blocks.1.hook_resid_pre": ("model.layers.1", "input"),
        "blocks.1.hook_resid_post": ("model.layers.1", "output"),
        "blocks.1.hook_resid_mid": ("model.layers.1.pre_feedforward_layernorm", "input"),
        # The residual stream gets the post-normalized sublayer outputs.
        "blocks.1.hook_attn_out": ("model.layers.1.post_attention_layernorm", "output"),
        "blocks.1.hook_mlp_out": ("model.layers.1.post_feedforward_layernorm", "output"),
        "blocks.1.attn.hook_z": ("model.layers.1.self_attn.o_proj", "input"),
        "blocks.1.ln2.hook_normalized": ("model.layers.1.pre_feedforward_layernorm", "input"),
    }
    for hook, (module, at) in expected.items():
        site = saes.hook_site(lab, hook)
        assert (site.module, site.at) == (module, at), hook
    # hook_z is the output projection's input: 4 heads of width 4.
    assert saes.hook_site(lab, "blocks.1.attn.hook_z").width == 16


def test_unmapped_and_out_of_range_hooks_are_refused(lab_for):
    lab = lab_for("gpt2")
    for hook in ["hook_embed", "blocks.0.hook_q_input", "ln_final.hook_normalized", "nonsense"]:
        with pytest.raises(ValueError, match="not mapped|no mapping"):
            saes.hook_site(lab, hook)
    with pytest.raises(ValueError, match="layer 7"):
        saes.hook_site(lab, "blocks.7.hook_resid_pre")


def test_centering_follows_the_models_kwargs_and_only_layer_norm_models(lab_for):
    gpt2, llama = lab_for("gpt2"), lab_for("llama_gqa")
    plain = saes.hook_site(gpt2, "blocks.0.hook_resid_pre")
    centered = saes.hook_site(gpt2, "blocks.0.hook_resid_pre", {"center_writing_weights": True})
    assert not plain.center and centered.center
    # RMSNorm models have no centering in TransformerLens.
    assert not saes.hook_site(
        llama, "blocks.0.hook_resid_pre", {"center_writing_weights": True}
    ).center
    x = torch.randn(2, 3, D) + 5
    assert torch.allclose(centered.apply(x).mean(-1), torch.zeros(2, 3), atol=1e-5)
    assert torch.equal(plain.apply(x), x)


def test_normalized_site_matches_the_models_own_norm(lab_for):
    for kind, norm_name in [
        ("gpt2", "transformer.h.0.ln_2"),
        ("llama_gqa", "model.layers.0.post_attention_layernorm"),
        ("gemma2", "model.layers.0.pre_feedforward_layernorm"),
    ]:
        lab = lab_for(kind)
        norm = lab.model.get_submodule(norm_name)
        # With unit scale and no bias the module's output is the bare normalization.
        with torch.no_grad():
            if hasattr(norm, "bias") and norm.bias is not None:
                norm.bias.zero_()
            norm.weight.fill_(1.0 if kind != "gemma2" else 0.0)
        site = saes.hook_site(lab, "blocks.0.ln2.hook_normalized")
        x = torch.randn(2, 3, D) * 3 + 1
        with torch.no_grad():
            assert torch.allclose(site.apply(x), norm(x), atol=1e-5), kind
        assert not site.writable


def test_value_bias_is_removed_from_hook_z_when_folded(lab_for):
    lab = lab_for("gpt2")
    module = lab.model.transformer.h[0].attn
    z = torch.randn(1, 3, D)
    with torch.no_grad():
        module.c_attn.bias[2 * D :] = torch.arange(D, dtype=torch.float32)
    plain = saes.hook_site(lab, "blocks.0.attn.hook_z")
    folded = saes.hook_site(lab, "blocks.0.attn.hook_z", {"fold_value_biases": True})
    assert torch.equal(plain.apply(z), z)
    assert torch.allclose(folded.apply(z), z - torch.arange(D, dtype=torch.float32), atol=1e-6)
    # With fold_ln the layer norm's bias reaches the values too.
    both = saes.hook_site(lab, "blocks.0.attn.hook_z", {"fold_value_biases": True, "fold_ln": True})
    ln_bias = lab.model.transformer.h[0].ln_1.bias
    weight = module.c_attn.weight[:, 2 * D :]
    expected = z - (torch.arange(D, dtype=torch.float32) + ln_bias @ weight)
    assert torch.allclose(both.apply(z), expected, atol=1e-5)


# Loading


def test_load_sae_reads_a_saved_sae(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post", d_sae=24)
    sae = lab.load_sae(path=tmp_path / "sae")
    assert isinstance(sae, saes.LabSAE)
    assert (sae.module, sae.at) == ("transformer.h.1", "output")
    assert sae.hook_name == "blocks.1.hook_resid_post"
    assert (sae.d_in, sae.d_sae, sae.kind) == (D, 24, "sae")
    assert sae.output_module is None and sae.output_at is None
    assert sae.model_check == "matched"
    assert lab.forward_passes == 0


def test_load_sae_refuses_a_width_mismatch_before_any_pass(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post", d_in=8)
    with pytest.raises(ValueError, match="width 16.*d_in 8"):
        lab.load_sae(path=tmp_path / "sae")
    assert lab.forward_passes == 0


def test_load_sae_refuses_an_unmapped_hook(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_q_input")
    with pytest.raises(ValueError, match="not mapped"):
        lab.load_sae(path=tmp_path / "sae")


def test_load_sae_refuses_a_known_model_mismatch(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post", model_name="gpt2-small")
    with pytest.raises(ValueError, match="gpt2-small.*test-model"):
        lab.load_sae(path=tmp_path / "sae")
    make_sae(tmp_path / "hub", "blocks.1.hook_resid_post", model_name="org/other-model")
    with pytest.raises(ValueError, match="org/other-model"):
        lab.load_sae(path=tmp_path / "hub")


def test_model_name_table_matches_hub_ids(lab_for, tmp_path):
    lab = lab_for("gpt2")
    lab.model_id = "openai-community/gpt2"
    for name in ["gpt2", "gpt2-small"]:
        make_sae(tmp_path / name, "blocks.1.hook_resid_post", model_name=name)
        assert lab.load_sae(path=tmp_path / name).model_check == "matched"
    lab.model_id = "google/gemma-2-2b"
    make_sae(tmp_path / "gemma", "blocks.1.hook_resid_post", model_name="gemma-2-2b")
    assert lab.load_sae(path=tmp_path / "gemma").model_check == "matched"
    lab.model_id = "google/gemma-2-2b-it"
    with pytest.raises(ValueError, match="gemma-2-2b"):
        lab.load_sae(path=tmp_path / "gemma")


def test_unknown_model_names_continue_on_the_shape_check(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post", model_name="mystery-model")
    with pytest.warns(UserWarning, match="could not be checked"):
        sae = lab.load_sae(path=tmp_path / "sae")
    assert sae.model_check.startswith("not checked")
    # A deliberate override loads a fine-tune of the model the SAE was trained on.
    make_sae(tmp_path / "known", "blocks.1.hook_resid_post", model_name="gpt2-small")
    sae = lab.load_sae(path=tmp_path / "known", allow_model_mismatch=True)
    assert sae.model_check.startswith("not checked")


def test_load_sae_refuses_adapter_sessions(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    lab.model.peft_config = {"default": object()}
    with pytest.raises(ValueError, match="adapter|PEFT"):
        lab.load_sae(path=tmp_path / "sae")


def test_load_sae_refuses_normalization_it_cannot_undo(lab_for, tmp_path):
    lab = lab_for("gpt2")
    for mode in ["covariance_whitening", "expected_average_only_in"]:
        make_sae(tmp_path / mode, "blocks.1.hook_resid_post", normalize_activations=mode)
        with pytest.raises(ValueError, match=mode):
            lab.load_sae(path=tmp_path / mode)


@pytest.mark.parametrize("mode", ["layer_norm", "constant_norm_rescale"])
def test_normalizing_saes_read_and_patch_in_model_units(lab_for, tmp_path, mode):
    lab = lab_for("llama_gqa")
    reference = make_sae(tmp_path / "sae", "blocks.0.hook_resid_post", normalize_activations=mode)
    sae = lab.load_sae(path=tmp_path / "sae")
    before = record(lab, sae.module, sae.at)
    with torch.no_grad():
        features = sae.encode(before)
        assert torch.equal(features, reference.encode(before))
        # SAELens undoes the input scaling on the way out.
        assert torch.equal(sae.decode(features), reference.decode(features))
    result = sae.features("hello world yes no hello", positions="all")
    error = (before[0] - reference(before)[0]).detach()
    assert result["mse"] == pytest.approx(float(error.pow(2).mean()), rel=1e-4)
    row = sae.decoder_rows([6])[0]
    scale = sae._output_scale(before)
    with sae.patch({6: lambda value: value + 1.0}, positions="all"):
        after = record(lab, sae.module, sae.at)
    assert torch.allclose(after - before, row * scale, atol=1e-5)
    with sae.steer({6: 2.0}, positions="all"):
        steered = record(lab, sae.module, sae.at)
    assert torch.allclose(steered - before, 2.0 * row * scale, atol=1e-5)
    baseline = run(lab)
    with sae.patch({6: lambda value: value}, positions="all"):
        assert torch.equal(run(lab), baseline)


def test_model_kwargs_replace_what_the_sae_records(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    assert not lab.load_sae(path=tmp_path / "sae").site.center
    sae = lab.load_sae(path=tmp_path / "sae", model_kwargs={"center_writing_weights": True})
    assert sae.site.center
    make_sae(
        tmp_path / "recorded", "blocks.1.hook_resid_post", kwargs={"center_writing_weights": True}
    )
    assert lab.load_sae(path=tmp_path / "recorded").site.center
    assert not lab.load_sae(path=tmp_path / "recorded", model_kwargs={}).site.center


def test_load_sae_needs_a_release_or_a_path(lab_for):
    lab = lab_for("gpt2")
    with pytest.raises(ValueError, match="release and sae_id, or path"):
        lab.load_sae()
    with pytest.raises(ValueError, match="release and sae_id, or path"):
        lab.load_sae("release", "id", path="somewhere")


def test_missing_extra_names_the_extra(lab_for, monkeypatch, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    monkeypatch.setitem(sys.modules, "sae_lens", None)
    with pytest.raises(ImportError, match=r"oumi-weights-lab\[saes\]"):
        lab.load_sae(path=tmp_path / "sae")


# Reading


ARCHITECTURES = [
    ("standard", {}),
    ("jumprelu", {}),
    ("topk", {"k": 5}),
    ("topk", {"k": 5, "rescale_acts_by_decoder_norm": True}),
    ("gated", {}),
]


@pytest.mark.parametrize("architecture,config", ARCHITECTURES)
@pytest.mark.parametrize("hook", ["blocks.1.hook_resid_post", "blocks.0.hook_mlp_out"])
def test_encode_decode_match_saelens(lab_for, tmp_path, architecture, config, hook):
    lab = lab_for("gpt2")
    reference = make_sae(
        tmp_path / "sae",
        hook,
        architecture=architecture,
        kwargs={"center_writing_weights": True},
        **config,
    )
    sae = lab.load_sae(path=tmp_path / "sae")
    assert sae.site.center
    raw = record(lab, sae.module, sae.at) + 3.0
    transformed = sae.transform(raw)
    assert torch.allclose(transformed.mean(-1), torch.zeros(1, 5), atol=1e-5)
    features = sae.encode(raw)
    # SAELens subtracts b_dec on the way in and adds it on the way out; neither happens twice.
    with torch.no_grad():
        assert torch.equal(features, reference.encode(transformed))
        assert torch.equal(sae.decode(features), reference.decode(features))
        assert torch.allclose(sae.decode(features), reference(transformed), atol=1e-6)
    assert features.shape == (1, 5, 24)


def test_encode_decode_on_llama_gqa(lab_for, tmp_path):
    lab = lab_for("llama_gqa")
    reference = make_sae(tmp_path / "sae", "blocks.1.hook_resid_pre", architecture="jumprelu")
    sae = lab.load_sae(path=tmp_path / "sae")
    raw = record(lab, sae.module, sae.at)
    assert not sae.site.center
    with torch.no_grad():
        assert torch.equal(sae.decode(sae.encode(raw)), reference(raw))


def test_hook_z_sae_reads_flat_head_outputs(lab_for, tmp_path):
    lab = lab_for("llama_gqa")
    reference = make_sae(
        tmp_path / "sae", "blocks.0.attn.hook_z", reshape_activations="hook_z", d_sae=12
    )
    sae = lab.load_sae(path=tmp_path / "sae")
    raw = record(lab, sae.module, sae.at)
    assert raw.shape == (1, 5, D)
    reference.turn_off_forward_pass_hook_z_reshaping()
    with torch.no_grad():
        assert torch.equal(sae.decode(sae.encode(raw)), reference(raw))


def test_features_shapes_and_error_statistics(lab_for, tmp_path):
    lab = lab_for("gpt2")
    reference = make_sae(
        tmp_path / "sae", "blocks.1.hook_resid_post", kwargs={"center_writing_weights": True}
    )
    sae = lab.load_sae(path=tmp_path / "sae")
    result = sae.features("hello world yes no hello", positions="all")
    assert result["features"].shape == (5, 24)
    assert result["sequence"].tolist() == [0] * 5
    assert result["position"].tolist() == [0, 1, 2, 3, 4]
    raw = record(lab, sae.module, sae.at)[0]
    target = raw - raw.mean(-1, keepdim=True)
    with torch.no_grad():
        features = reference.encode(target)
        error = target - reference.decode(features)
    assert torch.allclose(result["features"], features, atol=1e-6)
    assert result["mse"] == pytest.approx(float(error.pow(2).mean()), rel=1e-4)
    spread = (target - target.mean(0)).pow(2).sum()
    assert result["fvu"] == pytest.approx(float(error.pow(2).sum() / spread), rel=1e-4)
    assert result["l0"] == pytest.approx(float((features != 0).sum(-1).float().mean()), rel=1e-6)
    assert result["rows"] == 5


def test_features_default_to_the_last_token(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    result = sae.features(["hello world", "yes no hello"])
    assert result["features"].shape == (2, 24)
    assert result["sequence"].tolist() == [0, 1]
    assert result["position"].tolist() == [1, 2]


def test_features_count_one_pass_per_sequence(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    sae.features(["hello world", "yes no", "hello"])
    assert lab.forward_passes == 3
    batch = {"input_ids": torch.tensor([[2, 3, 4], [3, 4, 5]])}
    sae.features(batch, positions="all")
    assert lab.forward_passes == 5


def test_features_accept_a_padded_batch_and_skip_padding(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    batch = {
        "input_ids": torch.tensor([[2, 3, 4], [1, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1], [0, 1, 1]]),
    }
    result = sae.features(batch, positions="all")
    assert result["sequence"].tolist() == [0, 0, 0, 1, 1]
    assert result["position"].tolist() == [0, 1, 2, 1, 2]


def test_features_restore_training_mode_and_leave_no_hooks(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    lab.model.train()
    sae.features("hello world")
    assert lab.model.training
    lab.model.eval()
    with pytest.raises(ValueError):
        sae.features("hello world", positions=40)
    assert not lab.model.training
    assert not lab.model.transformer.h[1]._forward_hooks


def test_text_gets_a_bos_token_when_the_sae_was_trained_with_one(lab_for, tmp_path):
    lab = lab_for("gpt2")
    lab.tokenizer.bos_token = "<eos>"
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    sae.prepend_bos = True
    result = sae.features("hello world", positions="all")
    assert result["position"].tolist() == [0, 1, 2]
    assert result["rows"] == 3


# Patching and steering


@pytest.mark.parametrize("architecture,config", ARCHITECTURES)
def test_patch_without_changes_leaves_logits_bitwise_equal(lab_for, tmp_path, architecture, config):
    lab = lab_for("llama_gqa")
    make_sae(tmp_path / "sae", "blocks.0.hook_resid_post", architecture=architecture, **config)
    sae = lab.load_sae(path=tmp_path / "sae")
    baseline = run(lab)
    with sae.patch({}, positions="all"):
        assert torch.equal(run(lab), baseline)
    with sae.patch({3: lambda value: value}, positions="all"):
        assert torch.equal(run(lab), baseline)
    # Setting a feature to the value it already has changes nothing either.
    current = sae.encode(record(lab, sae.module, sae.at, positions=[-1]))[0, 0]
    same = {3: float(current[3]), 4: float(current[4])}
    with sae.patch(same, positions=[-1]):
        assert torch.equal(run(lab), baseline)
    assert torch.equal(run(lab), baseline)


@pytest.mark.parametrize("architecture,config", ARCHITECTURES)
def test_patch_changes_the_site_by_the_decoder_row_only(lab_for, tmp_path, architecture, config):
    lab = lab_for("gpt2")
    reference = make_sae(
        tmp_path / "sae",
        "blocks.1.hook_resid_post",
        architecture=architecture,
        kwargs={"center_writing_weights": True},
        **config,
    )
    sae = lab.load_sae(path=tmp_path / "sae")
    before = record(lab, sae.module, sae.at)
    features = sae.encode(before)
    index = 5
    row = sae.decoder_rows([index])[0]
    with sae.patch({index: lambda value: value + 2.0}, positions="all"):
        after = record(lab, sae.module, sae.at)
    expected = before + 2.0 * row
    assert torch.allclose(after, expected, atol=1e-5)
    other = torch.ones(24, dtype=torch.bool)
    other[index] = False
    # Everything else about the activation, including the reconstruction error, is kept.
    delta = (after - before)[0]
    assert torch.allclose(delta, 2.0 * row.expand_as(delta), atol=1e-5)
    # The decoder row is SAELens's own.
    expected_row = reference.W_dec[index].detach()
    if config.get("rescale_acts_by_decoder_norm"):
        expected_row = expected_row / expected_row.norm()
    assert torch.allclose(row, expected_row, atol=1e-6)
    assert features.shape == (1, 5, 24)


def test_patch_sets_a_value_and_respects_positions(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    before = record(lab, sae.module, sae.at)
    old = sae.encode(before)[..., 7]
    with sae.patch({7: 10.0}, positions=[1, 3]):
        after = record(lab, sae.module, sae.at)
    row = sae.decoder_rows([7])[0]
    for position in range(5):
        if position in (1, 3):
            expected = before[0, position] + (10.0 - old[0, position]) * row
        else:
            expected = before[0, position]
        assert torch.allclose(after[0, position], expected, atol=1e-5)


def test_patch_validates_before_registering_hooks(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    for bad in [{24: 1.0}, {-1: 1.0}, {1.5: 1.0}, {True: 1.0}, {1: "high"}, [1, 2]]:
        with pytest.raises(ValueError), sae.patch(bad):
            pass
    assert not lab.model.transformer.h[1]._forward_hooks
    assert lab.patches() == []


def test_patch_and_steer_remove_hooks_after_an_error(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    baseline = run(lab)
    with pytest.raises(RuntimeError), sae.patch({1: 5.0}):
        raise RuntimeError("caller failed")
    with pytest.raises(RuntimeError), sae.steer({1: 5.0}):
        raise RuntimeError("caller failed")
    with pytest.raises(ValueError), sae.patch({1: 5.0}, positions=40):
        run(lab)
    assert not lab.model.transformer.h[1]._forward_hooks
    assert not lab.model.transformer.h[1]._forward_pre_hooks
    assert lab.patches() == []
    assert torch.equal(run(lab), baseline)


def test_steer_adds_decoder_rows(lab_for, tmp_path):
    lab = lab_for("llama_gqa")
    make_sae(tmp_path / "sae", "blocks.0.hook_resid_post", architecture="jumprelu")
    sae = lab.load_sae(path=tmp_path / "sae")
    before = record(lab, sae.module, sae.at)
    with sae.steer({2: 3.0, 9: -1.5}, positions="all"):
        after = record(lab, sae.module, sae.at)
    rows = sae.decoder_rows([2, 9])
    assert torch.allclose(after - before, 3.0 * rows[0] - 1.5 * rows[1], atol=1e-5)
    with sae.steer({2: 3.0}):
        last = record(lab, sae.module, sae.at, positions="all")
    assert torch.equal(last[:, :-1], before[:, :-1])
    assert torch.allclose(last[:, -1] - before[:, -1], 3.0 * rows[0], atol=1e-5)
    with pytest.raises(ValueError):
        with sae.steer({99: 1.0}):
            pass


def test_topk_steering_rows_have_unit_norm_only_when_rescaled(lab_for, tmp_path):
    lab = lab_for("llama_gqa")
    make_sae(tmp_path / "plain", "blocks.0.hook_resid_post", architecture="topk", k=3)
    make_sae(
        tmp_path / "unit",
        "blocks.0.hook_resid_post",
        architecture="topk",
        k=3,
        rescale_acts_by_decoder_norm=True,
    )
    plain = lab.load_sae(path=tmp_path / "plain").decoder_rows(range(24)).norm(dim=-1)
    unit = lab.load_sae(path=tmp_path / "unit").decoder_rows(range(24)).norm(dim=-1)
    assert not torch.allclose(plain, torch.ones(24), atol=1e-3)
    assert torch.allclose(unit, torch.ones(24), atol=1e-5)


def test_normalized_sites_can_be_read_but_not_edited(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.ln2.hook_normalized")
    sae = lab.load_sae(path=tmp_path / "sae")
    result = sae.features("hello world yes", positions="all")
    assert result["features"].shape == (3, 24)
    with pytest.raises(ValueError, match="normalized"), sae.patch({1: 1.0}):
        pass
    with pytest.raises(ValueError, match="normalized"), sae.steer({1: 1.0}):
        pass


# Transcoders


@pytest.mark.parametrize("architecture", ["transcoder", "skip_transcoder"])
def test_transcoder_reads_mlp_input_and_predicts_mlp_output(lab_for, tmp_path, architecture):
    lab = lab_for("gpt2")
    reference = make_sae(
        tmp_path / "tc",
        "blocks.1.ln2.hook_normalized",
        architecture=architecture,
        hook_name_out="blocks.1.hook_mlp_out",
        d_out=D,
    )
    sae = lab.load_sae(path=tmp_path / "tc")
    assert sae.kind == "transcoder"
    assert (sae.module, sae.at) == ("transformer.h.1.ln_2", "input")
    assert (sae.output_module, sae.output_at) == ("transformer.h.1.mlp", "output")
    result = sae.features("hello world yes no", positions="all")
    inputs = record(lab, sae.module, sae.at, text="hello world yes no")[0]
    target = record(lab, sae.output_module, sae.output_at, text="hello world yes no")[0]
    normalized = sae.transform(inputs)
    with torch.no_grad():
        prediction = reference(normalized)
    error = target - prediction
    assert result["mse"] == pytest.approx(float(error.pow(2).mean()), rel=1e-4)
    assert result["features"].shape == (4, 24)


def test_transcoder_patch_changes_the_mlp_output(lab_for, tmp_path):
    lab = lab_for("llama_gqa")
    make_sae(
        tmp_path / "tc",
        "blocks.1.ln2.hook_normalized",
        architecture="transcoder",
        hook_name_out="blocks.1.hook_mlp_out",
        d_out=D,
    )
    sae = lab.load_sae(path=tmp_path / "tc")
    baseline = run(lab)
    with sae.patch({}, positions="all"):
        assert torch.equal(run(lab), baseline)
    before = record(lab, sae.output_module, sae.output_at)
    inputs = record(lab, sae.module, sae.at)
    old = sae.encode(inputs)[..., 4]
    with sae.patch({4: lambda value: value + 1.5}, positions="all"):
        after = record(lab, sae.output_module, sae.output_at)
        assert not torch.equal(run(lab), baseline)
    row = sae.decoder_rows([4])[0]
    assert torch.allclose(after - before, 1.5 * row.expand_as(after), atol=1e-5)
    assert old.shape == (1, 5)
    with sae.steer({4: 2.0}, positions="all"):
        steered = record(lab, sae.output_module, sae.output_at)
    assert torch.allclose(steered - before, 2.0 * row.expand_as(steered), atol=1e-5)
    assert torch.equal(run(lab), baseline)
    assert not lab.model.model.layers[1].mlp._forward_hooks
    assert not lab.model.model.layers[1].post_attention_layernorm._forward_pre_hooks


def test_transcoder_needs_an_output_hook(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "tc", "blocks.1.ln2.hook_normalized", architecture="transcoder")
    with pytest.raises(ValueError, match="output hook"):
        lab.load_sae(path=tmp_path / "tc")


def test_transcoder_output_width_is_checked(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(
        tmp_path / "tc",
        "blocks.1.ln2.hook_normalized",
        architecture="transcoder",
        hook_name_out="blocks.1.hook_mlp_out",
        d_out=8,
    )
    with pytest.raises(ValueError, match="d_out 8"):
        lab.load_sae(path=tmp_path / "tc")


# Labels


def test_labels_are_none_offline_and_nothing_is_raised(lab_for, tmp_path, monkeypatch):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    sae.neuronpedia_id = "gpt2-small/1-res-jb"

    def offline(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(saes, "_fetch_label", offline)
    assert sae.labels([1, 2]) == {1: None, 2: None}


def test_labels_without_a_neuronpedia_id_make_no_request(lab_for, tmp_path, monkeypatch):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    assert sae.neuronpedia_id is None

    def fail(*args, **kwargs):
        raise AssertionError("no request expected")

    monkeypatch.setattr(saes, "_fetch_label", fail)
    assert sae.labels([3]) == {3: None}


def test_labels_are_cached_in_the_session(lab_for, tmp_path, monkeypatch):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    sae.neuronpedia_id = "gpt2-small/1-res-jb"
    calls = []

    def fetch(neuronpedia_id, index, timeout):
        calls.append((neuronpedia_id, index, timeout))
        return f"feature {index}" if index != 2 else None

    monkeypatch.setattr(saes, "_fetch_label", fetch)
    assert sae.labels([1, 2], timeout=3.0) == {1: "feature 1", 2: None}
    assert calls == [("gpt2-small/1-res-jb", 1, 3.0), ("gpt2-small/1-res-jb", 2, 3.0)]
    # Found labels are cached, and a miss is remembered for the session.
    assert sae.labels([1, 2]) == {1: "feature 1", 2: None}
    assert [call[1] for call in calls] == [1, 2]
    cached = list((lab.artifacts / "saes").glob("*.json"))
    assert len(cached) == 1
    assert json.loads(cached[0].read_text()) == {"1": "feature 1"}
    # A new LabSAE for the same id reads the cache.
    again = lab.load_sae(path=tmp_path / "sae")
    again.neuronpedia_id = "gpt2-small/1-res-jb"
    monkeypatch.setattr(saes, "_fetch_label", lambda *args: (_ for _ in ()).throw(OSError("off")))
    assert again.labels([1]) == {1: "feature 1"}


def test_label_text_comes_from_the_explanations_list():
    assert (
        saes._label_from({"explanations": [{"description": " dogs "}, {"description": "x"}]})
        == "dogs"
    )
    assert saes._label_from({"explanations": []}) is None
    assert saes._label_from({}) is None
    assert saes._label_from([1]) is None


def test_residual_between_attention_and_mlp_can_be_read_but_not_edited(lab_for, tmp_path):
    lab = lab_for("llama_gqa")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_mid")
    sae = lab.load_sae(path=tmp_path / "sae")
    assert sae.features("hello world yes", positions="all")["features"].shape == (3, 24)
    with pytest.raises(ValueError, match="skip connection"), sae.patch({1: 1.0}):
        pass


# Review fixes


def test_features_check_the_pass_cap_before_running(lab_for, tmp_path):
    from oumi_weights_lab.budget import BudgetExceeded

    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    lab.budget.limit = 2
    with pytest.raises(BudgetExceeded):
        sae.features(["hello world", "yes no", "hello"])
    assert lab.forward_passes == 0 and lab.budget.used == 0
    with pytest.raises(BudgetExceeded):
        sae.features({"input_ids": torch.tensor([[2, 3], [3, 4], [4, 5]])})
    assert lab.forward_passes == 0
    sae.features(["hello world", "yes no"])
    assert lab.forward_passes == 2
    lab.budget.limit = None
    with lab.limit_passes(1), pytest.raises(BudgetExceeded):
        sae.features(["hello", "world"])
    assert lab.forward_passes == 2


def build_olmo2():
    from transformers import AutoModelForCausalLM, Olmo2Config

    config = Olmo2Config(
        vocab_size=6,
        hidden_size=D,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=1,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()


def test_post_norm_blocks_without_a_pre_mlp_norm_are_refused(make_lab, tiny_tokenizer):
    lab = make_lab(build_olmo2(), tiny_tokenizer)
    for hook in ["hook_resid_mid", "hook_attn_out", "hook_mlp_out", "ln2.hook_normalized"]:
        with pytest.raises(ValueError, match="post-norm"):
            saes.hook_site(lab, f"blocks.0.{hook}")


def test_residual_hooks_work_at_linear_attention_layers(lab_for, tmp_path):
    lab = lab_for("qwen3_5")
    # Layer 0 is linear attention, layer 1 is full attention.
    assert saes.hook_site(lab, "blocks.0.hook_resid_pre").module == "model.layers.0"
    assert saes.hook_site(lab, "blocks.0.hook_resid_post").at == "output"
    assert saes.hook_site(lab, "blocks.0.hook_mlp_out").module == "model.layers.0.mlp"
    assert saes.hook_site(lab, "blocks.0.hook_resid_mid").module.startswith("model.layers.0.")
    assert saes.hook_site(lab, "blocks.1.hook_attn_out").module == "model.layers.1.self_attn"
    for hook in ["hook_attn_out", "attn.hook_z"]:
        with pytest.raises(ValueError, match="layer 0 has no full-attention block"):
            saes.hook_site(lab, f"blocks.0.{hook}")
    with pytest.raises(ValueError, match="no layer 5"):
        saes.hook_site(lab, "blocks.5.hook_resid_pre")
    make_sae(tmp_path / "sae", "blocks.0.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    assert sae.features("hello world yes", positions="all")["features"].shape == (3, 24)


def test_model_check_looks_up_the_path_before_lowercasing(lab_for, tmp_path, monkeypatch):
    import pathlib

    lab = lab_for("gpt2")
    local = tmp_path / "Gemma-2-2B"
    local.mkdir()
    lab.model_id = str(local)
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post", model_name="gemma-2-2b")
    real = pathlib.Path.exists

    def case_sensitive(self):
        # Only the exact spelling of the local directory exists.
        return str(self) == str(local) or (real(self) and str(self) != str(local).lower())

    monkeypatch.setattr(pathlib.Path, "exists", case_sensitive)
    with pytest.warns(UserWarning, match="local path"):
        sae = lab.load_sae(path=tmp_path / "sae")
    assert "local path" in sae.model_check


def test_a_failing_feature_does_not_stop_the_other_labels(lab_for, tmp_path, monkeypatch):
    import urllib.error

    lab = lab_for("gpt2")
    make_sae(tmp_path / "sae", "blocks.1.hook_resid_post")
    sae = lab.load_sae(path=tmp_path / "sae")
    sae.neuronpedia_id = "gpt2-small/1-res-jb"
    seen = []

    def fetch(neuronpedia_id, index, timeout):
        seen.append(index)
        if index in (2, 3):
            raise urllib.error.HTTPError("url", 404 if index == 2 else 429, "no", {}, None)
        if index == 5:
            raise urllib.error.URLError("no route")
        return f"feature {index}"

    monkeypatch.setattr(saes, "_fetch_label", fetch)
    found = sae.labels([1, 2, 3, 4, 5, 6])
    assert found == {1: "feature 1", 2: None, 3: None, 4: "feature 4", 5: None, 6: None}
    # A network-level failure stops the rest.
    assert seen == [1, 2, 3, 4, 5]


# Hook name override


def test_hook_name_override_changes_the_site_and_is_checked(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(
        tmp_path / "tc",
        "blocks.1.hook_mlp_in",
        architecture="transcoder",
        hook_name_out="blocks.1.hook_mlp_out",
        d_out=D,
    )
    recorded = lab.load_sae(path=tmp_path / "tc")
    assert (recorded.module, recorded.at) == ("transformer.h.1.ln_2", "input")
    assert recorded.hook_name == recorded.recorded_hook_name == "blocks.1.hook_mlp_in"
    assert "recorded" not in repr(recorded)
    fixed = lab.load_sae(path=tmp_path / "tc", hook_name="blocks.1.mlp.hook_in")
    assert (fixed.module, fixed.at) == ("transformer.h.1.mlp", "input")
    assert fixed.hook_name == "blocks.1.mlp.hook_in"
    assert fixed.recorded_hook_name == "blocks.1.hook_mlp_in"
    assert "blocks.1.mlp.hook_in (recorded blocks.1.hook_mlp_in)" in repr(fixed)
    # The output site still comes from the recorded output hook.
    assert (fixed.output_module, fixed.output_at) == ("transformer.h.1.mlp", "output")
    for bad, message in [("nonsense", "not mapped"), ("blocks.7.hook_resid_pre", "layer 7")]:
        with pytest.raises(ValueError, match=message):
            lab.load_sae(path=tmp_path / "tc", hook_name=bad)
    make_sae(tmp_path / "narrow", "blocks.1.hook_resid_pre", d_in=8)
    with pytest.raises(ValueError, match="blocks.1.hook_resid_post has width 16.*d_in 8"):
        lab.load_sae(path=tmp_path / "narrow", hook_name="blocks.1.hook_resid_post")
