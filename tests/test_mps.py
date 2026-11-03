import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("fail", [False, True])
def test_sampled_evaluation_restores_mps_rng_and_repeats(tiny_lab, monkeypatch, device, fail):
    if not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS hardware")
    lab = tiny_lab
    lab.model.to(device)
    generate = lab.model.generate
    sequences = []

    def tracked_generate(**kwargs):
        result = generate(**kwargs)
        sequences.append(result.cpu().clone())
        if fail:
            raise RuntimeError("generation failed after sampling")
        return result

    monkeypatch.setattr(lab.model, "generate", tracked_generate)
    torch.manual_seed(89)
    cpu_before = torch.get_rng_state().clone()
    mps_before = torch.mps.get_rng_state().clone()
    for index in range(2):
        kwargs = {
            "name": f"sample-{index}",
            "mode": "sampled",
            "seed": 73,
            "generation_config": {
                "do_sample": True,
                "max_new_tokens": 4,
                "eos_token_id": None,
            },
        }
        if fail:
            with pytest.raises(RuntimeError, match="generation failed after sampling"):
                lab.evaluate([{"prompt": "hello", "answer": "world"}], **kwargs)
        else:
            lab.evaluate([{"prompt": "hello", "answer": "world"}], **kwargs)
        assert torch.equal(torch.get_rng_state(), cpu_before)
        assert torch.equal(torch.mps.get_rng_state(), mps_before)
    assert torch.equal(sequences[0], sequences[1])


def test_mps_inference_record_edit_undo_gradient_and_budget(tiny_lab):
    if not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS hardware")
    lab = tiny_lab
    lab.model.to("mps")
    lab.budget.limit = 10
    inputs = lab.tokenizer("hello world", return_tensors="pt").to(lab.device)
    with lab.record("transformer.h.0", positions=-1) as captured, torch.no_grad():
        baseline = lab.model(**inputs).logits.clone()
        assert torch.equal(baseline, lab.model(**inputs).logits)
    assert captured["transformer.h.0"][0].device.type == "cpu"
    name = "transformer.h.0.mlp.c_proj.weight"
    original = lab.model.get_parameter(name).detach().clone()
    edit_id = lab.zero(name)
    assert not lab.model.get_parameter(name).count_nonzero()
    lab.undo(edit_id)
    assert torch.equal(lab.model.get_parameter(name), original)
    with torch.no_grad():
        assert torch.equal(lab.model(**inputs).logits, baseline)
    gradients = lab.gradient("hello world", parameters=[name], max_tokens=4)
    assert gradients[name].device.type == "cpu"
    assert torch.isfinite(gradients[name]).all()
    assert lab.forward_passes == lab.budget.used == 4
