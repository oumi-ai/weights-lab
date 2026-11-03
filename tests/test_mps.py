import pytest
import torch


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
