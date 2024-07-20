import pytest

from cdpo import model_ops


@pytest.mark.skip(reason="This test is being skipped for speed")
def test_get_model():
    model, tokenizer = model_ops.get_partially_trainable_model(
        n_layers_freeze=23, dropout=0.02
    )

    P = list(model.model.embed_tokens.parameters())
    assert P[0].requires_grad is False

    data = list(model.model.layers[0].parameters())
    assert data[0].requires_grad is False

    data = list(model.model.layers[22].parameters())
    assert data[0].requires_grad is False

    data = list(model.model.layers[23].parameters())
    assert data[0].requires_grad


def test_get_gpt2():
    model, tokenizer = model_ops.get_partially_trainable_model(
        "openai-community/gpt2",
        n_layers_freeze=1, dropout=0.02
    )

    P = list(model.transformer.wte.parameters())
    assert P[0].requires_grad is False

    P = list(model.transformer.wpe.parameters())
    assert P[0].requires_grad is False

    P = list(model.transformer.h[1].parameters())
    assert P[2].requires_grad

    assert model.transformer.drop.p == 0.02
