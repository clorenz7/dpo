from cdpo import model_ops


def test_get_model():
    model, tokenizer = model_ops.get_partially_trainable_model(
        n_layers_freeze=23, dropout=0.02
    )

    P = list(model.model.embed_tokens.parameters())
    assert P[0].requires_grad is False
    model.model.layers

    data = list(model.model.layers[0].parameters())
    assert data[0].requires_grad is False

    data = list(model.model.layers[22].parameters())
    assert data[0].requires_grad is False

    data = list(model.model.layers[23].parameters())
    assert data[0].requires_grad
