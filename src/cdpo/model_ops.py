from torch.nn import Dropout
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = 'left'

    return tokenizer


def get_partially_trainable_model(model_name="microsoft/phi-1_5",
                                  tokenizer_name=None,
                                  n_layers_freeze=0, dropout=0.0):

    tokenizer = get_tokenizer(tokenizer_name or model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if model_name == "microsoft/phi-1_5":
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False

        for layer_idx in range(n_layers_freeze):
            for param in model.model.layers[layer_idx].parameters():
                param.requires_grad = False
    elif model_name.startswith('openai-community/gpt2'):
        if n_layers_freeze > 0:
            for param in model.transformer.wte.parameters():
                param.requires_grad = False
            for param in model.transformer.wpe.parameters():
                param.requires_grad = False

            for layer_idx in range(n_layers_freeze):
                for param in model.transformer.h[layer_idx].parameters():
                    param.requires_grad = False
    else:
        if n_layers_freeze > 0:
            raise ValueError(f"model name {model_name} unknown!")

    # Set the dropout value
    for module in model.modules():
        if isinstance(module, Dropout):
            module.p = dropout

    return model, tokenizer
