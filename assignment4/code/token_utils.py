from __future__ import annotations

import torch


def setup_custom_token(tokenizer, text_encoder, instance_token: str, initializer: str) -> int:
    """Add `instance_token` and initialize its embedding from existing tokens."""
    num_added = tokenizer.add_tokens([instance_token])
    token_id = tokenizer.convert_tokens_to_ids(instance_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Could not add or resolve instance token: {instance_token}")

    tokenized = tokenizer(instance_token, add_special_tokens=False).input_ids
    if len(tokenized) != 1:
        raise ValueError(f"{instance_token!r} must tokenize to one id after insertion, got {tokenized}")

    if num_added:
        text_encoder.resize_token_embeddings(len(tokenizer))

    initializer_ids = tokenizer(initializer, add_special_tokens=False).input_ids
    initializer_ids = [idx for idx in initializer_ids if idx != token_id]
    if not initializer_ids:
        raise ValueError("--token_initializer must tokenize to at least one existing token")

    embedding = text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        initializer_tensor = torch.tensor(initializer_ids, device=embedding.device)
        embedding[token_id].copy_(embedding[initializer_tensor].mean(dim=0))

    return token_id


def assert_custom_token_ready(tokenizer, text_encoder, instance_token: str, token_id: int) -> None:
    tokenized = tokenizer(instance_token, add_special_tokens=False).input_ids
    if tokenized != [token_id]:
        raise AssertionError(f"{instance_token!r} tokenized as {tokenized}, expected [{token_id}]")

    embedding = text_encoder.get_input_embeddings().weight
    if token_id >= embedding.shape[0]:
        raise AssertionError(f"Token id {token_id} is outside embedding table with shape {tuple(embedding.shape)}")
