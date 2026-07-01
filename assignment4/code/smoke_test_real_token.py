#!/usr/bin/env python

from __future__ import annotations

import torch
from transformers import AutoTokenizer, CLIPTextModel

from config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_INSTANCE_TOKEN,
    DEFAULT_TOKEN_INITIALIZER,
)

from token_utils import setup_custom_token


def load_tokenizer_and_text_encoder(model_name: str):
    """Load the same tokenizer and CLIP text encoder used by training."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
        use_fast=False,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
    )

    return tokenizer, text_encoder


def get_embedding_weight(text_encoder):
    """Return the token embedding matrix from the CLIP text encoder."""
    return text_encoder.get_input_embeddings().weight


def print_before_state(tokenizer, text_encoder):
    """Print tokenizer and embedding state before adding the custom token."""
    embedding = get_embedding_weight(text_encoder)

    print("\nBefore adding token")
    print("-------------------")
    print(f"Tokenizer vocab size:      {len(tokenizer)}")
    print(f"Text encoder embeddings:   {tuple(embedding.shape)}")


def print_after_state(tokenizer, text_encoder, token_id):
    """Print tokenizer and embedding state after adding the custom token."""
    embedding = get_embedding_weight(text_encoder)
    tokenized = tokenizer(DEFAULT_INSTANCE_TOKEN, add_special_tokens=False).input_ids

    print("\nAfter adding token")
    print("------------------")
    print(f"Instance token:            {DEFAULT_INSTANCE_TOKEN}")
    print(f"Token ID:                  {token_id}")
    print(f"Tokenized form:            {tokenized}")
    print(f"Tokenizer vocab size:      {len(tokenizer)}")
    print(f"Text encoder embeddings:   {tuple(embedding.shape)}")


def assert_token_was_added(tokenizer, old_vocab_size, token_id):
    """Verify the custom token was added as a new vocabulary entry."""
    assert len(tokenizer) == old_vocab_size + 1, (
        f"Expected vocab size {old_vocab_size + 1}, got {len(tokenizer)}"
    )

    assert token_id == old_vocab_size, (
        f"Expected new token ID {old_vocab_size}, got {token_id}"
    )


def assert_token_is_single_id(tokenizer, token_id):
    """Verify the custom token does not split into multiple IDs."""
    tokenized = tokenizer(DEFAULT_INSTANCE_TOKEN, add_special_tokens=False).input_ids

    assert tokenized == [token_id], (
        f"Expected {DEFAULT_INSTANCE_TOKEN!r} to tokenize to [{token_id}], got {tokenized}"
    )


def assert_embedding_table_was_resized(text_encoder, old_embedding_shape):
    """Verify the CLIP embedding matrix gained exactly one row."""
    new_embedding_shape = get_embedding_weight(text_encoder).shape

    expected_shape = torch.Size(
        [old_embedding_shape[0] + 1, old_embedding_shape[1]]
    )

    assert new_embedding_shape == expected_shape, (
        f"Expected embedding shape {expected_shape}, got {new_embedding_shape}"
    )


def assert_initializer_embedding_was_copied(tokenizer, text_encoder, token_id):
    """
    Verify the new token embedding equals the mean embedding of DEFAULT_TOKEN_INITIALIZER.

    This checks the final copy_() step inside setup_custom_token().
    """
    embedding = get_embedding_weight(text_encoder)

    initializer_ids = tokenizer(
        DEFAULT_TOKEN_INITIALIZER,
        add_special_tokens=False,
    ).input_ids

    initializer_ids = [idx for idx in initializer_ids if idx != token_id]

    expected_embedding = embedding[initializer_ids].mean(dim=0)
    actual_embedding = embedding[token_id]

    assert torch.allclose(actual_embedding, expected_embedding, atol=1e-6), (
        "Custom token embedding was not initialized from the token initializer"
    )


def print_step(number: int, message: str) -> None:
    print(f"\nStep {number}: {message}")
    print("-" * (len(f"Step {number}: {message}")))

 
def main():
    print("===============================")
    print("Running custom token smoke test")
    print("===============================")

    print_step(1, "Load the real SD 1.5 tokenizer and CLIP text encoder")
    tokenizer, text_encoder = load_tokenizer_and_text_encoder(DEFAULT_MODEL_NAME)

    old_vocab_size = len(tokenizer)
    old_embedding_shape = get_embedding_weight(text_encoder).shape

    print_before_state(tokenizer, text_encoder)

    print_step(2, "Add the custom instance token using setup_custom_token()")
    token_id = setup_custom_token(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        instance_token=DEFAULT_INSTANCE_TOKEN,
        initializer=DEFAULT_TOKEN_INITIALIZER,
    )

    print_after_state(tokenizer, text_encoder, token_id)

    print_step(3, "Verify the token was added to the tokenizer")
    assert_token_was_added(tokenizer, old_vocab_size, token_id)
    print("OK: tokenizer vocab size increased by 1")
    print(f"OK: new token ID is {token_id}")

    print_step(4, "Verify the token resolves to exactly one token ID")
    assert_token_is_single_id(tokenizer, token_id)
    print(f"OK: {DEFAULT_INSTANCE_TOKEN!r} tokenizes to [{token_id}]")

    print_step(5, "Verify the text encoder embedding table was resized")
    assert_embedding_table_was_resized(text_encoder, old_embedding_shape)
    print("OK: text encoder embedding table gained one row")

    print_step(6, "Verify the new embedding was initialized from the initializer token")
    assert_initializer_embedding_was_copied(tokenizer, text_encoder, token_id)
    print(f"OK: custom token embedding was initialized from {DEFAULT_TOKEN_INITIALIZER!r}")

    print("\nResult")
    print("------")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()