"""Basic smoke tests for ik-llama-cpp-python."""

import pytest


def test_import():
    from ik_llama_cpp import IkLlama
    assert IkLlama is not None


def test_version():
    from ik_llama_cpp import __version__
    assert __version__ == "0.1.5"


@pytest.mark.skipif(
    not __import__("os").environ.get("IK_LLAMA_TEST_MODEL"),
    reason="Set IK_LLAMA_TEST_MODEL to a .gguf path to run inference tests",
)
def test_chat():
    import os
    from ik_llama_cpp import IkLlama

    model_path = os.environ["IK_LLAMA_TEST_MODEL"]
    llm = IkLlama(model_path, n_ctx=512, verbose=False)
    text = llm.chat("What is 2+2? Answer in one word.", max_tokens=16)
    assert len(text) > 0
    llm.close()


@pytest.mark.skipif(
    not __import__("os").environ.get("IK_LLAMA_EMBED_TEST_MODEL"),
    reason="Set IK_LLAMA_EMBED_TEST_MODEL to an embedding GGUF path",
)
def test_repeated_batched_embeddings_are_stable():
    import math
    import os
    from ik_llama_cpp import IkLlama

    llm = IkLlama(
        os.environ["IK_LLAMA_EMBED_TEST_MODEL"],
        n_ctx=512,
        n_threads=8,
        embedding=True,
        verbose=False,
    )
    texts = ["hello world", "vector search", "中文向量检索"]
    first = llm.embed(texts)
    second = llm.embed(texts)
    llm.close()

    assert len(first) == len(texts)
    assert all(len(vector) > 0 for vector in first)
    for left, right in zip(first, second):
        similarity = sum(a * b for a, b in zip(left, right)) / (
            math.sqrt(sum(a * a for a in left))
            * math.sqrt(sum(b * b for b in right))
        )
        assert similarity > 0.99999
