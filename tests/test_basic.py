"""Basic smoke tests for ik-llama-cpp-python."""

import pytest


def test_import():
    from ik_llama_cpp import IkLlama
    assert IkLlama is not None


def test_version():
    from ik_llama_cpp import __version__
    assert __version__ == "0.1.3"


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
