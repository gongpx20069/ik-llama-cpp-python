"""Unit tests that don't require a model file or the compiled C library.

These tests mock out the native library loading so they can run in CI
without building the C++ extension.
"""

import re
import sys
import types
from unittest import mock


def _make_mock_modules():
    """Create mock modules to replace the native library import chain."""
    mock_lib_loader = types.ModuleType("ik_llama_cpp._lib_loader")
    mock_lib_loader.load_shared_library = mock.MagicMock()

    mock_ctypes_api = types.ModuleType("ik_llama_cpp._ctypes_api")
    # Add commonly used attributes so downstream imports don't fail
    for name in [
        "llama_backend_init", "llama_backend_free",
        "llama_model_load_from_file", "llama_free_model",
        "llama_init_from_model", "llama_free",
        "llama_tokenize", "llama_token_to_piece",
        "llama_decode", "llama_batch_init", "llama_batch_free",
        "llama_sample_top_k", "llama_sample_top_p",
        "llama_sample_temp", "llama_sample_token_greedy",
        "llama_sample_token", "llama_get_logits_ith",
        "llama_kv_cache_clear", "llama_token_is_eog",
        "llama_get_timings", "llama_print_timings", "llama_reset_timings",
        "llama_model_default_params", "llama_context_default_params",
        "llama_model_desc", "llama_n_ctx",
    ]:
        setattr(mock_ctypes_api, name, mock.MagicMock())

    return mock_lib_loader, mock_ctypes_api


# Patch modules before importing ik_llama_cpp
_mock_lib_loader, _mock_ctypes_api = _make_mock_modules()
sys.modules.setdefault("ik_llama_cpp._lib_loader", _mock_lib_loader)
sys.modules.setdefault("ik_llama_cpp._ctypes_api", _mock_ctypes_api)

from ik_llama_cpp.llama import IkLlama, _SPECIAL_TOKEN_RE, _cpu_has_avx_vnni  # noqa: E402

_apply_chat_template = IkLlama._apply_chat_template


def test_import():
    assert IkLlama is not None


def test_version():
    from ik_llama_cpp import __version__
    assert __version__ == "0.1.3"


def test_chat_template_single_user():
    result = _apply_chat_template([{"role": "user", "content": "Hello"}])
    assert result == (
        "<bos>"
        "<start_of_turn>user\nHello<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def test_chat_template_system_user():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    result = _apply_chat_template(messages)
    assert "<bos>" in result
    assert "You are helpful." in result
    assert "Hi" in result
    assert result.endswith("<start_of_turn>model\n")


def test_chat_template_multi_turn():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    result = _apply_chat_template(messages)
    assert "<start_of_turn>user\nHi<end_of_turn>" in result
    assert "<start_of_turn>model\nHello!<end_of_turn>" in result
    assert "<start_of_turn>user\nHow are you?<end_of_turn>" in result
    assert result.endswith("<start_of_turn>model\n")


def test_special_token_regex():
    assert _SPECIAL_TOKEN_RE.sub("", "hello<end_of_turn>world") == "helloworld"
    assert _SPECIAL_TOKEN_RE.sub("", "<start_of_turn>test") == "test"
    assert _SPECIAL_TOKEN_RE.sub("", "clean text") == "clean text"
    assert _SPECIAL_TOKEN_RE.sub("", "foo</s>bar") == "foobar"


def test_cpu_has_avx_vnni_returns_bool():
    result = _cpu_has_avx_vnni()
    assert isinstance(result, bool)
