"""Unit tests that don't require a model file."""

from ik_llama_cpp.llama import IkLlama, _SPECIAL_TOKEN_RE, _cpu_has_avx_vnni


# Access the static method directly
_apply_chat_template = IkLlama._apply_chat_template


def test_import():
    from ik_llama_cpp import IkLlama
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


def test_quantize_exports():
    from ik_llama_cpp import find_quantize_bin, quantize, quantize_from_hf
    assert callable(find_quantize_bin)
    assert callable(quantize)
    assert callable(quantize_from_hf)
