"""High-level IkLlama class — drop-in compatible with llama_cpp.Llama."""

from __future__ import annotations

import logging
import re
import struct
from typing import Any

from . import _ctypes_api as C
from ._internals import IkModel, IkContext, make_batch_range, make_batch_single

logger = logging.getLogger(__name__)

# Special token markers that may leak into generated text
_SPECIAL_TOKEN_RE = re.compile(
    r"<start_of_turn>|<end_of_turn>|<turn\|>|<\|tool_response\|?>|</s>"
)


def _cpu_has_avx_vnni() -> bool:
    """Detect AVX-VNNI support via CPUID (leaf 7, sub-leaf 1, EAX bit 4)."""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        return "avx_vnni" in flags or "avxvnni" in flags
    except ImportError:
        pass
    # Fallback: not detectable, assume absent
    return False


class IkLlama:
    """High-level wrapper for ik_llama.cpp inference.

    API designed to be compatible with ``llama_cpp.Llama`` so that
    ``litegraph.LlamaCppBackend`` can use it as a drop-in replacement.

    Usage::

        llm = IkLlama("model.gguf", n_ctx=4096)
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.3,
            max_tokens=256,
        )
        print(response["choices"][0]["message"]["content"])
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 4096,
        n_threads: int = 0,
        use_mmap: bool = True,
        use_mlock: bool = False,
        flash_attn: bool = True,
        n_gpu_layers: int = 0,
        verbose: bool = True,
    ):
        self._model = IkModel(
            model_path, use_mmap=use_mmap, use_mlock=use_mlock,
            n_gpu_layers=n_gpu_layers,
        )

        # Detect non-VNNI CPU — ik_llama.cpp flash attention
        # (iqk_fa_templates.h) triggers GGML_ASSERT(S > 0) on longer
        # prompts without AVX-VNNI, regardless of quant type.
        self._has_vnni = _cpu_has_avx_vnni()
        if flash_attn and not self._has_vnni:
            logger.warning(
                "AVX-VNNI not detected — disabling flash_attn to avoid "
                "ik_llama.cpp flash attention assert failures on longer prompts. "
                "For full ik_llama.cpp performance, use a Zen 4+ or Alder Lake+ CPU."
            )
            flash_attn = False

        self._context = IkContext(
            self._model, n_ctx=n_ctx, n_threads=n_threads,
            flash_attn=flash_attn,
        )
        self._n_ctx = n_ctx
        self._verbose = verbose

    @property
    def ctx(self):
        """Raw context pointer — for perf timing access."""
        return self._context.ctx

    def tokenize(self, text: str, *, add_bos: bool = True, special: bool = False) -> list[int]:
        return self._model.tokenize(text, add_bos=add_bos, special=special)

    def detokenize(self, tokens: list[int]) -> str:
        return self._model.detokenize(tokens)

    def generate(
        self,
        tokens: list[int],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> list[int]:
        """Generate tokens from a prompt token list. Returns generated token ids."""
        self._context.perf_reset()

        n_ubatch = self._context._n_ubatch
        n_tokens = len(tokens)

        # Prefill in n_ubatch-sized chunks to avoid compute buffer overflow
        for i in range(0, n_tokens, n_ubatch):
            chunk = tokens[i : i + n_ubatch]
            is_last_chunk = (i + n_ubatch >= n_tokens)
            batch = make_batch_range(chunk, pos_start=i, logits_last=is_last_chunk)
            ret = self._context.decode(batch)
            C.llama_batch_free(batch)
            if ret != 0:
                raise RuntimeError(
                    f"llama_decode failed during prefill (chunk {i}..{i+len(chunk)}, "
                    f"n_ubatch={n_ubatch}): {ret}"
                )

        generated: list[int] = []
        pos = len(tokens)

        for _ in range(max_tokens):
            token_id = self._context.sample(
                -1, temperature=temperature, top_k=top_k, top_p=top_p,
            )

            # EOG check using the model's own EOG token list
            if C.llama_token_is_eog(self._model.model, token_id):
                break

            generated.append(token_id)

            # Decode next token
            batch = make_batch_single(token_id, pos)
            ret = self._context.decode(batch)
            C.llama_batch_free(batch)
            if ret != 0:
                break
            pos += 1

        return generated

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 256,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> dict[str, Any]:
        """OpenAI-compatible chat completion.

        Returns a dict matching the ``llama_cpp.Llama.create_chat_completion``
        schema: choices[0].message.content, usage.prompt_tokens, etc.
        """
        prompt = self._apply_chat_template(messages)
        tokens = self.tokenize(prompt, add_bos=False, special=True)
        prompt_tokens = len(tokens)

        gen_ids = self.generate(
            tokens, max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p,
        )

        text = self.detokenize(gen_ids)
        # Strip special token markers that leak through sub-token generation
        text = _SPECIAL_TOKEN_RE.sub("", text).strip()
        completion_tokens = len(gen_ids)

        return {
            "id": "chatcmpl-ik",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def chat(self, prompt: str, *, temperature: float = 0.3,
             max_tokens: int = 256) -> str:
        """Convenience: single user message -> response text."""
        resp = self.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"]

    def close(self):
        if self._context:
            self._context.close()
            self._context = None
        if self._model:
            self._model.close()
            self._model = None

    def __del__(self):
        self.close()

    @staticmethod
    def _apply_chat_template(messages: list[dict[str, str]]) -> str:
        """Apply Gemma-style chat template.

        Format::

            <bos><start_of_turn>user
            {content}<end_of_turn>
            <start_of_turn>model
        """
        parts = ["<bos>"]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
            elif role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")
        parts.append("<start_of_turn>model\n")
        return "".join(parts)
