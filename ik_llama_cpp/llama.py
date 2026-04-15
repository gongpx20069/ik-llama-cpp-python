"""High-level IkLlama class — drop-in compatible with llama_cpp.Llama."""

from __future__ import annotations

import ctypes
import time
from typing import Any, Optional

from . import _ctypes_api as C
from ._internals import IkModel, IkContext, IkSampler, make_batch, make_batch_single


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
        self._context = IkContext(
            self._model, n_ctx=n_ctx, n_threads=n_threads,
            flash_attn=flash_attn,
        )
        self._n_ctx = n_ctx
        self._verbose = verbose

    @property
    def ctx(self):
        """Raw context pointer — for llama_perf_context() access."""
        return self._context.ctx

    def tokenize(self, text: str, *, add_bos: bool = True) -> list[int]:
        return self._model.tokenize(text, add_bos=add_bos)

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

        # Prefill
        batch = make_batch(tokens, logits_last=True)
        ret = self._context.decode(batch)
        C.llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed during prefill: {ret}")

        # Sampler
        sampler = IkSampler(temperature=temperature, top_k=top_k, top_p=top_p)

        generated: list[int] = []
        n_vocab = self._model.n_vocab
        pos = len(tokens)

        for _ in range(max_tokens):
            token_id = sampler.sample(self._context, -1)

            # EOS check (token id 1 and 106 for Gemma)
            if token_id == 1 or token_id == 106:
                break

            generated.append(token_id)

            # Decode next token
            batch = make_batch_single(token_id, pos)
            ret = self._context.decode(batch)
            C.llama_batch_free(batch)
            if ret != 0:
                break
            pos += 1

        sampler.close()
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

        Returns a dict matching the OpenAI ``ChatCompletion`` schema.
        """
        prompt = self._apply_chat_template(messages)
        tokens = self.tokenize(prompt, add_bos=False)
        prompt_tokens = len(tokens)

        gen_ids = self.generate(
            tokens, max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p,
        )

        text = self.detokenize(gen_ids)
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
        """Convenience: single user message → response text."""
        resp = self.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"]

    def chat_with_stats(self, prompt: str, *, temperature: float = 0.3,
                        max_tokens: int = 256) -> tuple[str, dict]:
        """Like :meth:`chat` but also returns perf stats.

        Returns ``(text, stats)`` where *stats* contains:
        ``prompt_tokens``, ``eval_tokens``, ``prompt_tps``, ``eval_tps``.
        """
        self._context.perf_reset()

        text = self.chat(prompt, temperature=temperature, max_tokens=max_tokens)

        perf = self._context.perf()
        prompt_tokens = perf["n_p_eval"]
        eval_tokens = perf["n_eval"]
        t_p_eval = perf["t_p_eval_ms"]
        t_eval = perf["t_eval_ms"]

        stats = {
            "prompt_tokens": prompt_tokens,
            "eval_tokens": eval_tokens,
            "prompt_tps": round(prompt_tokens / (t_p_eval / 1000), 2) if t_p_eval > 0 else 0.0,
            "eval_tps": round(eval_tokens / (t_eval / 1000), 2) if t_eval > 0 else 0.0,
        }
        return text, stats

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
