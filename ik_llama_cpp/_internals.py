"""RAII wrappers for ik_llama.cpp C objects."""

from __future__ import annotations

import ctypes
from typing import Optional

from . import _ctypes_api as C


class IkModel:
    """Wraps a llama_model pointer with automatic cleanup."""

    def __init__(self, path: str, *, use_mmap: bool = True, use_mlock: bool = False,
                 n_gpu_layers: int = 0):
        C.llama_backend_init()

        params = C.llama_model_default_params()
        params.use_mmap = use_mmap
        params.use_mlock = use_mlock
        params.n_gpu_layers = n_gpu_layers

        self._model = C.llama_model_load_from_file(path.encode("utf-8"), params)
        if not self._model:
            raise RuntimeError(f"Failed to load model: {path}")

        self._vocab = C.llama_model_get_vocab(self._model)
        self.n_vocab = C.llama_vocab_n_tokens(self._vocab)

    @property
    def model(self):
        return self._model

    @property
    def vocab(self):
        return self._vocab

    @property
    def desc(self) -> str:
        """Model description string (e.g. 'gemma4 2B IQ4_KT - 4.0 bpw')."""
        buf = ctypes.create_string_buffer(256)
        C.llama_model_desc(self._model, buf, 256)
        return buf.value.decode("utf-8", errors="replace")

    def tokenize(self, text: str, *, add_bos: bool = True, special: bool = False) -> list[int]:
        text_bytes = text.encode("utf-8")
        # First call to get required size (ik_llama.cpp takes model*, not vocab*)
        n = C.llama_tokenize(
            self._model, text_bytes, len(text_bytes),
            None, 0, add_bos, special,
        )
        n = abs(n)
        buf = (C.llama_token * n)()
        n_actual = C.llama_tokenize(
            self._model, text_bytes, len(text_bytes),
            buf, n, add_bos, special,
        )
        return list(buf[:n_actual])

    def detokenize(self, tokens: list[int], *, special: bool = False) -> str:
        pieces = []
        buf = ctypes.create_string_buffer(256)
        for tok in tokens:
            n = C.llama_token_to_piece(self._model, tok, buf, 256, 0, special)
            if n > 0:
                pieces.append(buf.value[:n].decode("utf-8", errors="replace"))
        return "".join(pieces)

    def close(self):
        if self._model:
            C.llama_free_model(self._model)
            self._model = None

    def __del__(self):
        self.close()


class IkContext:
    """Wraps a llama_context pointer with automatic cleanup."""

    def __init__(self, model: IkModel, *, n_ctx: int = 4096, n_threads: int = 0,
                 flash_attn: bool = True):
        params = C.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_ctx
        if n_threads > 0:
            params.n_threads = n_threads
            params.n_threads_batch = n_threads
        params.flash_attn = flash_attn

        self._ctx = C.llama_init_from_model(model.model, params)
        if not self._ctx:
            raise RuntimeError("Failed to create context")
        self._model = model
        self._n_ubatch = params.n_ubatch or 512

    @property
    def ctx(self):
        return self._ctx

    @property
    def model(self) -> IkModel:
        return self._model

    def decode(self, batch: C.llama_batch) -> int:
        return C.llama_decode(self._ctx, batch)

    def kv_cache_clear(self):
        C.llama_kv_cache_clear(self._ctx)

    def get_logits(self, idx: int = -1):
        return C.llama_get_logits_ith(self._ctx, idx)

    def perf(self) -> dict:
        data = C.llama_get_timings(self._ctx)
        return {
            "t_p_eval_ms": data.t_p_eval_ms,
            "t_eval_ms": data.t_eval_ms,
            "n_p_eval": data.n_p_eval,
            "n_eval": data.n_eval,
        }

    def perf_reset(self):
        C.llama_reset_timings(self._ctx)

    def sample(self, idx: int, *, temperature: float = 0.0,
               top_k: int = 40, top_p: float = 0.95) -> int:
        """Sample a token from logits at position idx using direct sampling."""
        n_vocab = self._model.n_vocab
        logits = C.llama_get_logits_ith(self._ctx, idx)

        # Build candidate array
        candidates_data = (C.llama_token_data * n_vocab)()
        for i in range(n_vocab):
            candidates_data[i].id = i
            candidates_data[i].logit = logits[i]
            candidates_data[i].p = 0.0

        candidates = C.llama_token_data_array()
        candidates.data = candidates_data
        candidates.size = n_vocab
        candidates.selected = -1
        candidates.sorted = False

        candidates_p = ctypes.pointer(candidates)

        if temperature <= 0:
            return C.llama_sample_token_greedy(self._ctx, candidates_p)
        else:
            C.llama_sample_top_k(self._ctx, candidates_p, top_k, 1)
            C.llama_sample_top_p(self._ctx, candidates_p, top_p, 1)
            C.llama_sample_temp(self._ctx, candidates_p, temperature)
            return C.llama_sample_token(self._ctx, candidates_p)

    def close(self):
        if self._ctx:
            C.llama_free(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()


def make_batch(tokens: list[int], *, logits_last: bool = True) -> C.llama_batch:
    """Create a llama_batch from a token list (positions start at 0)."""
    return make_batch_range(tokens, pos_start=0, logits_last=logits_last)


def make_batch_range(tokens: list[int], *, pos_start: int = 0,
                     logits_last: bool = True) -> C.llama_batch:
    """Create a llama_batch from a token list with explicit position offset."""
    n = len(tokens)
    batch = C.llama_batch_init(n, 0, 1)
    batch.n_tokens = n

    for i, tok in enumerate(tokens):
        batch.token[i] = tok
        batch.pos[i] = pos_start + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = 1 if (logits_last and i == n - 1) else 0

    return batch


def make_batch_single(token: int, pos: int) -> C.llama_batch:
    """Create a single-token batch for autoregressive generation."""
    batch = C.llama_batch_init(1, 0, 1)
    batch.n_tokens = 1
    batch.token[0] = token
    batch.pos[0] = pos
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = 0
    batch.logits[0] = 1
    return batch
