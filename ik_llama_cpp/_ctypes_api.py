"""Low-level ctypes bindings for the ik_llama.cpp C API (llama.h)."""

from __future__ import annotations

import ctypes
import functools
from typing import Any, Callable, List, Optional, TypeVar

from ._lib_loader import load_shared_library

# ---------------------------------------------------------------------------
# Load shared library
# ---------------------------------------------------------------------------

_lib = load_shared_library()

# ---------------------------------------------------------------------------
# Decorator — binds a Python stub to a C symbol in the shared library
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def _cfunc(name: str, argtypes: List[Any], restype: Any):
    def decorator(f: F) -> F:
        func = getattr(_lib, name)
        func.argtypes = argtypes
        func.restype = restype
        functools.wraps(f)(func)
        return func  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

llama_model_p = ctypes.c_void_p
llama_context_p = ctypes.c_void_p
llama_vocab_p = ctypes.c_void_p
llama_sampler_p = ctypes.POINTER(ctypes.c_void_p)  # opaque

llama_token = ctypes.c_int32
llama_token_p = ctypes.POINTER(llama_token)
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

# Callback types
llama_progress_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_float, ctypes.c_void_p
)
ggml_backend_sched_eval_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p
)
ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)

# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


class llama_model_kv_override_value(ctypes.Union):
    _fields_ = [
        ("val_i64", ctypes.c_int64),
        ("val_f64", ctypes.c_double),
        ("val_bool", ctypes.c_bool),
        ("val_str", ctypes.c_char * 128),
    ]


class llama_model_kv_override(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_int),
        ("key", ctypes.c_char * 128),
        ("value", llama_model_kv_override_value),
    ]


class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", llama_progress_callback),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(llama_model_kv_override)),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]


class llama_sampler_seq_config(ctypes.Structure):
    _fields_ = [
        ("seq_id", llama_seq_id),
        ("sampler", ctypes.c_void_p),
    ]


class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ggml_backend_sched_eval_callback),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("abort_callback", ggml_abort_callback),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(llama_sampler_seq_config)),
        ("n_samplers", ctypes.c_size_t),
    ]


class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


class llama_token_data(ctypes.Structure):
    _fields_ = [
        ("id", llama_token),
        ("logit", ctypes.c_float),
        ("p", ctypes.c_float),
    ]


class llama_token_data_array(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(llama_token_data)),
        ("size", ctypes.c_size_t),
        ("selected", ctypes.c_int64),
        ("sorted", ctypes.c_bool),
    ]


class llama_sampler_chain_params(ctypes.Structure):
    _fields_ = [
        ("no_perf", ctypes.c_bool),
    ]


class llama_perf_context_data(ctypes.Structure):
    _fields_ = [
        ("t_start_ms", ctypes.c_double),
        ("t_load_ms", ctypes.c_double),
        ("t_p_eval_ms", ctypes.c_double),
        ("t_eval_ms", ctypes.c_double),
        ("n_p_eval", ctypes.c_int32),
        ("n_eval", ctypes.c_int32),
        ("n_reused", ctypes.c_int32),
    ]


# ---------------------------------------------------------------------------
# Function bindings
# ---------------------------------------------------------------------------

# -- Backend lifecycle --

@_cfunc("llama_backend_init", [], None)
def llama_backend_init() -> None: ...


@_cfunc("llama_backend_free", [], None)
def llama_backend_free() -> None: ...


# -- Default params --

@_cfunc("llama_model_default_params", [], llama_model_params)
def llama_model_default_params() -> llama_model_params: ...


@_cfunc("llama_context_default_params", [], llama_context_params)
def llama_context_default_params() -> llama_context_params: ...


@_cfunc("llama_sampler_chain_default_params", [], llama_sampler_chain_params)
def llama_sampler_chain_default_params() -> llama_sampler_chain_params: ...


# -- Model load / free --

@_cfunc("llama_model_load_from_file", [ctypes.c_char_p, llama_model_params], ctypes.c_void_p)
def llama_model_load_from_file(path: bytes, params: llama_model_params) -> Optional[int]: ...


@_cfunc("llama_model_free", [ctypes.c_void_p], None)
def llama_model_free(model: int) -> None: ...


# -- Context init / free --

@_cfunc("llama_init_from_model", [ctypes.c_void_p, llama_context_params], ctypes.c_void_p)
def llama_init_from_model(model: int, params: llama_context_params) -> Optional[int]: ...


@_cfunc("llama_free", [ctypes.c_void_p], None)
def llama_free(ctx: int) -> None: ...


# -- Vocab --

@_cfunc("llama_model_get_vocab", [ctypes.c_void_p], ctypes.c_void_p)
def llama_model_get_vocab(model: int) -> Optional[int]: ...


@_cfunc("llama_vocab_n_tokens", [ctypes.c_void_p], ctypes.c_int32)
def llama_vocab_n_tokens(vocab: int) -> int: ...


# -- Tokenize / Detokenize --

@_cfunc(
    "llama_tokenize",
    [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
     llama_token_p, ctypes.c_int32, ctypes.c_bool, ctypes.c_bool],
    ctypes.c_int32,
)
def llama_tokenize(
    vocab: int, text: bytes, text_len: int,
    tokens: Any, n_tokens_max: int,
    add_special: bool, parse_special: bool,
) -> int: ...


@_cfunc(
    "llama_token_to_piece",
    [ctypes.c_void_p, llama_token, ctypes.c_char_p,
     ctypes.c_int32, ctypes.c_int32, ctypes.c_bool],
    ctypes.c_int32,
)
def llama_token_to_piece(
    vocab: int, token: int, buf: Any,
    length: int, lstrip: int, special: bool,
) -> int: ...


# -- Batch --

@_cfunc("llama_batch_init", [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], llama_batch)
def llama_batch_init(n_tokens: int, embd: int, n_seq_max: int) -> llama_batch: ...


@_cfunc("llama_batch_free", [llama_batch], None)
def llama_batch_free(batch: llama_batch) -> None: ...


# -- Decode --

@_cfunc("llama_decode", [ctypes.c_void_p, llama_batch], ctypes.c_int32)
def llama_decode(ctx: int, batch: llama_batch) -> int: ...


# -- Logits --

@_cfunc("llama_get_logits_ith", [ctypes.c_void_p, ctypes.c_int32], ctypes.POINTER(ctypes.c_float))
def llama_get_logits_ith(ctx: int, i: int) -> Any: ...


# -- Sampler --

@_cfunc("llama_sampler_chain_init", [llama_sampler_chain_params], ctypes.c_void_p)
def llama_sampler_chain_init(params: llama_sampler_chain_params) -> int: ...


@_cfunc("llama_sampler_chain_add", [ctypes.c_void_p, ctypes.c_void_p], None)
def llama_sampler_chain_add(chain: int, smpl: int) -> None: ...


@_cfunc("llama_sampler_sample", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32], llama_token)
def llama_sampler_sample(smpl: int, ctx: int, idx: int) -> int: ...


@_cfunc("llama_sampler_free", [ctypes.c_void_p], None)
def llama_sampler_free(smpl: int) -> None: ...


@_cfunc("llama_sampler_init_greedy", [], ctypes.c_void_p)
def llama_sampler_init_greedy() -> int: ...


@_cfunc("llama_sampler_init_temp", [ctypes.c_float], ctypes.c_void_p)
def llama_sampler_init_temp(t: float) -> int: ...


@_cfunc("llama_sampler_init_top_k", [ctypes.c_int32], ctypes.c_void_p)
def llama_sampler_init_top_k(k: int) -> int: ...


@_cfunc("llama_sampler_init_top_p", [ctypes.c_float, ctypes.c_size_t], ctypes.c_void_p)
def llama_sampler_init_top_p(p: float, min_keep: int) -> int: ...


# -- Perf --

@_cfunc("llama_perf_context", [ctypes.c_void_p], llama_perf_context_data)
def llama_perf_context(ctx: int) -> llama_perf_context_data: ...


@_cfunc("llama_perf_context_print", [ctypes.c_void_p], None)
def llama_perf_context_print(ctx: int) -> None: ...


@_cfunc("llama_perf_context_reset", [ctypes.c_void_p], None)
def llama_perf_context_reset(ctx: int) -> None: ...
