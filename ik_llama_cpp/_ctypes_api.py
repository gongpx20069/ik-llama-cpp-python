"""Low-level ctypes bindings for the ik_llama.cpp C API (llama.h).

NOTE: ik_llama.cpp uses an older-style API compared to upstream llama.cpp:
  - Sampling is direct (llama_sample_*) instead of sampler-chain objects
  - Model free is llama_free_model (not llama_model_free)
  - Timings use llama_get_timings / llama_print_timings / llama_reset_timings
  - Struct layouts differ significantly
"""

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
# Structures (matching ik_llama.cpp fork layouts)
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
    """ik_llama.cpp llama_model_params — differs from upstream."""
    _fields_ = [
        ("devices", ctypes.c_char_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("mla", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("max_gpu", ctypes.c_int32),
        ("ncmoe", ctypes.c_int32),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("max_ctx_size", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_int32),
        ("n_ubatch", ctypes.c_int32),
        ("amb", ctypes.c_int32),
        ("fit_margin", ctypes.c_int32),
        ("fit", ctypes.c_bool),
        ("worst_graph_tokens", ctypes.c_int32),
        ("type_k_first", ctypes.c_int),
        ("type_k_last", ctypes.c_int),
        ("type_v_first", ctypes.c_int),
        ("type_v_last", ctypes.c_int),
        ("n_k_first", ctypes.c_int32),
        ("n_k_last", ctypes.c_int32),
        ("n_v_first", ctypes.c_int32),
        ("n_v_last", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("rpc_servers", ctypes.c_char_p),
        ("progress_callback", llama_progress_callback),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(llama_model_kv_override)),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("repack_tensors", ctypes.c_bool),
        ("use_thp", ctypes.c_bool),
        ("validate_quants", ctypes.c_bool),
        ("merge_qkv", ctypes.c_bool),
        ("merge_up_gate_exps", ctypes.c_bool),
        ("mtp", ctypes.c_bool),
        ("dry_run", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
    ]


class llama_context_params(ctypes.Structure):
    """ik_llama.cpp llama_context_params — differs from upstream."""
    _fields_ = [
        ("seed", ctypes.c_uint32),
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_uint32),
        ("n_threads_batch", ctypes.c_uint32),
        ("max_extra_alloc", ctypes.c_int32),
        ("worst_case_tokens", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
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
        ("type_reduce", ctypes.c_int),
        ("type_k_first", ctypes.c_int),
        ("type_k_last", ctypes.c_int),
        ("type_v_first", ctypes.c_int),
        ("type_v_last", ctypes.c_int),
        ("n_k_first", ctypes.c_int32),
        ("n_k_last", ctypes.c_int32),
        ("n_v_first", ctypes.c_int32),
        ("n_v_last", ctypes.c_int32),
        ("logits_all", ctypes.c_bool),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("mla_attn", ctypes.c_int),
        ("attn_max_batch", ctypes.c_int),
        ("fused_moe_up_gate", ctypes.c_bool),
        ("grouped_expert_routing", ctypes.c_bool),
        ("fused_up_gate", ctypes.c_bool),
        ("fused_mmad", ctypes.c_bool),
        ("rope_cache", ctypes.c_bool),
        ("graph_reuse", ctypes.c_bool),
        ("min_experts", ctypes.c_int),
        ("thresh_experts", ctypes.c_float),
        ("only_active_experts", ctypes.c_bool),
        ("k_cache_hadamard", ctypes.c_bool),
        ("v_cache_hadamard", ctypes.c_bool),
        ("split_mode_graph_scheduling", ctypes.c_bool),
        ("scheduler_async", ctypes.c_bool),
        ("mtp", ctypes.c_bool),
        ("mtp_op_type", ctypes.c_int),
        ("abort_callback", ggml_abort_callback),
        ("abort_callback_data", ctypes.c_void_p),
        ("offload_policy", ctypes.c_void_p),
        ("cuda_params", ctypes.c_void_p),
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
        ("all_pos_0", llama_pos),
        ("all_pos_1", llama_pos),
        ("all_seq_id", llama_seq_id),
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


class llama_timings(ctypes.Structure):
    """ik_llama.cpp uses llama_timings instead of llama_perf_context_data."""
    _fields_ = [
        ("t_start_ms", ctypes.c_double),
        ("t_end_ms", ctypes.c_double),
        ("t_load_ms", ctypes.c_double),
        ("t_sample_ms", ctypes.c_double),
        ("t_p_eval_ms", ctypes.c_double),
        ("t_eval_ms", ctypes.c_double),
        ("n_sample", ctypes.c_int32),
        ("n_p_eval", ctypes.c_int32),
        ("n_eval", ctypes.c_int32),
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


# -- Model load / free --

@_cfunc("llama_model_load_from_file", [ctypes.c_char_p, llama_model_params], ctypes.c_void_p)
def llama_model_load_from_file(path: bytes, params: llama_model_params) -> Optional[int]: ...


@_cfunc("llama_free_model", [ctypes.c_void_p], None)
def llama_free_model(model: int) -> None: ...


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


# -- Direct sampling (ik_llama.cpp style — no sampler chain) --

@_cfunc(
    "llama_sample_top_k",
    [ctypes.c_void_p, ctypes.POINTER(llama_token_data_array), ctypes.c_int32, ctypes.c_size_t],
    None,
)
def llama_sample_top_k(ctx: int, candidates: Any, k: int, min_keep: int) -> None: ...


@_cfunc(
    "llama_sample_top_p",
    [ctypes.c_void_p, ctypes.POINTER(llama_token_data_array), ctypes.c_float, ctypes.c_size_t],
    None,
)
def llama_sample_top_p(ctx: int, candidates: Any, p: float, min_keep: int) -> None: ...


@_cfunc(
    "llama_sample_temp",
    [ctypes.c_void_p, ctypes.POINTER(llama_token_data_array), ctypes.c_float],
    None,
)
def llama_sample_temp(ctx: int, candidates: Any, temp: float) -> None: ...


@_cfunc(
    "llama_sample_softmax",
    [ctypes.c_void_p, ctypes.POINTER(llama_token_data_array)],
    None,
)
def llama_sample_softmax(ctx: int, candidates: Any) -> None: ...


@_cfunc(
    "llama_sample_token_greedy",
    [ctypes.c_void_p, ctypes.POINTER(llama_token_data_array)],
    llama_token,
)
def llama_sample_token_greedy(ctx: int, candidates: Any) -> int: ...


@_cfunc(
    "llama_sample_token",
    [ctypes.c_void_p, ctypes.POINTER(llama_token_data_array)],
    llama_token,
)
def llama_sample_token(ctx: int, candidates: Any) -> int: ...


# -- Timings (ik_llama.cpp style) --

@_cfunc("llama_get_timings", [ctypes.c_void_p], llama_timings)
def llama_get_timings(ctx: int) -> llama_timings: ...


@_cfunc("llama_print_timings", [ctypes.c_void_p], None)
def llama_print_timings(ctx: int) -> None: ...


@_cfunc("llama_reset_timings", [ctypes.c_void_p], None)
def llama_reset_timings(ctx: int) -> None: ...
