# ik-llama-cpp-python

[![PyPI version](https://img.shields.io/pypi/v/ik-llama-cpp-python)](https://pypi.org/project/ik-llama-cpp-python/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ik-llama-cpp-python)](https://pypi.org/project/ik-llama-cpp-python/)
[![License: MIT](https://img.shields.io/pypi/l/ik-llama-cpp-python)](https://github.com/gongpx20069/ik-llama-cpp-python/blob/main/LICENSE)

Python bindings for [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) — a high-performance fork of llama.cpp with faster CPU inference, novel quantization types (Trellis / IQK quants), and AVX-VNNI / AVX-512 optimizations.

Designed as a **drop-in replacement** for [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python).

## Installation

### Pre-built wheels (CPU, with AVX2)

```bash
pip install ik-llama-cpp-python
```

### Pre-built wheels (CUDA)

```bash
pip install ik-llama-cpp-python-cuda
```

<details>
<summary><strong>From source (requires CMake ≥ 3.21 and a C++20 compiler)</strong></summary>

```bash
git clone https://github.com/gongpx20069/ik-llama-cpp-python
cd ik-llama-cpp-python
git submodule update --init --recursive
pip install -e .
```

</details>

<details>
<summary><strong>From source with CUDA</strong></summary>

```bash
CMAKE_ARGS="-DGGML_CUDA=ON" pip install -e .
```

</details>

<details>
<summary><strong>From source with native CPU optimizations</strong></summary>

For maximum performance on your specific CPU (AVX-512, AVX-VNNI, etc.):

```bash
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install -e .
```

</details>

## Quick Start

```python
from ik_llama_cpp import IkLlama

llm = IkLlama("model.gguf", n_ctx=4096)

# Simple chat
text = llm.chat("What is the theory of relativity?")
print(text)
```

## API

### `create_chat_completion` — OpenAI-compatible

Returns a dict matching the `llama_cpp.Llama.create_chat_completion` schema.

```python
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    temperature=0.3,
    max_tokens=256,
)
print(response["choices"][0]["message"]["content"])
print(response["usage"])
```

### `chat` — Convenience wrapper

```python
text = llm.chat("Explain quantum mechanics in one sentence.")
```

### `generate` — Low-level token generation

```python
tokens = llm.tokenize("Hello, world!")
output_ids = llm.generate(tokens, max_tokens=128, temperature=0.7)
text = llm.detokenize(output_ids)
```

### Drop-in replacement for llama-cpp-python

```python
# Change this:
# from llama_cpp import Llama
# To this:
from ik_llama_cpp import IkLlama as Llama

llm = Llama("model.gguf", n_ctx=4096, flash_attn=True)
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | *required* | Path to GGUF model file |
| `n_ctx` | `int` | `4096` | Context window size |
| `n_threads` | `int` | `0` | CPU threads (0 = auto) |
| `use_mmap` | `bool` | `True` | Memory-map model file |
| `use_mlock` | `bool` | `False` | Lock model in RAM |
| `flash_attn` | `bool` | `True` | Enable flash attention |
| `n_gpu_layers` | `int` | `0` | Number of layers to offload to GPU |
| `verbose` | `bool` | `True` | Logging verbosity |

## Supported Platforms

| Platform | Wheels | Notes |
|----------|--------|-------|
| Linux x86_64 | CPU (AVX2), CUDA 12.4 | Python 3.10–3.13 |
| Linux aarch64 | CPU | Python 3.10–3.13 |
| macOS arm64 | CPU + Metal | Python 3.10–3.13 |
| Windows x86_64 | CPU (AVX2) | Python 3.10–3.13 |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `IK_LLAMA_CPP_LIB_PATH` | Override path to the compiled shared library |
| `CMAKE_ARGS` | Extra CMake flags for source builds |

## Why ik_llama.cpp?

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) is a llama.cpp fork focused on performance and quantization research. Key advantages:

- **Faster CPU inference** — improved prompt processing across all quantization types, better Flash Attention token generation
- **Novel quantization types** — Trellis quants (`IQ1_KT`–`IQ4_KT`), IQK quants (`IQ2_K`–`IQ6_K`), row-interleaved R4 variants, MXFP4
- **Better KV cache** — `Q8_KV` / `Q4_0` KV-cache quantization with Hadamard transforms
- **DeepSeek optimizations** — FlashMLA (v1–v3), fused MoE operations, Smart Expert Reduction
- **Hardware support** — optimized kernels for AVX2, AVX-512, AVX-VNNI, ARM NEON, CUDA (Turing+)
- **Broad model support** — LLaMA-3/4, Qwen3, DeepSeek-V3, Gemma3/4, Mistral, and many more

## Architecture

```
ik_llama_cpp/
  __init__.py        # Public API: IkLlama
  _lib_loader.py     # Finds and loads the shared library (.dll/.so/.dylib)
  _ctypes_api.py     # Low-level ctypes bindings to llama.h C API
  _internals.py      # RAII wrappers: IkModel, IkContext, IkSampler
  llama.py           # High-level IkLlama class
  lib/               # Compiled shared libraries (installed by CMake)
```

## License

MIT
