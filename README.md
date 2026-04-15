# ik-llama-cpp-python

Python bindings for [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) — a high-performance fork of llama.cpp with better CPU inference, novel quantization types (IQ4_KT, Trellis quants), and AVX-VNNI optimizations.

## Install

### From source (requires CMake and a C++ compiler)

```bash
git clone https://github.com/peixiangong/ik-llama-cpp-python
cd ik-llama-cpp-python

# Pull ik_llama.cpp source
git submodule update --init --recursive

# Build and install
pip install -e .
```

### With CUDA support

```bash
CMAKE_ARGS="-DGGML_CUDA=ON" pip install -e .
```

### Pre-built wheels (planned)

```bash
pip install ik-llama-cpp-python
```

## Usage

### Basic

```python
from ik_llama_cpp import IkLlama

llm = IkLlama("models/gemma4-e2b.gguf", n_ctx=4096)

# Simple chat
text = llm.chat("What is the theory of relativity?")
print(text)

# With stats
text, stats = llm.chat_with_stats("Explain quantum mechanics.")
print(f"Response: {text}")
print(f"Eval TPS: {stats['eval_tps']}")
```

### OpenAI-compatible API

```python
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.3,
    max_tokens=256,
)
print(response["choices"][0]["message"]["content"])
print(response["usage"])
```

### Drop-in replacement for llama-cpp-python

```python
# In your existing code, change:
# from llama_cpp import Llama
from ik_llama_cpp import IkLlama as Llama

llm = Llama("model.gguf", n_ctx=4096, flash_attn=True)
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to GGUF model file |
| `n_ctx` | int | 4096 | Context window size |
| `n_threads` | int | 0 | CPU threads (0 = auto) |
| `use_mmap` | bool | True | Memory-map model file |
| `use_mlock` | bool | False | Lock model in RAM |
| `flash_attn` | bool | True | Enable flash attention |
| `n_gpu_layers` | int | 0 | Layers to offload to GPU |
| `verbose` | bool | True | Logging verbosity |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `IK_LLAMA_CPP_LIB_PATH` | Override path to the shared library |
| `CMAKE_ARGS` | Extra CMake flags for building |

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

## Why ik_llama.cpp?

Compared to upstream llama.cpp:
- Faster CPU prompt processing across all quantization types
- Novel Trellis quantization (IQ4_KT) for better quality at same size
- AVX-VNNI and improved ARM NEON support
- Better KV cache quantization via Hadamard transforms
- FlashMLA for DeepSeek models on CPU

## License

MIT
